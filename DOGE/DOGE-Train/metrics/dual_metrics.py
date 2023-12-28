import torch
from torchmetrics import Metric
from torch_scatter import scatter_mean
import numpy as np
import os, time

class DualMetrics(Metric):
    def __init__(self, num_rounds, num_dual_iter_per_round, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        default = torch.zeros(num_rounds + 1, dtype=torch.float32) # 0-th round index is for initial state.
        self.num_dual_iter_per_round = num_dual_iter_per_round
        self.add_state('time_spent_per_round_sum', default=torch.zeros(num_rounds + 1, dtype=torch.float64), dist_reduce_fx="max")
        self.add_state('loss', default=default, dist_reduce_fx="sum")
        self.add_state('pred_lb_sums', default=default, dist_reduce_fx="sum")
        self.add_state('pred_clip_lb_sums', default=default, dist_reduce_fx="sum")
        self.add_state('gt_obj_sums', default=default, dist_reduce_fx="sum")
        self.add_state('rel_gap_sums', default=default, dist_reduce_fx="sum")
        self.add_state('rel_gap_clip_sums', default=default, dist_reduce_fx="sum")
        self.add_state('total_instances', default=default + 1e-7, dist_reduce_fx="sum")
        self.add_state('num_gt_known', default = torch.zeros(1, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state('num_gt_time_known', default = torch.zeros(1, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state('loss_known', default = torch.ones(1, dtype=torch.bool), dist_reduce_fx="min")
        self.add_state('max_round', default = torch.zeros(1, dtype=torch.int32), dist_reduce_fx="max")
        self.add_state('gt_compute_time_sum', default = torch.zeros(1, dtype=torch.float32), dist_reduce_fx="max")
        self.max_lb_per_instance = {}

    def update(self, batch, logs):
        with torch.no_grad():
            initial_lbs = []
            max_pred_lb = np.NINF
            min_pred_rel_gap = np.inf
            prev_time = 0
            for (i, logs_round) in enumerate(logs):
                round = logs_round['r']
                time = logs_round['t']
                if round > 0:
                    self.time_spent_per_round_sum[round] += time - prev_time
                prev_time = time
                if 'loss' in logs_round:
                    self.loss[round] += logs_round['loss']
                    self.loss_known[0] = True
                else:
                    self.loss_known[0] = False

                self.max_round[0] = max(self.max_round[0].item(), round)
                bdd_start = 0
                for b, (gt_obj, num_bdds) in enumerate(zip(batch.gt_info['lp_stats']['obj'], batch.num_cons)):
                    current_lb = (logs_round['lb_per_instance'][b] / batch.obj_multiplier[b]) + batch.obj_offset[b]
                    max_pred_lb = max(max_pred_lb, current_lb)
                    if round == 0:
                        initial_lbs.append(current_lb)
                    self.pred_lb_sums[round] += current_lb
                    self.pred_clip_lb_sums[round] += max_pred_lb
                    self.total_instances[round] += 1.0
                    lb_obj_type = True
                    if 'obj_type' in batch.gt_info['lp_stats']:
                        lb_obj_type = batch.gt_info['lp_stats']['obj_type'][b] != 'primal_obj'

                    if gt_obj is not None and lb_obj_type:
                        self.gt_obj_sums[round] += gt_obj
                        self.rel_gap_sums[round] += (gt_obj - current_lb) / (gt_obj - initial_lbs[b])
                        # self.rel_gap_clip_sums[round] += max((gt_obj - max_pred_lb) / (gt_obj - initial_lbs[b]), 0.0)
                        self.rel_gap_clip_sums[round] += (gt_obj - max_pred_lb) / (gt_obj - initial_lbs[b])
                        if i == 0:
                            self.num_gt_known[0] += 1
                            if batch.gt_info['lp_stats']['time'][b] is not None:
                                self.gt_compute_time_sum[0] += batch.gt_info['lp_stats']['time'][b]
                                self.num_gt_time_known[0] += 1

                    bdd_start += num_bdds
                    file_name = os.path.basename(batch.file_path[b])
                    if file_name in self.max_lb_per_instance:
                        self.max_lb_per_instance[file_name] = max(self.max_lb_per_instance[file_name], current_lb.item())
                    else:
                        self.max_lb_per_instance[file_name] = current_lb.item()

    def compute(self):
        # compute final result
        merged_results = {}
        num_valid_rounds = self.max_round[0].item() + 1
        if self.num_gt_known[0].item() > 0:
            gt_obj_mean = self.gt_obj_sums / self.num_gt_known
            rel_gap_mean = self.rel_gap_sums / self.num_gt_known
            rel_gap_clip_mean = self.rel_gap_clip_sums / self.num_gt_known
        if self.num_gt_time_known[0].item() == self.num_gt_known[0].item():
            gt_time_mean = self.gt_compute_time_sum / self.num_gt_time_known
        else:
            gt_time_mean = None

        pred_lb_mean = self.pred_lb_sums / self.total_instances
        pred_lb_clip_mean = self.pred_clip_lb_sums / self.total_instances
        time_spent_mean = self.time_spent_per_round_sum / self.total_instances
        lower_bounds = {}
        gaps = {}
        gaps_clipped = {}
        time_spent = {}
        time_acc = time.time()
        for r in range(num_valid_rounds):
            tag = f'itr_{r * self.num_dual_iter_per_round}_time_{time_spent_mean[r] + time_acc}'
            time_acc += time_spent_mean[r]
            lower_bounds[f'pred_{tag}'] = pred_lb_mean[r]
            lower_bounds[f'pred_clip_{tag}'] = pred_lb_clip_mean[r]
            time_spent[f'_{tag}'] = time_spent_mean[r]
            if self.loss_known:
                if not 'loss' in merged_results:
                    merged_results['loss'] = {}
                merged_results['loss'].update({tag: self.loss[r]})
            if self.num_gt_known[0].item() > 0:
                lower_bounds[f'gt_{tag}'] = gt_obj_mean[r]
                gaps[f'_{tag}'] = rel_gap_mean[r]
                gaps_clipped[f'_{tag}'] = rel_gap_clip_mean[r]
        merged_results['lower_bounds'] = lower_bounds
        merged_results['time_per_round'] = time_spent
        if self.num_gt_known[0].item() > 0:
            merged_results['relative_gaps'] = gaps
            merged_results['relative_gaps_clipped'] = gaps_clipped
        return merged_results, self.max_lb_per_instance, gt_time_mean