import torch
from torch_scatter.scatter import scatter_mean, scatter_sum
from pytorch_lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam, SGD
import logging, os, time
import numpy as np
from model.model import FeatureExtractor, DOGEPredictor
import model.solver_utils as sol_utils
from metrics.dual_metrics import DualMetrics 
from data.replay_buffer import ReplayBuffer
from pytorch_lightning.utilities import grad_norm

class DOGE(LightningModule):
    def __init__(self, 
                num_train_rounds: int,
                num_train_rounds_with_grad: int,
                num_test_rounds: int,
                num_dual_iter_train: int,
                num_dual_iter_test: int,
                dual_improvement_slope_train: float,
                dual_improvement_slope_test: float,
                grad_dual_itr_max_itr: int,
                lr: float,
                loss_discount_factor:float,
                omega_initial: float,
                var_lp_features: List[str],
                con_lp_features: List[str],
                edge_lp_features: List[str],
                var_lp_features_init: List[str],
                con_lp_features_init: List[str],
                edge_lp_features_init: List[str],
                num_learned_var_f: int, 
                num_learned_con_f: int, 
                num_learned_edge_f: int,
                num_hidden_layers_edge: int, 
                feature_extractor_depth: int,
                dual_predictor_depth: int,
                optimizer_name: str,
                start_episodic_training_after_epoch: int,
                val_fraction: List[int],
                use_layer_norm: True,
                use_net_solver_costs: False,
                free_update_loss_weight: Optional[float] = 0.0, 
                num_journeys: Optional[int] = 10,
                use_lstm_var: Optional[bool] = False,
                use_rel_gap_loss: Optional[bool] = False,
                predict_dist_weights: Optional[bool] = True,
                predict_omega: Optional[bool] = True,
                free_update: Optional[bool] = False,
                scale_free_update: Optional[bool] = False,
                denormalize_free_update: Optional[bool] = False,
                use_celu_activation: Optional[bool] = False,
                mp_aggr: Optional[str] = 'mean',
                use_separate_model_later_stage: Optional[bool] = False,
                val_datanames: Optional[List[str]] = None,
                test_datanames: Optional[List[str]] = None,
                non_learned_updates_test = False,
                only_test_non_learned = False,
                use_replay_buffer = False,
                use_solver_costs_as_feature: Optional[bool] = True
                ):
        super(DOGE, self).__init__()
        self.save_hyperparameters()
        self.dual_block = DOGEPredictor(
                        var_lp_f_names = var_lp_features,
                        con_lp_f_names = con_lp_features, 
                        edge_lp_f_names = edge_lp_features,
                        depth = dual_predictor_depth,
                        var_dim = num_learned_var_f, 
                        con_dim = num_learned_con_f,
                        edge_dim = num_learned_edge_f,
                        use_layer_norm = use_layer_norm,
                        predict_dist_weights = predict_dist_weights,
                        predict_omega = predict_omega,
                        num_hidden_layers_edge = num_hidden_layers_edge,
                        use_net_solver_costs = use_net_solver_costs,
                        use_lstm_var = use_lstm_var,
                        free_update = free_update,
                        history_num_itr = num_dual_iter_train,
                        free_update_loss_weight = free_update_loss_weight,
                        denormalize_free_update = denormalize_free_update,
                        scale_free_update = scale_free_update,
                        use_celu_activation = use_celu_activation,
                        aggr = mp_aggr,
                        use_solver_costs = use_solver_costs_as_feature)
        if use_separate_model_later_stage:
            self.dual_block_later = DOGEPredictor(
                                    var_lp_f_names = var_lp_features,
                                    con_lp_f_names = con_lp_features, 
                                    edge_lp_f_names = edge_lp_features,
                                    depth = dual_predictor_depth,
                                    var_dim = num_learned_var_f, 
                                    con_dim = num_learned_con_f,
                                    edge_dim = num_learned_edge_f,
                                    use_layer_norm = use_layer_norm,
                                    predict_dist_weights = predict_dist_weights,
                                    predict_omega = predict_omega,
                                    num_hidden_layers_edge = num_hidden_layers_edge,
                                    use_net_solver_costs = use_net_solver_costs,
                                    use_lstm_var = use_lstm_var,
                                    free_update = free_update,
                                    history_num_itr = num_dual_iter_train,
                                    free_update_loss_weight = free_update_loss_weight,
                                    denormalize_free_update = denormalize_free_update,
                                    scale_free_update = scale_free_update,
                                    use_celu_activation = use_celu_activation,
                                    aggr = mp_aggr,
                                    use_solver_costs = use_solver_costs_as_feature)

        self.val_datanames = val_datanames
        self.test_datanames = test_datanames
        self.console_logger = logging.getLogger('lightning')
        self.train_log_every_n_epoch = 5
        self.train_metrics = DualMetrics(num_train_rounds, num_dual_iter_train)

        self.eval_metrics_val = torch.nn.ModuleDict()
        self.eval_metrics_val_non_learned = torch.nn.ModuleDict()
        self.non_learned_updates_val = True
        for data_name in val_datanames:
            self.eval_metrics_val[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
            self.eval_metrics_val_non_learned[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)

        self.non_learned_updates_test = non_learned_updates_test
        self.only_test_non_learned = only_test_non_learned
        self.eval_metrics_test = torch.nn.ModuleDict()
        self.eval_metrics_test_non_learned = torch.nn.ModuleDict()
        self.logged_hparams = False
        for data_name in test_datanames:
            self.eval_metrics_test[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
            self.eval_metrics_test_non_learned[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
        self.replay_buffer = {} if use_replay_buffer else None
        self.num_rounds = 1

    @classmethod
    def from_config(cls, cfg, val_datanames, test_datanames, num_test_rounds, num_dual_iter_test, dual_improvement_slope_test, non_learned_updates_test, only_test_non_learned):
        return cls(
            num_train_rounds = cfg.TRAIN.NUM_ROUNDS,
            num_train_rounds_with_grad = cfg.TRAIN.NUM_ROUNDS_WITH_GRAD,
            num_test_rounds = num_test_rounds,
            var_lp_features = cfg.MODEL.VAR_LP_FEATURES,
            con_lp_features = cfg.MODEL.CON_LP_FEATURES,
            edge_lp_features = cfg.MODEL.EDGE_LP_FEATURES,
            var_lp_features_init = cfg.MODEL.VAR_LP_FEATURES_INIT,
            con_lp_features_init = cfg.MODEL.CON_LP_FEATURES_INIT,
            edge_lp_features_init = cfg.MODEL.EDGE_LP_FEATURES_INIT,
            num_hidden_layers_edge = cfg.MODEL.NUM_HIDDEN_LAYERS_EDGE,
            use_lstm_var = cfg.MODEL.USE_LSTM_VAR,
            num_dual_iter_train = cfg.TRAIN.NUM_DUAL_ITERATIONS,
            num_dual_iter_test = num_dual_iter_test,
            use_layer_norm = cfg.MODEL.USE_LAYER_NORM,
            use_net_solver_costs = cfg.MODEL.USE_NET_SOLVER_COSTS,
            use_rel_gap_loss = cfg.TRAIN.USE_RELATIVE_GAP_LOSS,
            dual_improvement_slope_train = cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE,
            dual_improvement_slope_test = dual_improvement_slope_test,
            grad_dual_itr_max_itr = cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR,
            lr = cfg.TRAIN.BASE_LR,
            loss_discount_factor = cfg.TRAIN.LOSS_DISCOUNT_FACTOR,
            omega_initial = cfg.MODEL.OMEGA_INITIAL,
            free_update = cfg.MODEL.FREE_UPDATE,
            free_update_loss_weight = cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT,
            scale_free_update = cfg.MODEL.SCALE_FREE_UPDATE,
            denormalize_free_update = cfg.MODEL.DENORM_FREE_UPDATE,
            use_celu_activation = cfg.MODEL.USE_CELU_ACTIVATION,
            mp_aggr = cfg.MODEL.MP_AGGR,
            num_learned_var_f = cfg.MODEL.VAR_FEATURE_DIM, 
            num_learned_con_f = cfg.MODEL.CON_FEATURE_DIM,
            num_learned_edge_f = cfg.MODEL.EDGE_FEATURE_DIM,
            feature_extractor_depth = cfg.MODEL.FEATURE_EXTRACTOR_DEPTH,
            dual_predictor_depth = cfg.MODEL.DUAL_PRED_DEPTH,
            start_episodic_training_after_epoch = cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH,
            optimizer_name = cfg.TRAIN.OPTIMIZER,
            val_fraction = cfg.DATA.VAL_FRACTION,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            predict_dist_weights = cfg.MODEL.PREDICT_DIST_WEIGHTS,
            predict_omega = cfg.MODEL.PREDICT_OMEGA,
            non_learned_updates_test = non_learned_updates_test,
            only_test_non_learned = only_test_non_learned,
            num_journeys = cfg.TRAIN.NUM_JOURNEYS,
            use_separate_model_later_stage = cfg.MODEL.USE_SEPARATE_MODEL_LATER_STAGE,
            use_replay_buffer = cfg.TRAIN.USE_REPLAY_BUFFER,
            use_solver_costs_as_feature = cfg.MODEL.USE_SOLVER_COST_AS_FEATURE)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            print("Using Adam optimizer.")
            return Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer_name == 'SGD':
            print("Using SGD optimizer.")
            return SGD(self.parameters(), lr=self.hparams.lr, momentum = 0.9, dampening = 0.9)
        else:
            raise ValueError(f'Optimizer {self.hparams.optimizer_name} not exposed.')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        gt_type = None
        replay_buffer_start_round = 0
        if self.hparams.use_replay_buffer:
            # Samples much early in the trajectory for lstm (by taking square) since the cell state needs to be built to summarize history.
            replay_buffer_start_round = max(self.num_rounds - (self.hparams.num_train_rounds_with_grad * self.hparams.num_train_rounds_with_grad), 0)
        batch, dist_weights = sol_utils.init_solver_and_get_states(batch, device, gt_type,
                    self.hparams.var_lp_features_init, self.hparams.con_lp_features_init, self.hparams.edge_lp_features_init, 
                    self.hparams.num_dual_iter_train * 2, 0.0, self.hparams.omega_initial, 
                    distribute_deltaa = False, num_grad_iterations_dual_features = 0, 
                    compute_history_for_itrs = self.hparams.num_dual_iter_train, replay_buffer = self.replay_buffer, 
                    replay_buffer_start_round = replay_buffer_start_round, use_lstm_var = self.hparams.use_lstm_var,
                    num_learned_var_f = self.hparams.num_learned_var_f)
        batch.dist_weights = dist_weights
        batch.objective_dev = batch.objective.to(device).to(torch.float64)
        batch.initial_lb_per_instance = scatter_sum(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con)
        if batch.gt_info['lp_stats']['obj'][0] is not None:
            batch.gt_obj_normalized = (batch.gt_info['lp_stats']['obj'] - batch.obj_offset) * batch.obj_multiplier
            batch.gt_obj_normalized = batch.gt_obj_normalized.to(device)
            for (b, fp) in enumerate(batch.file_path):
                diff = batch.gt_obj_normalized[b] - batch.initial_lb_per_instance[b]
                assert diff > 1e-6, f"lb difference for file: {fp} = {diff} < 1e-6."
        return batch

    # Samples location in the optimization trajectory where solver will be trained. 
    # 1. We start training the solver at 'easy' paths first i.e., just at the start of optimization.
    # 2. Afterwards we start exposing the solver to difficult regions where proposed_start_step will be much larger than 0.
    # 3. Then we restart and start sampling again at easy regions and increase difficulty again.
    # 4. Step 3 is repeated 'num_journeys' times. 
    # Note that if 'use_replay_buffer' is set to True and assume we want to learn GNN at N-th iteration of the solver then
    # we will search in the replay buffer for an iterate (see line 199) instead of running the solver for N iterations. This cuts down a large
    # portion of training cost. However for LSTM based model the replay buffer does not help much since LSTM needs to build its 
    # cell state. Naively storing this cell state in replay buffer makes it out-of-date and gives worse results than not using 
    # replay buffer. 
    def compute_training_start_round(self):
        current_start_epoch = self.current_epoch - self.hparams.start_episodic_training_after_epoch
        max_training_epoch_mod = 1 + (self.trainer.max_epochs // self.hparams.num_journeys) # Divides num_journeys many journeys.
        fraction = float(current_start_epoch) / max_training_epoch_mod
        if self.current_epoch % max_training_epoch_mod == 0 and self.hparams.use_replay_buffer:
            print(f'Clearing replay buffer at epoch: {self.current_epoch}')
            self.replay_buffer = {}

        if fraction < 0: # Step 1. 
            return max(0, self.hparams.num_train_rounds_with_grad - 1)
        fraction = fraction % 1
        if not self.hparams.use_separate_model_later_stage:
            fraction = fraction * fraction
        mean_start_step = fraction * (self.hparams.num_train_rounds)
        proposed_start_step = np.round(np.random.normal(mean_start_step, self.hparams.num_train_rounds // 5)).astype(np.int32).item(0)
        proposed_start_step = max(max(min(proposed_start_step, self.hparams.num_train_rounds - 1), 0), self.hparams.num_train_rounds_with_grad - 1)
        if proposed_start_step < 3 and np.random.rand(1) > 0.5: # DOGE and DOGE-M on worms was computed by this additional randomization.
            return max(0, self.hparams.num_train_rounds_with_grad - 1)
        self.logger.experiment.add_scalar('train/start_grad_round', proposed_start_step, global_step = self.global_step)
        return proposed_start_step

    def log_metrics(self, metrics_calculator, mode, log_to_tb = True):
        metrics_dict, _, _ = metrics_calculator.compute()
        best_pred_lb = np.NINF
        last_itr = 0
        for metric_name, metric_value in metrics_dict.items():
            metrics_wo_time = {}
            for metric_name_itr_time, val in metric_value.items():
                name_itr, ctime = metric_name_itr_time.split('_time_')
                _, itr = name_itr.split('itr_')
                metrics_wo_time[name_itr] = val
                if 'lower_bounds' in metric_name and 'pred_' in name_itr and not 'pred_clip_' in name_itr: #and int(itr) == self.hparams.num_train_rounds * self.hparams.num_dual_iter_train:
                    best_pred_lb = max(val.item(), best_pred_lb)
            if log_to_tb:
                self.logger.experiment.add_scalars(f'{mode}/{metric_name}', metrics_wo_time, global_step = self.global_step, walltime = float(ctime))
        self.logger.experiment.flush()
        if np.isfinite(best_pred_lb):
            self.log('train_last_round_lb', best_pred_lb)
            if not self.logged_hparams:
                self.logger.log_hyperparams(self.hparams, {"train_last_round_lb": best_pred_lb})
                self.logged_hparams = True
        metrics_calculator.reset()
        return metrics_dict['loss']

    def log_metrics_test(self, metrics_calculator, prefix, suffix):
        assert('test' in prefix)
        metrics_dict, max_lb_per_instance, gt_time_mean = metrics_calculator.compute()
        print(f'\n{prefix}')
        for metric_name, metric_value in metrics_dict.items():
            print(f'\t {metric_name}_{suffix}: ')
            prev_value = None
            for name_itr, value in metric_value.items():
                # self.logger.experiment.add_scalar(f'{mode}/{metric_name}{suffix}', value, global_step = int(itr.replace('itr_', '')))
                name, itr_time = name_itr.split('itr_')
                itr, ctime = itr_time.split('_time_')
                self.logger.experiment.add_scalars(f'{prefix}/{metric_name}_{name}', {suffix: value}, global_step = int(itr), walltime = float(ctime))
                if prev_value is None or prev_value != value:
                    print(f'\t \t {name_itr:30}: {value}')
                prev_value = value

        if gt_time_mean is not None:
            self.logger.experiment.add_scalar(f'{prefix}/gt_lb_compute_time', gt_time_mean, global_step = 0)
            print(f'\t ground truth LB compute time: {gt_time_mean}')
        self.logger.experiment.flush()
        metrics_calculator.reset()
        return max_lb_per_instance

    # Override LM hook
    def on_before_optimizer_step(self, optimizer):
        # inspect (unscaled) gradients here
        grad_norm_dict = grad_norm(self, norm_type=2)
        self.logger.experiment.add_scalar('train/grad_norm', grad_norm_dict['grad_2.0_norm_total'], global_step = self.global_step)
        
    def log_dist_weights(self, dist_weights, data_name, edge_index_var_con, itr):
        var_indices = edge_index_var_con[0]
        dist_weights_mean = scatter_mean(dist_weights, var_indices)[var_indices]
        dist_weights_variance_per_var = scatter_mean(torch.square(dist_weights - dist_weights_mean), var_indices) / torch.numel(dist_weights)
        self.logger.experiment.add_histogram(f'{data_name}/std_dist_weights', dist_weights_variance_per_var, global_step = itr)

    def log_omega_vector(self, omega_vec, data_name, edge_index_var_con, itr):
        if omega_vec is None:
            return
        var_indices = edge_index_var_con[0]
        omega_vec_mean = scatter_mean(omega_vec, var_indices)
        self.logger.experiment.add_histogram(f'{data_name}/omega_vec_mean', omega_vec_mean, global_step = itr)
        omega_vec_variance_per_var = scatter_mean(torch.square(omega_vec - omega_vec_mean[var_indices]), var_indices) / torch.numel(omega_vec)
        self.logger.experiment.add_histogram(f'{data_name}/std_omega_vec', omega_vec_variance_per_var, global_step = itr)

    def on_train_epoch_start(self):
        self.num_rounds = self.compute_training_start_round() + 1

    def on_train_epoch_end(self):
        self.log_metrics(self.train_metrics, 'train', False)

    def on_validation_epoch_end(self):
        for data_name in self.val_datanames:
            if self.non_learned_updates_val: # Computes baseline via non-learned updates.
                self.log_metrics(self.eval_metrics_val_non_learned[data_name], f'val_{data_name}_non_learned')
            else:
                losses = self.log_metrics(self.eval_metrics_val[data_name], f'val_{data_name}')
                prev_itr = -1
                best_loss_value = 0
                for name_itr_time, value in losses.items():
                    # self.logger.experiment.add_scalar(f'{mode}/{metric_name}{suffix}', value, global_step = int(itr.replace('itr_', '')))
                    name, itr_time = name_itr_time.split('itr_')
                    itr, time = itr_time.split('_time_')
                    if (prev_itr < int(itr)):
                        prev_itr = int(itr)
                        best_loss_value = value
                self.log('val_loss', best_loss_value)

        self.non_learned_updates_val = False

    def on_test_epoch_end(self): # Logs per dataset metrics. Instance level metrics are logged already in test-step(..).
        for data_name in self.test_datanames:
            max_lb_per_inst_learned = self.log_metrics_test(self.eval_metrics_test[data_name], f'test_{data_name}', 'learned')
            if self.non_learned_updates_test:
                max_lb_per_inst_non_learned = self.log_metrics_test(self.eval_metrics_test_non_learned[data_name], f'test_{data_name}', 'non_learned')

    def try_concat_gt_edge_solution(self, batch, is_training):
        edge_sol = None
        for file_path, current_sol in zip(batch.file_path, batch.gt_sol_edge):
            if current_sol is None:
                assert current_sol is not None or not is_training, f'gt solution should be known for files: {file_path}'
                return edge_sol
        return torch.cat(batch.gt_sol_edge, 0)

    def dual_loss_lb(self, lb_after_dist, batch_index_con, initial_lb_per_instance = None, gt_obj_normalized = None):
        # Larger ILPs should have more impact on loss. 
        if not self.hparams.use_rel_gap_loss:
            return -lb_after_dist.sum()
        else:
            if lb_after_dist.requires_grad:
                assert gt_obj_normalized is not None
            elif gt_obj_normalized is None:
                return None

            numer = gt_obj_normalized - scatter_sum(lb_after_dist, batch_index_con)
            denom = gt_obj_normalized - initial_lb_per_instance
            rel_gap = 100.0 * torch.square(numer / (1e-4 + denom))  # Focus more on larger gaps so taking square.
            return rel_gap.sum()

    def single_dual_round(self, batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, 
                        return_best_dual = False, current_max_lb = None, best_solver_state = None, randomize_num_itrs = False, use_later_dual_block = False):
        lb_after_free_update = None
        randomize_num_itrs = False # Not randomizing until the implementation is good.
        dual_block = self.dual_block
        if use_later_dual_block:
            dual_block = self.dual_block_later
        new_solver_state, batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f, dist_weights, omega_vec, batch.var_hidden_states_lstm, lb_after_free_update = dual_block(
                                                                batch.solvers, batch.var_lp_f, batch.con_lp_f, 
                                                                batch.solver_state, batch.edge_rest_lp_f, 
                                                                batch.omega, batch.edge_index_var_con,
                                                                num_dual_iterations, grad_dual_itr_max_itr, improvement_slope, batch.valid_edge_mask,
                                                                batch.batch_index_var, batch.batch_index_con, batch.batch_index_edge, 
                                                                batch.objective_dev, batch.var_hidden_states_lstm, batch.dist_weights, randomize_num_itrs)

        new_solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
        if return_best_dual:
            new_lb = batch.solvers[0].lower_bound()
            assert len(batch.solvers) == 1
            if current_max_lb is None or new_lb > current_max_lb:
                return batch, dist_weights, omega_vec, new_solver_state, new_lb, lb_after_free_update
        batch.solver_state = new_solver_state

        return batch, dist_weights, omega_vec, best_solver_state, current_max_lb, lb_after_free_update

    def dual_rounds(self, batch, starting_round, num_rounds, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, is_training = False, instance_log_name = None, non_learned_updates = False, return_best_dual = False):
        loss = 0
        logs = [{'r' : 0, 'lb_per_instance': scatter_sum(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con), 't': time.time()}]
        gt_obj_normalized = None
        if 'gt_obj_normalized' in batch.keys():
            gt_obj_normalized = batch.gt_obj_normalized
        current_loss = self.dual_loss_lb(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
        if current_loss is not None:
            logs[-1]['loss'] = current_loss.detach()
        current_non_grad_lb_per_instance = logs[0]['lb_per_instance']
        initial_lb = current_non_grad_lb_per_instance
        current_max_lb = np.NINF
        initial_lb_change = 0
        best_solver_state = None
        lb_after_free_update = None
        use_later_dual_block = False
        for r in range(num_rounds):
            if is_training and self.hparams.use_separate_model_later_stage and r + starting_round >= self.hparams.num_train_rounds // 2:
                use_later_dual_block = True

            if 'round_index' in self.hparams.con_lp_features:
                batch.con_lp_f[:, self.hparams.con_lp_features.index('round_index')] = r
            grad_enabled = r >= num_rounds - self.hparams.num_train_rounds_with_grad and is_training
            with torch.set_grad_enabled(grad_enabled):
                if not non_learned_updates:
                    batch, dist_weights, omega_vec, best_solver_state, current_max_lb, lb_after_free_update = self.single_dual_round(
                                                                                                            batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, 
                                                                                                            return_best_dual, current_max_lb, best_solver_state, is_training,
                                                                                                            use_later_dual_block = use_later_dual_block)
                else:
                    with torch.no_grad():
                        batch = sol_utils.non_learned_updates(batch, self.hparams.edge_lp_features, num_dual_iterations, improvement_slope = 0.0, omega = batch.omega.item())

                lb_after_dist = sol_utils.compute_per_bdd_lower_bound(batch.solvers, sol_utils.distribute_delta(batch.solvers, batch.solver_state))
                with torch.no_grad():
                    logs.append({'r' : r + 1, 'lb_per_instance': scatter_sum(lb_after_dist, batch.batch_index_con), 't': time.time()})
                if self.hparams.free_update_loss_weight < 1.0:
                    current_loss = (1.0 - self.hparams.free_update_loss_weight) * self.dual_loss_lb(lb_after_dist, batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
                    logs[-1]['loss'] = current_loss.detach()
                elif lb_after_free_update is not None:
                    current_loss = current_loss + self.hparams.free_update_loss_weight * self.dual_loss_lb(lb_after_free_update, batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
                    logs[-1]['loss'] = current_loss.detach()
                if current_loss is not None: 
                    loss = loss + torch.pow(torch.tensor(self.hparams.loss_discount_factor), num_rounds - r - 1) * current_loss
            if not is_training:
                lb_prev_round = current_non_grad_lb_per_instance
                current_non_grad_lb_per_instance = logs[-1]['lb_per_instance']
                if r == 1:
                    initial_lb_change = current_non_grad_lb_per_instance - lb_prev_round
                last_lb_change = current_non_grad_lb_per_instance - lb_prev_round
                rel_improvement = last_lb_change / initial_lb_change
                if (self.hparams.use_separate_model_later_stage and not use_later_dual_block and
                    (rel_improvement < 1e-6 or r + starting_round >= num_rounds // 2)):
                    print(f'Switching to later dual stage at round: {r} / {num_rounds}.')
                    use_later_dual_block = True
                    continue
                # if initial_lb > current_non_grad_lb_per_instance:
                #     print(f'\n Early termination. last_lb_change: {last_lb_change}, initial_lb_change: {initial_lb_change}, rel_improvement: {rel_improvement}')
                #     for r_fake in range(r + 1, num_rounds):
                #         logs.append({'r' : r_fake + 1, 'lb_per_instance': logs[-1]['lb_per_instance'], 't': time.time()})
                #         if current_loss is not None:
                #             logs[-1]['loss'] = current_loss.detach()
                #     break

        if is_training:
            # for (filepath, lb_inst) in zip(batch.file_path, logs[-1]['lb_per_instance']):
            #     filename = os.path.basename(filepath)
            #     self.logger.experiment.add_scalar(f'train/lb_{filename}', lb_inst, global_step = self.global_step)
            self.logger.experiment.add_scalar('train/subgradient_step_size', torch.abs(self.dual_block.subgradient_step_size[0]).item(), global_step = self.global_step)
        return loss, batch, logs, best_solver_state

    def training_step(self, batch, batch_idx):
        num_rounds_to_run = self.num_rounds
        if self.hparams.use_replay_buffer:
            # All trajectories within a batch are sampled at the same round, so taking 0-th index is ok.
            num_rounds_to_run = max(self.num_rounds - batch.sampled_trajectory_locations[0], 1)
        rounds_already_done = self.num_rounds - num_rounds_to_run
        loss, batch, logs, _ = self.dual_rounds(batch, rounds_already_done, num_rounds_to_run, self.hparams.num_dual_iter_train, self.hparams.dual_improvement_slope_train, self.hparams.grad_dual_itr_max_itr, is_training = True)
        self.train_metrics.update(batch, logs)
        if self.hparams.use_replay_buffer:
            lo_costs_c = batch.solver_state['lo_costs'].detach().cpu()
            hi_costs_c = batch.solver_state['hi_costs'].detach().cpu()
            def_mm_c = batch.solver_state['def_mm'].detach().cpu()
            if self.hparams.use_lstm_var:
                lstm_h = batch.var_hidden_states_lstm['h'].detach().cpu()
                lstm_c = batch.var_hidden_states_lstm['c'].detach().cpu()
                batch_index_var_c = batch.batch_index_var.cpu()
            batch_index_edge_c = batch.batch_index_edge.cpu()
            for (b, inst_p) in enumerate(batch.file_path):
                if not inst_p in self.replay_buffer:
                    self.replay_buffer[inst_p] = ReplayBuffer()
                mask = batch_index_edge_c == b
                if not self.hparams.use_lstm_var:
                    self.replay_buffer[inst_p].add(batch.sampled_trajectory_locations[b] + num_rounds_to_run, (lo_costs_c[mask], hi_costs_c[mask], def_mm_c[mask]))
                else:
                    mask_var = batch_index_var_c == b
                    self.replay_buffer[inst_p].add(batch.sampled_trajectory_locations[b] + num_rounds_to_run, (lo_costs_c[mask], hi_costs_c[mask], def_mm_c[mask], lstm_h[mask_var], lstm_c[mask_var]))
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        data_name = self.val_datanames[dataset_idx]
        if self.non_learned_updates_val:
            loss, batch, logs, _ = self.dual_rounds(batch, 0, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, non_learned_updates = True)
            # self.eval_metrics_val_non_learned[data_name].update(batch, logs) # creates too many files on disk
        else:
            instance_name = os.path.basename(batch.file_path[0])
            data_name = self.val_datanames[dataset_idx]
            instance_log_name = f'val_{data_name}_{self.global_step}/{instance_name}'
            loss, batch, logs, _ = self.dual_rounds(batch, 0, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, instance_log_name = instance_log_name)
            # self.eval_metrics_val[data_name].update(batch, logs) # creates too many files on disk

        return loss

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        self.replay_buffer = {}
        assert len(batch.file_path) == 1, 'batch size 1 required for testing.'
        print(f'Testing on {batch.file_path}')
        instance_name = os.path.basename(batch.file_path[0])
        data_name = self.test_datanames[dataset_idx]
        instance_log_name = f'test_{data_name}_{instance_name}'
        if self.non_learned_updates_test:
            orig_batch = batch.clone()
        if not self.only_test_non_learned:
            loss, batch, logs, best_solver_state = self.dual_rounds(batch, 0, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, 
                                                is_training = False, instance_log_name = instance_log_name, non_learned_updates = False, return_best_dual = True)
            instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
            instance_level_metrics.update(batch, logs)
            self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'learned')
            self.eval_metrics_test[data_name].update(batch, logs)

        if self.non_learned_updates_test:
            # Perform non-learned updates:
            _, batch, logs, _ = self.dual_rounds(orig_batch, 0, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, 
                                                is_training = False, instance_log_name = None, non_learned_updates = True)
            instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
            instance_level_metrics.update(batch, logs)
            self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'non_learned')
            self.eval_metrics_test_non_learned[data_name].update(batch, logs)
            loss = None
        return loss
