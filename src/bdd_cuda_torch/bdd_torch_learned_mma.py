import torch
from torch_scatter import scatter_sum, scatter_min, scatter_logsumexp, scatter_max
import bdd_cuda_torch.bdd_torch_base as bdd_torch_base

class bdd_torch_learned_mma(bdd_torch_base):
    def __init__(self, cuda_solver, device = 'cuda'):
        super().__init__(cuda_solver, device)

    def get_isotropic_alpha(self):
        primal_variable_index = self.get_variable_index()
        alpha = torch.ones(primal_variable_index.shape, device = primal_variable_index.device)
        return alpha / scatter_sum(alpha, primal_variable_index)[primal_variable_index]

    def iterations(self, lo_costs, hi_costs, def_mm, alpha, omega, num_iterations, smoothing = None):
        is_smooth = False
        if smoothing is not None:
            assert torch.is_tensor(smoothing)
            is_smooth = True
        cost_from_terminal = self.backward_run(lo_costs, hi_costs, smoothing)
        for _ in range(num_iterations):
            lo_costs_out, hi_costs_out, def_mm_out, cost_from_root = self.forward_iteration(lo_costs, hi_costs, def_mm, alpha, omega, cost_from_terminal, smoothing)
            lo_costs, hi_costs, def_mm, cost_from_terminal = self.backward_iteration(lo_costs_out, hi_costs_out, def_mm_out, alpha, omega, cost_from_root, smoothing)
        return lo_costs, hi_costs, def_mm

    def compute_delta(self, def_mm):
        primal_variable_index = self.get_variable_index()
        delta_lo = scatter_sum(torch.relu(-def_mm), primal_variable_index)
        delta_hi = scatter_sum(torch.relu(def_mm), primal_variable_index)
        return delta_lo[:-1], delta_hi[:-1], primal_variable_index

    def mm_diff_hop(self, hop_index, lo_cost_hop, hi_cost_hop, cost_from_root_hop, cost_from_terminal, lo_bdd_node_hop, hi_bdd_node_hop, valid_node_indices, smoothing_hop, smoothing_hop_per_layer):
        is_smooth = False
        if smoothing_hop is not None:
            assert torch.is_tensor(smoothing_hop)
            is_smooth = True
        start_node, end_node = self.start_end_bdd_node_indices(hop_index)
        bdd_node_to_layer_map_hop = self.bdd_node_to_layer_map_[valid_node_indices]
        bdd_node_to_layer_map_hop = bdd_node_to_layer_map_hop - bdd_node_to_layer_map_hop.min()
        lo_path_costs_hop = cost_from_root_hop + lo_cost_hop + cost_from_terminal[lo_bdd_node_hop]
        hi_path_costs_hop = cost_from_root_hop + hi_cost_hop + cost_from_terminal[hi_bdd_node_hop]
        if not is_smooth:
            marginal_lo_hop = scatter_min(lo_path_costs_hop, bdd_node_to_layer_map_hop)[0]
            marginal_hi_hop = scatter_min(hi_path_costs_hop, bdd_node_to_layer_map_hop)[0]
        else:
            marginal_lo_hop = -smoothing_hop_per_layer * scatter_logsumexp(-self.divide_smoothing(lo_path_costs_hop, smoothing_hop), bdd_node_to_layer_map_hop)
            marginal_hi_hop = -smoothing_hop_per_layer * scatter_logsumexp(-self.divide_smoothing(hi_path_costs_hop, smoothing_hop), bdd_node_to_layer_map_hop)

        return marginal_hi_hop - marginal_lo_hop

    def forward_iteration(self, lo_costs, hi_costs, def_mm, alpha, omega, cost_from_terminal, smoothing = None):
        is_smooth = False
        smoothing_hop_per_layer = None
        if smoothing is not None:
            assert torch.is_tensor(smoothing)
            is_smooth = True
        cost_from_root = self.init_costs_from_root(is_smooth)
        delta_lo, delta_hi, primal_variable_index = self.compute_delta(def_mm)
        lo_costs_out = lo_costs.clone()
        hi_costs_out = hi_costs.clone()
        mm_diff_out = torch.zeros_like(def_mm)
        if is_smooth:
            output_indices = torch.zeros_like(cost_from_root, dtype = torch.int32)
        for hop_index in range(self.nr_hops()):
            valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing_hop, node_to_layer_map_hop = self.get_hop_data(hop_index, lo_costs, hi_costs, smoothing, return_node_to_layer_map = True)

            cost_from_root_hop = cost_from_root[valid_node_indices]
            valid_layer_indices = self.valid_layer_indices(hop_index)
            if is_smooth:
                if smoothing.numel() > 1:
                    smoothing_hop_per_layer = smoothing[self.get_bdd_index()[valid_layer_indices]]
                else:
                    smoothing_hop_per_layer = smoothing_hop
            
            cur_mm_diff_hop = self.mm_diff_hop(hop_index, lo_cost_hop, hi_cost_hop, cost_from_root_hop, cost_from_terminal, lo_bdd_node_hop, hi_bdd_node_hop, valid_node_indices, smoothing_hop, smoothing_hop_per_layer)

            valid_layer_indices = self.valid_layer_indices(hop_index)
            mm_to_remove = omega[valid_layer_indices] * cur_mm_diff_hop
            lo_costs_out[valid_layer_indices] = lo_costs[valid_layer_indices] - torch.relu(-mm_to_remove) + alpha[valid_layer_indices] * delta_lo[primal_variable_index[valid_layer_indices]]
            hi_costs_out[valid_layer_indices] = hi_costs[valid_layer_indices] - torch.relu(mm_to_remove) + alpha[valid_layer_indices] * delta_hi[primal_variable_index[valid_layer_indices]]
            mm_diff_out[valid_layer_indices] = mm_to_remove

            lo_costs_out_hop, hi_costs_out_hop = lo_costs_out[node_to_layer_map_hop], hi_costs_out[node_to_layer_map_hop]
            if not is_smooth:
                next_cost_to_lo = cost_from_root_hop + lo_costs_out_hop
                next_cost_to_hi = cost_from_root_hop + hi_costs_out_hop
                scatter_min(next_cost_to_lo, lo_bdd_node_hop, out = cost_from_root)
                scatter_min(next_cost_to_hi, hi_bdd_node_hop, out = cost_from_root)
            else:
                next_costs = torch.cat(( (cost_from_root_hop + lo_costs_out_hop) / smoothing_hop, (cost_from_root_hop + hi_costs_out_hop) / smoothing_hop), 0)
                next_indices = torch.cat((lo_bdd_node_hop, hi_bdd_node_hop), 0)
                output_indices[next_indices] = 1
                cumulative_indices = torch.cumsum(output_indices, dim = 0)
                hop_logsumexp = scatter_logsumexp(-next_costs, (cumulative_indices[next_indices] - 1).to(torch.long))
                unique_next_indices = torch.where(output_indices)[0]
                unique_cumulative_next_indices = (cumulative_indices[unique_next_indices] - 1).to(torch.long)
                if smoothing.numel() == 1:
                    cost_from_root[unique_next_indices] = cost_from_root[unique_next_indices] - smoothing * hop_logsumexp[unique_cumulative_next_indices]
                else:
                    next_per_node_bdd_index_hop = self.get_bdd_index()[self.bdd_node_to_layer_map_[unique_next_indices]]
                    next_smoothing_hop = smoothing[next_per_node_bdd_index_hop]
                    cost_from_root[unique_next_indices] = cost_from_root[unique_next_indices] - next_smoothing_hop * hop_logsumexp[unique_cumulative_next_indices]
                output_indices[next_indices] = 0
        return lo_costs_out, hi_costs_out, mm_diff_out, cost_from_root

    def backward_iteration(self, lo_costs, hi_costs, def_mm, alpha, omega, cost_from_root, smoothing = None):
        is_smooth = False
        smoothing_hop_per_layer = None
        if smoothing is not None:
            assert torch.is_tensor(smoothing)
            is_smooth = True
        cost_from_terminal = self.init_costs_from_terminal(is_smooth)
        delta_lo, delta_hi, primal_variable_index = self.compute_delta(def_mm)
        lo_costs_out = lo_costs.clone()
        hi_costs_out = hi_costs.clone()
        mm_diff_out = torch.zeros_like(def_mm)
        for hop_index in reversed(range(self.nr_hops())):
            valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing_hop, node_to_layer_map_hop = self.get_hop_data(hop_index, lo_costs, hi_costs, smoothing, return_node_to_layer_map = True)
            cost_from_root_hop = cost_from_root[valid_node_indices]

            valid_layer_indices = self.valid_layer_indices(hop_index)
            if is_smooth:
                if smoothing.numel() > 1:
                    smoothing_hop_per_layer = smoothing[self.get_bdd_index()[valid_layer_indices]]
                else:
                    smoothing_hop_per_layer = smoothing_hop

            cur_mm_diff_hop = self.mm_diff_hop(hop_index, lo_cost_hop, hi_cost_hop, cost_from_root_hop, cost_from_terminal, lo_bdd_node_hop, hi_bdd_node_hop, valid_node_indices, smoothing_hop, smoothing_hop_per_layer)

            mm_to_remove = omega[valid_layer_indices] * cur_mm_diff_hop
            lo_costs_out[valid_layer_indices] = lo_costs[valid_layer_indices] - torch.relu(-mm_to_remove) + alpha[valid_layer_indices] * delta_lo[primal_variable_index[valid_layer_indices]]
            hi_costs_out[valid_layer_indices] = hi_costs[valid_layer_indices] - torch.relu(mm_to_remove) + alpha[valid_layer_indices] * delta_hi[primal_variable_index[valid_layer_indices]]
            mm_diff_out[valid_layer_indices] = mm_to_remove
            
            lo_costs_out_hop, hi_costs_out_hop = lo_costs_out[node_to_layer_map_hop], hi_costs_out[node_to_layer_map_hop]

            if is_smooth:
                next_lo_cost_from_terminal = self.divide_smoothing(cost_from_terminal[lo_bdd_node_hop] + lo_costs_out_hop, smoothing_hop)
                next_hi_cost_from_terminal = self.divide_smoothing(cost_from_terminal[hi_bdd_node_hop] + hi_costs_out_hop, smoothing_hop)
                net_costs = torch.logaddexp(-next_lo_cost_from_terminal, -next_hi_cost_from_terminal)
                cost_from_terminal[valid_node_indices] = -smoothing_hop * net_costs
            else:
                next_lo_cost_from_terminal = cost_from_terminal[lo_bdd_node_hop] + lo_costs_out_hop
                next_hi_cost_from_terminal = cost_from_terminal[hi_bdd_node_hop] + hi_costs_out_hop
                cost_from_terminal[valid_node_indices] = torch.minimum(next_lo_cost_from_terminal, next_hi_cost_from_terminal)

        return lo_costs_out, hi_costs_out, mm_diff_out, cost_from_terminal