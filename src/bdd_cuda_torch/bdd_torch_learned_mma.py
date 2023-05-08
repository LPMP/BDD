import torch
from torch_scatter import scatter_sum, scatter_min, scatter_logsumexp, scatter_max
import bdd_cuda_torch.bdd_torch_base as bdd_torch_base

class bdd_torch_learned_mma(bdd_torch_base):
    def __init__(self, cuda_solver):
        super().__init__(cuda_solver)

    def get_isotropic_alpha(self):
        primal_variable_index = self.get_variable_index()
        alpha = torch.ones(primal_variable_index.shape, device = primal_variable_index.device)
        return alpha / scatter_sum(alpha, primal_variable_index)[primal_variable_index]

    def iterations(self, lo_costs, hi_costs, def_mm, alpha, omega, num_iterations, smoothing_factor = 0.0):
        is_smooth = False
        if smoothing_factor > 0.0:
            lo_costs, hi_costs, def_mm = self.scale_costs(lo_costs, hi_costs, def_mm, 1.0 / smoothing_factor)
            is_smooth = True
        cost_from_terminal = self.backward_run(lo_costs, hi_costs, is_smooth)
        for _ in range(num_iterations):
            lo_costs_out, hi_costs_out, def_mm_out, cost_from_root = self.forward_iteration(lo_costs, hi_costs, def_mm, alpha, omega, cost_from_terminal, is_smooth)
            lo_costs, hi_costs, def_mm, cost_from_terminal = self.backward_iteration(lo_costs_out, hi_costs_out, def_mm_out, alpha, omega, cost_from_root, is_smooth)

        if smoothing_factor > 0.0:
            lo_costs, hi_costs, def_mm = self.scale_costs(lo_costs, hi_costs, def_mm, smoothing_factor)
        return lo_costs, hi_costs, def_mm

    def compute_delta(self, def_mm):
        primal_variable_index = self.get_variable_index()
        delta_lo = scatter_sum(torch.relu(-def_mm), primal_variable_index)
        delta_hi = scatter_sum(torch.relu(def_mm), primal_variable_index)
        return delta_lo[:-1], delta_hi[:-1], primal_variable_index

    def mm_diff_hop(self, hop_index, lo_cost_hop, hi_cost_hop, cost_from_root_hop, cost_from_terminal, lo_bdd_node_hop, hi_bdd_node_hop, valid_node_mask, is_smooth = False):
        start_node, end_node = self.start_end_bdd_node_indices(hop_index)
        bdd_node_to_layer_map_hop = self.bdd_node_to_layer_map_[start_node:end_node][valid_node_mask]
        bdd_node_to_layer_map_hop = bdd_node_to_layer_map_hop - bdd_node_to_layer_map_hop.min()
        lo_path_costs_hop = cost_from_root_hop + lo_cost_hop + cost_from_terminal[lo_bdd_node_hop]
        hi_path_costs_hop = cost_from_root_hop + hi_cost_hop + cost_from_terminal[hi_bdd_node_hop]
        if not is_smooth:
            marginal_lo_hop = scatter_min(lo_path_costs_hop, bdd_node_to_layer_map_hop)[0]
            marginal_hi_hop = scatter_min(hi_path_costs_hop, bdd_node_to_layer_map_hop)[0]
        else:
            marginal_lo_hop = -scatter_logsumexp(-lo_path_costs_hop, bdd_node_to_layer_map_hop)
            marginal_hi_hop = -scatter_logsumexp(-hi_path_costs_hop, bdd_node_to_layer_map_hop)

        return marginal_hi_hop - marginal_lo_hop

    def get_valid_layer_mask(self, start_layer, end_layer, primal_variable_index):
        return primal_variable_index[start_layer:end_layer] < self.cuda_solver.nr_primal_variables()

    def forward_iteration(self, lo_costs, hi_costs, def_mm, alpha, omega, cost_from_terminal, is_smooth = False):
        cost_from_root = self.init_costs_from_root(is_smooth)
        delta_lo, delta_hi, primal_variable_index = self.compute_delta(def_mm)
        lo_costs_out = lo_costs.clone()
        hi_costs_out = hi_costs.clone()
        mm_diff_out = torch.zeros_like(def_mm)
        if is_smooth:
            output_indices = torch.zeros_like(cost_from_root, dtype = torch.int32)
        for hop_index in range(self.nr_hops()):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, valid_node_mask = self.get_hop_data(hop_index, lo_costs, hi_costs)
            cost_from_root_hop = cost_from_root[start_node:end_node][valid_node_mask]
            lo_cost_hop = lo_cost_hop[valid_node_mask]
            hi_cost_hop = hi_cost_hop[valid_node_mask]
            lo_bdd_node_hop = lo_bdd_node_hop[valid_node_mask]
            hi_bdd_node_hop = hi_bdd_node_hop[valid_node_mask]
            
            cur_mm_diff_hop = self.mm_diff_hop(hop_index, lo_cost_hop, hi_cost_hop, cost_from_root_hop, cost_from_terminal, lo_bdd_node_hop, hi_bdd_node_hop, valid_node_mask, is_smooth)

            start_layer, end_layer = self.start_end_layer_indices(hop_index)
            valid_layer_mask = self.get_valid_layer_mask(start_layer, end_layer, primal_variable_index)
            primal_index_hop = primal_variable_index[start_layer:end_layer][valid_layer_mask]
            alpha_hop = alpha[start_layer:end_layer][valid_layer_mask]
            omega_hop = omega[start_layer:end_layer][valid_layer_mask]

            mm_to_remove = omega_hop * cur_mm_diff_hop
            lo_costs_out[start_layer:end_layer][valid_layer_mask] = lo_costs[start_layer:end_layer][valid_layer_mask] - torch.relu(-mm_to_remove) + alpha_hop * delta_lo[primal_index_hop]
            hi_costs_out[start_layer:end_layer][valid_layer_mask] = hi_costs[start_layer:end_layer][valid_layer_mask] - torch.relu(mm_to_remove) + alpha_hop * delta_hi[primal_index_hop]
            mm_diff_out[start_layer:end_layer][valid_layer_mask] = mm_to_remove

            lo_costs_out_hop, hi_costs_out_hop = self.get_arc_costs_bdd_node(hop_index, lo_costs_out, hi_costs_out)
            if not is_smooth:
                next_cost_to_lo = cost_from_root_hop + lo_costs_out_hop[valid_node_mask]
                next_cost_to_hi = cost_from_root_hop + hi_costs_out_hop[valid_node_mask]
                scatter_min(next_cost_to_lo, lo_bdd_node_hop, out = cost_from_root)
                scatter_min(next_cost_to_hi, hi_bdd_node_hop, out = cost_from_root)
            else:
                next_costs = torch.cat((cost_from_root_hop + lo_costs_out_hop[valid_node_mask], 
                                        cost_from_root_hop + hi_costs_out_hop[valid_node_mask]), 0)
                next_indices = torch.cat((lo_bdd_node_hop, hi_bdd_node_hop), 0)
                output_indices[next_indices] = 1
                cumulative_indices = torch.cumsum(output_indices, dim = 0)
                hop_logsumexp = -scatter_logsumexp(-next_costs, (cumulative_indices[next_indices] - 1).to(torch.long))
                unique_next_indices = torch.where(output_indices)[0]
                unique_cumulative_next_indices = (cumulative_indices[unique_next_indices] - 1).to(torch.long)
                cost_from_root[unique_next_indices] = cost_from_root[unique_next_indices] + hop_logsumexp[unique_cumulative_next_indices]
                output_indices[next_indices] = 0
        return lo_costs_out, hi_costs_out, mm_diff_out, cost_from_root

    def backward_iteration(self, lo_costs, hi_costs, def_mm, alpha, omega, cost_from_root, is_smooth = False):
        cost_from_terminal = self.init_costs_from_terminal(is_smooth)
        delta_lo, delta_hi, primal_variable_index = self.compute_delta(def_mm)
        lo_costs_out = lo_costs.clone()
        hi_costs_out = hi_costs.clone()
        mm_diff_out = torch.zeros_like(def_mm)
        for hop_index in reversed(range(self.nr_hops())):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, valid_node_mask = self.get_hop_data(hop_index, lo_costs, hi_costs)
            cost_from_root_hop = cost_from_root[start_node:end_node][valid_node_mask]
            lo_cost_hop = lo_cost_hop[valid_node_mask]
            hi_cost_hop = hi_cost_hop[valid_node_mask]
            lo_bdd_node_hop = lo_bdd_node_hop[valid_node_mask]
            hi_bdd_node_hop = hi_bdd_node_hop[valid_node_mask]

            cur_mm_diff_hop = self.mm_diff_hop(hop_index, lo_cost_hop, hi_cost_hop, cost_from_root_hop, cost_from_terminal, lo_bdd_node_hop, hi_bdd_node_hop, valid_node_mask, is_smooth)

            start_layer, end_layer = self.start_end_layer_indices(hop_index)
            valid_layer_mask = self.get_valid_layer_mask(start_layer, end_layer, primal_variable_index)
            primal_index_hop = primal_variable_index[start_layer:end_layer][valid_layer_mask]
            alpha_hop = alpha[start_layer:end_layer][valid_layer_mask]
            omega_hop = omega[start_layer:end_layer][valid_layer_mask]
            
            mm_to_remove = omega_hop * cur_mm_diff_hop

            lo_costs_out[start_layer:end_layer][valid_layer_mask] = lo_costs[start_layer:end_layer][valid_layer_mask] - torch.relu(-mm_to_remove) + alpha_hop * delta_lo[primal_index_hop]
            hi_costs_out[start_layer:end_layer][valid_layer_mask] = hi_costs[start_layer:end_layer][valid_layer_mask] - torch.relu(mm_to_remove) + alpha_hop * delta_hi[primal_index_hop]
            mm_diff_out[start_layer:end_layer][valid_layer_mask] = mm_to_remove
            
            lo_costs_out_hop, hi_costs_out_hop = self.get_arc_costs_bdd_node(hop_index, lo_costs_out, hi_costs_out)
            hop_indices = torch.arange(start_node, end_node, dtype = torch.long, device = 'cuda')
            next_lo_cost_from_terminal = cost_from_terminal[lo_bdd_node_hop] + lo_costs_out_hop[valid_node_mask]
            next_hi_cost_from_terminal = cost_from_terminal[hi_bdd_node_hop] + hi_costs_out_hop[valid_node_mask]
            if not is_smooth:
                cost_from_terminal[hop_indices[valid_node_mask]] = torch.minimum(next_lo_cost_from_terminal, next_hi_cost_from_terminal)
            else:
                cost_from_terminal[hop_indices[valid_node_mask]] = -torch.logaddexp(-next_lo_cost_from_terminal, -next_hi_cost_from_terminal)

        return lo_costs_out, hi_costs_out, mm_diff_out, cost_from_terminal