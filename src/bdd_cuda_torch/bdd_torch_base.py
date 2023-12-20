import torch
from torch_scatter import scatter_sum, scatter_min, scatter_logsumexp, scatter_max
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_cuda_solver

class bdd_torch_base:
    def __init__(self, cuda_solver, device = 'cuda'):
        self.device = device
        self.cuda_solver = cuda_solver
        self.cum_nr_bdd_nodes_per_hop_dist_ = cuda_solver.get_cum_nr_bdd_nodes_per_hop_dist()
        self.cum_nr_layers_per_hop_dist_ = cuda_solver.get_cum_nr_layers_per_hop_dist()
        self.nr_variables_per_hop_dist_ = cuda_solver.get_nr_variables_per_hop_dist()

        self.lo_bdd_node_ = torch.empty((cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        self.hi_bdd_node_ = torch.empty((cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        cuda_solver.lo_hi_bdd_node_index(self.lo_bdd_node_.data_ptr(), self.hi_bdd_node_.data_ptr())
        self.lo_bdd_node_ = self.lo_bdd_node_.to(torch.long).to(device)
        self.hi_bdd_node_ = self.hi_bdd_node_.to(torch.long).to(device)

        self.bdd_node_to_layer_map_ = torch.empty((cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        cuda_solver.bdd_node_to_layer_map(self.bdd_node_to_layer_map_.data_ptr())
        self.bdd_node_to_layer_map_ = self.bdd_node_to_layer_map_.to(torch.long).to(device)

        self.root_indices_ = torch.empty((cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.top_sink_indices_ = torch.empty((cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.bot_sink_indices_ = torch.empty((cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        cuda_solver.root_indices(self.root_indices_.data_ptr())
        cuda_solver.top_sink_indices(self.top_sink_indices_.data_ptr())
        cuda_solver.bot_sink_indices(self.bot_sink_indices_.data_ptr())
        self.root_indices_ = self.root_indices_.to(torch.long).to(device)
        self.top_sink_indices_ = self.top_sink_indices_.to(torch.long).to(device)
        self.bot_sink_indices_ = self.bot_sink_indices_.to(torch.long).to(device)

        self.primal_variable_index = None
        self.bdd_index = None

    def divide_smoothing(self, arg, smoothing_hop):
        if smoothing_hop is None:
            return arg
        valid_mask = torch.isfinite(arg)
        if smoothing_hop.numel() == arg.numel():
            arg[valid_mask] = arg[valid_mask] / smoothing_hop[valid_mask]
            return arg
        else:
            assert smoothing_hop.numel() == 1
            arg[valid_mask] = arg[valid_mask] / smoothing_hop
            return arg

    def nr_hops(self):
        return len(self.cum_nr_layers_per_hop_dist_) - 1 # ignores terminal nodes.

    def start_end_bdd_node_indices(self, hop_index):
        if hop_index == 0:
            return 0, self.cum_nr_bdd_nodes_per_hop_dist_[0]
        else:
            return self.cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1], self.cum_nr_bdd_nodes_per_hop_dist_[hop_index]

    def get_valid_layer_mask(self, start_layer, end_layer):
        return self.get_variable_index()[start_layer:end_layer] < self.cuda_solver.nr_primal_variables()

    def valid_layer_indices(self, hop_index):
        if hop_index == 0:
            start_layer, end_layer = 0, self.cum_nr_layers_per_hop_dist_[0]
        else:
            start_layer, end_layer = self.cum_nr_layers_per_hop_dist_[hop_index - 1], self.cum_nr_layers_per_hop_dist_[hop_index]
        return torch.arange(start_layer, end_layer, dtype = torch.long, device = self.device)[self.get_valid_layer_mask(start_layer, end_layer)]
        
    def get_variable_index(self):
        if self.primal_variable_index is None:
            self.primal_variable_index = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
            self.cuda_solver.primal_variable_index(self.primal_variable_index.data_ptr())
            self.primal_variable_index = self.primal_variable_index.to(torch.long).to(self.device)
        return self.primal_variable_index

    def get_bdd_index(self):
        if self.bdd_index is None:
            self.bdd_index = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
            self.cuda_solver.bdd_index(self.bdd_index.data_ptr())
            self.bdd_index = self.bdd_index.to(torch.long).to(self.device)
        return self.bdd_index.to(torch.long)

    def init_costs_from_root(self, is_log_sum_exp = False):
        cost_from_root = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.get_default_dtype(), device = self.device)
        if not is_log_sum_exp:
            cost_from_root[:] = float("Inf")
            cost_from_root[self.root_indices_] = 0.0
        else:
            cost_from_root[:] = 0.0
            cost_from_root[self.bot_sink_indices_] = float("Inf")
        return cost_from_root

    def init_costs_from_terminal(self, is_log_sum_exp = False):
        cost_from_terminal = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.get_default_dtype(), device = self.device)
        cost_from_terminal[:] = 0.0
        cost_from_terminal[self.bot_sink_indices_] = float("Inf")
        return cost_from_terminal

    # def get_arc_costs_bdd_node(self, hop_index, lo_costs, hi_costs):
    #     bdd_node_start, bdd_node_end = self.start_end_bdd_node_indices(hop_index)
    #     node_to_layer_map = self.bdd_node_to_layer_map_[bdd_node_start: bdd_node_end]
    #     return lo_costs[node_to_layer_map], hi_costs[node_to_layer_map]

    def get_hop_data(self, hop_index, lo_costs, hi_costs, smoothing = None, return_node_to_layer_map = False):
        start_node, end_node = self.start_end_bdd_node_indices(hop_index)
        valid_node_mask = self.lo_bdd_node_[start_node:end_node] >= 0 # No outgoing arcs from terminal nodes.
        valid_node_indices = torch.arange(start_node, end_node, dtype = torch.long, device = self.device)[valid_node_mask]
        lo_bdd_node_hop = self.lo_bdd_node_[valid_node_indices]
        hi_bdd_node_hop = self.hi_bdd_node_[valid_node_indices]
        node_to_layer_map_hop = self.bdd_node_to_layer_map_[valid_node_indices]
        return_map = node_to_layer_map_hop if return_node_to_layer_map else None 
        lo_cost_hop = lo_costs[node_to_layer_map_hop]
        hi_cost_hop = hi_costs[node_to_layer_map_hop]
        if smoothing is not None and smoothing.numel() > 1:
            assert smoothing.numel() == self.cuda_solver.nr_bdds()
            per_node_bdd_index_hop = self.get_bdd_index()[node_to_layer_map_hop]
            smoothing_hop = smoothing[per_node_bdd_index_hop]
            return valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing_hop, return_map
        else:
            return valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing, return_map

    def forward_run(self, lo_costs, hi_costs, smoothing = None):
        is_smooth = False
        if smoothing is not None:
            assert torch.is_tensor(smoothing)
            is_smooth = True
        cost_from_root = self.init_costs_from_root(is_smooth)
        if is_smooth:
            output_indices = torch.zeros_like(cost_from_root, dtype = torch.int32)
        for hop_index in range(self.nr_hops()):
            valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing_hop, _ = self.get_hop_data(hop_index, lo_costs, hi_costs, smoothing)
            cost_from_root_hop = cost_from_root[valid_node_indices]
            
            if is_smooth:
                next_costs = torch.cat(( (cost_from_root_hop + lo_cost_hop) / smoothing_hop, (cost_from_root_hop + hi_cost_hop) / smoothing_hop), 0)
                next_indices = torch.cat((lo_bdd_node_hop, hi_bdd_node_hop), 0)

                # Implements following in an efficient manner:
                # cost_from_root = cost_from_root - smoothing * scatter_logsumexp(-next_costs, next_indices, dim_size = cost_from_root.numel())
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
            else:
                next_cost_to_lo = cost_from_root_hop + lo_cost_hop
                next_cost_to_hi = cost_from_root_hop + hi_cost_hop
                scatter_min(next_cost_to_lo, lo_bdd_node_hop, out = cost_from_root)
                scatter_min(next_cost_to_hi, hi_bdd_node_hop, out = cost_from_root)
        return cost_from_root

    def backward_run(self, lo_costs, hi_costs, smoothing = None):
        is_smooth = False
        if smoothing is not None:
            assert torch.is_tensor(smoothing)
            is_smooth = True
        cost_from_terminal = self.init_costs_from_terminal(is_smooth)
        for hop_index in reversed(range(self.nr_hops())):
            valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing_hop, _ = self.get_hop_data(hop_index, lo_costs, hi_costs, smoothing)

            if is_smooth:
                next_lo_cost_from_terminal = self.divide_smoothing(cost_from_terminal[lo_bdd_node_hop] + lo_cost_hop, smoothing_hop)
                next_hi_cost_from_terminal = self.divide_smoothing(cost_from_terminal[hi_bdd_node_hop] + hi_cost_hop, smoothing_hop)
                cost_from_terminal[valid_node_indices] = -smoothing_hop * torch.logaddexp(-next_lo_cost_from_terminal, -next_hi_cost_from_terminal)
            else:
                next_lo_cost_from_terminal = cost_from_terminal[lo_bdd_node_hop] + lo_cost_hop
                next_hi_cost_from_terminal = cost_from_terminal[hi_bdd_node_hop] + hi_cost_hop                
                cost_from_terminal[valid_node_indices] = torch.minimum(next_lo_cost_from_terminal, next_hi_cost_from_terminal)
        return cost_from_terminal

    def compute_lower_bound(self, lo_costs, hi_costs, smoothing = None, def_mm = None, from_cuda = False):
        if not from_cuda:
            cost_from_terminal = self.backward_run(lo_costs, hi_costs, smoothing)
            lb_sum = torch.sum(cost_from_terminal[self.root_indices_])
            return lb_sum
            # cost_from_root = self.forward_run(lo_costs, hi_costs, is_smooth)
            # return torch.sum(cost_from_root[self.top_sink_indices_])
        else:
            assert smoothing is None
            assert lo_costs.is_cuda
            assert hi_costs.is_cuda
            if def_mm is None:
                def_mm = torch.zeros_like(lo_costs)
            assert def_mm.is_cuda
            lo_costs, hi_costs, def_mm = lo_costs.contiguous(), hi_costs.contiguous(), def_mm.contiguous()
            self.cuda_solver.set_solver_costs(lo_costs.data_ptr(), hi_costs.data_ptr(), def_mm.data_ptr())
            return self.cuda_solver.lower_bound()

    def per_bdd_lower_bound(self, cost_from_terminal):
        rearranged = torch.empty(self.cuda_solver.nr_bdds(), device = self.device)
        rearranged[self.get_bdd_index()[:self.cuda_solver.nr_bdds()]] = cost_from_terminal[:self.cuda_solver.nr_bdds()]
        return rearranged

    def compute_solution_objective(self, lo_costs, hi_costs, solution):
        net_costs = hi_costs - lo_costs
        return torch.sum(solution * net_costs[self.valid_layer_mask()])

    def marginals(self, lo_costs, hi_costs, smoothing = None, return_lb = False):
        is_smooth = False
        if smoothing is not None:
            assert torch.is_tensor(smoothing)
            is_smooth = True
        cost_from_root = self.forward_run(lo_costs, hi_costs, smoothing)
        cost_from_terminal = self.backward_run(lo_costs, hi_costs, smoothing)
        lo_path_costs = torch.empty_like(cost_from_root)
        hi_path_costs = torch.empty_like(cost_from_root)
        for hop_index in range(self.nr_hops()):
            valid_node_indices, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, smoothing_hop, _ = self.get_hop_data(hop_index, lo_costs, hi_costs, smoothing)
            lo_path_costs[valid_node_indices] = self.divide_smoothing(cost_from_root[valid_node_indices] + lo_cost_hop + cost_from_terminal[lo_bdd_node_hop], smoothing_hop)
            hi_path_costs[valid_node_indices] = self.divide_smoothing(cost_from_root[valid_node_indices] + hi_cost_hop + cost_from_terminal[hi_bdd_node_hop], smoothing_hop)
        if is_smooth:
            if torch.numel(smoothing) > 1:
                smoothing_per_layer = smoothing[self.get_bdd_index()]
                marginal_lo = -smoothing_per_layer * scatter_logsumexp(-lo_path_costs, self.bdd_node_to_layer_map_)
                marginal_hi = -smoothing_per_layer * scatter_logsumexp(-hi_path_costs, self.bdd_node_to_layer_map_)
            else:
                assert torch.numel(smoothing) == 1
                marginal_lo = -smoothing * scatter_logsumexp(-lo_path_costs, self.bdd_node_to_layer_map_)
                marginal_hi = -smoothing * scatter_logsumexp(-hi_path_costs, self.bdd_node_to_layer_map_)
        else:
            marginal_lo = scatter_min(lo_path_costs, self.bdd_node_to_layer_map_)[0]
            marginal_hi = scatter_min(hi_path_costs, self.bdd_node_to_layer_map_)[0]
        if not return_lb:
            return marginal_lo, marginal_hi
        else:
            return marginal_lo, marginal_hi, per_bdd_lower_bound(cost_from_terminal)

    def smooth_solution(self, lo_costs, hi_costs, smoothing = None):
        marginal_lo, marginal_hi = self.marginals(lo_costs, hi_costs, smoothing)
        return torch.nn.Softmax(dim = 1)(torch.stack((-marginal_lo, -marginal_hi), 1))[:, 1]

    def smooth_solution_logits(self, lo_costs, hi_costs, smoothing = None):
        marginal_lo, marginal_hi = self.marginals(lo_costs, hi_costs, smoothing)
        return torch.nn.LogSoftmax(dim = 1)(torch.stack((-marginal_lo, -marginal_hi), 1))[:, 1]        

    def valid_layer_mask(self):
        mask = torch.ones(self.cuda_solver.nr_layers(), dtype=torch.bool, device = self.device)
        mask[self.bdd_node_to_layer_map_[self.bot_sink_indices_]] = False
        return mask
        
    def valid_bdd_node_mask(self):
        mask = torch.ones(self.cuda_solver.nr_bdd_nodes(), dtype=torch.bool, device = self.device)
        mask[self.bot_sink_indices_] = False
        return mask
