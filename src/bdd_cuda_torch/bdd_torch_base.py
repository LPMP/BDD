import torch
from torch_scatter import scatter_sum, scatter_min, scatter_logsumexp, scatter_max
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_cuda_solver

class bdd_torch_base:
    def __init__(self, cuda_solver):
        self.cuda_solver = cuda_solver
        self.cum_nr_bdd_nodes_per_hop_dist_ = cuda_solver.get_cum_nr_bdd_nodes_per_hop_dist()
        self.cum_nr_layers_per_hop_dist_ = cuda_solver.get_cum_nr_layers_per_hop_dist()
        self.nr_variables_per_hop_dist_ = cuda_solver.get_nr_variables_per_hop_dist()

        self.lo_bdd_node_ = torch.empty((cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        self.hi_bdd_node_ = torch.empty((cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        cuda_solver.lo_hi_bdd_node_index(self.lo_bdd_node_.data_ptr(), self.hi_bdd_node_.data_ptr())
        self.lo_bdd_node_ = self.lo_bdd_node_.to(torch.long)
        self.hi_bdd_node_ = self.hi_bdd_node_.to(torch.long)

        self.bdd_node_to_layer_map_ = torch.empty((cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        cuda_solver.bdd_node_to_layer_map(self.bdd_node_to_layer_map_.data_ptr())
        self.bdd_node_to_layer_map_ = self.bdd_node_to_layer_map_.to(torch.long)

        self.root_indices_ = torch.empty((cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.top_sink_indices_ = torch.empty((cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.bot_sink_indices_ = torch.empty((cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        cuda_solver.root_indices(self.root_indices_.data_ptr())
        cuda_solver.top_sink_indices(self.top_sink_indices_.data_ptr())
        cuda_solver.bot_sink_indices(self.bot_sink_indices_.data_ptr())
        self.root_indices_ = self.root_indices_.to(torch.long)
        self.top_sink_indices_ = self.top_sink_indices_.to(torch.long)
        self.bot_sink_indices_ = self.bot_sink_indices_.to(torch.long)

    # def __init__(self, bdd_ilp_instance):
    #     # bdd_ilp_instance = ilp_instance_bbd.read_ILP('ilp_path')
    #     self.cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(bdd_ilp_instance, True, 1.0)
    #     self.cum_nr_bdd_nodes_per_hop_dist_ = self.cuda_solver.get_cum_nr_bdd_nodes_per_hop_dist()
    #     self.cum_nr_layers_per_hop_dist_ = self.cuda_solver.get_cum_nr_layers_per_hop_dist()
    #     self.nr_variables_per_hop_dist_ = self.cuda_solver.get_nr_variables_per_hop_dist()

    #     self.primal_variable_index_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
    #     self.bdd_index_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
    #     self.cuda_solver.primal_variable_index(self.primal_variable_index_.data_ptr())
    #     self.cuda_solver.bdd_index(self.bdd_index_.data_ptr())
    #     self.primal_variable_index_ = self.primal_variable_index_.to(torch.long)
    #     self.bdd_index_ = self.bdd_index_.to(torch.long)

    #     self.lo_bdd_node_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
    #     self.hi_bdd_node_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
    #     self.cuda_solver.lo_hi_bdd_node_index(self.lo_bdd_node_.data_ptr(), self.hi_bdd_node_.data_ptr())
    #     self.lo_bdd_node_ = self.lo_bdd_node_.to(torch.long)
    #     self.hi_bdd_node_ = self.hi_bdd_node_.to(torch.long)

    #     self.bdd_node_to_layer_map_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
    #     self.cuda_solver.bdd_node_to_layer_map(self.bdd_node_to_layer_map_.data_ptr())
    #     self.bdd_node_to_layer_map_ = self.bdd_node_to_layer_map_.to(torch.long)

    #     self.root_indices_ = torch.empty((self.cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
    #     self.top_sink_indices_ = torch.empty((self.cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
    #     self.bot_sink_indices_ = torch.empty((self.cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
    #     self.cuda_solver.root_indices(self.root_indices_.data_ptr())
    #     self.cuda_solver.top_sink_indices(self.top_sink_indices_.data_ptr())
    #     self.cuda_solver.bot_sink_indices(self.bot_sink_indices_.data_ptr())
    #     self.root_indices_ = self.root_indices_.to(torch.long)
    #     self.top_sink_indices_ = self.top_sink_indices_.to(torch.long)
    #     self.bot_sink_indices_ = self.bot_sink_indices_.to(torch.long)

    def nr_hops(self):
        return len(self.cum_nr_layers_per_hop_dist_) - 1 # ignores terminal nodes.

    def start_end_bdd_node_indices(self, hop_index):
        if hop_index == 0:
            return 0, self.cum_nr_bdd_nodes_per_hop_dist_[0]
        else:
            return self.cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1], self.cum_nr_bdd_nodes_per_hop_dist_[hop_index]

    def get_variable_index(self):
        primal_variable_index = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
        self.cuda_solver.primal_variable_index(primal_variable_index.data_ptr())
        return primal_variable_index.to(torch.long)

    def init_costs_from_root(self, is_log_sum_exp = False):
        cost_from_root = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.get_default_dtype(), device = 'cuda')
        if not is_log_sum_exp:
            cost_from_root[:] = float("Inf")
            cost_from_root[self.root_indices_] = 0.0
        else:
            cost_from_root[:] = 0.0
            cost_from_root[self.bot_sink_indices_] = float("Inf")
        return cost_from_root

    def init_costs_from_terminal(self, is_log_sum_exp = False):
        cost_from_terminal = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.get_default_dtype(), device = 'cuda')
        if not is_log_sum_exp:
            cost_from_terminal[:] = 0.0
            cost_from_terminal[self.bot_sink_indices_] = float("Inf")
        else:
            cost_from_terminal[:] = 0.0
            cost_from_terminal[self.bot_sink_indices_] = -float("Inf")
        return cost_from_terminal

    def get_arc_costs_bdd_node(self, hop_index, lo_costs, hi_costs):
        bdd_node_start, bdd_node_end = self.start_end_bdd_node_indices(hop_index)
        node_to_layer_map = self.bdd_node_to_layer_map_[bdd_node_start: bdd_node_end]
        return lo_costs[node_to_layer_map], hi_costs[node_to_layer_map]

    def get_hop_data(self, hop_index, lo_costs, hi_costs):
        start_node, end_node = self.start_end_bdd_node_indices(hop_index)
        lo_bdd_node_hop = self.lo_bdd_node_[start_node: end_node]
        hi_bdd_node_hop = self.hi_bdd_node_[start_node: end_node]
        lo_cost_hop, hi_cost_hop = self.get_arc_costs_bdd_node(hop_index, lo_costs, hi_costs)
        valid_node_mask = lo_bdd_node_hop >= 0 # No outgoing arcs from terminal nodes.
        return start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, valid_node_mask

    def forward_run(self, lo_costs, hi_costs, is_smooth = False, new_impl = True):
        cost_from_root = self.init_costs_from_root(is_smooth)
        if is_smooth and new_impl:
            output_indices = torch.zeros_like(cost_from_root, dtype = torch.int32)
        for hop_index in range(self.nr_hops()):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, valid_node_mask = self.get_hop_data(hop_index, lo_costs, hi_costs)
            cost_from_root_hop = cost_from_root[start_node:end_node]
            
            if is_smooth:
                next_costs = torch.cat((cost_from_root_hop[valid_node_mask] - lo_cost_hop[valid_node_mask], 
                                        cost_from_root_hop[valid_node_mask] - hi_cost_hop[valid_node_mask]), 0)
                next_indices = torch.cat((lo_bdd_node_hop[valid_node_mask], hi_bdd_node_hop[valid_node_mask]), 0)

                if new_impl:
                    output_indices[next_indices] = 1
                    cumulative_indices = torch.cumsum(output_indices, dim = 0)
                    hop_logsumexp = scatter_logsumexp(next_costs, (cumulative_indices[next_indices] - 1).to(torch.long))
                    unique_next_indices = torch.where(output_indices)[0]
                    unique_cumulative_next_indices = (cumulative_indices[unique_next_indices] - 1).to(torch.long)
                    cost_from_root[unique_next_indices] = cost_from_root[unique_next_indices] + hop_logsumexp[unique_cumulative_next_indices]
                    output_indices[next_indices] = 0
                else:
                    cost_from_root = cost_from_root + scatter_logsumexp(next_costs, next_indices, dim_size = cost_from_root.numel())
            else:
                next_cost_to_lo = cost_from_root_hop[valid_node_mask] + lo_cost_hop[valid_node_mask]
                next_cost_to_hi = cost_from_root_hop[valid_node_mask] + hi_cost_hop[valid_node_mask]
                scatter_min(next_cost_to_lo, lo_bdd_node_hop[valid_node_mask], out = cost_from_root)
                scatter_min(next_cost_to_hi, hi_bdd_node_hop[valid_node_mask], out = cost_from_root)
        return cost_from_root

    def backward_run(self, lo_costs, hi_costs, is_smooth = False):
        cost_from_terminal = self.init_costs_from_terminal(is_smooth)
        for hop_index in reversed(range(self.nr_hops())):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, valid_node_mask = self.get_hop_data(hop_index, lo_costs, hi_costs)
            hop_indices = torch.arange(start_node, end_node, dtype = torch.long, device = 'cuda')
            if is_smooth:
                next_lo_cost_from_terminal = cost_from_terminal[lo_bdd_node_hop[valid_node_mask]] - lo_cost_hop[valid_node_mask]
                next_hi_cost_from_terminal = cost_from_terminal[hi_bdd_node_hop[valid_node_mask]] - hi_cost_hop[valid_node_mask]
                cost_from_terminal[hop_indices[valid_node_mask]] = torch.logaddexp(next_lo_cost_from_terminal, next_hi_cost_from_terminal)
            else:
                next_lo_cost_from_terminal = cost_from_terminal[lo_bdd_node_hop[valid_node_mask]] + lo_cost_hop[valid_node_mask]
                next_hi_cost_from_terminal = cost_from_terminal[hi_bdd_node_hop[valid_node_mask]] + hi_cost_hop[valid_node_mask]
                cost_from_terminal[hop_indices[valid_node_mask]] = torch.minimum(next_lo_cost_from_terminal, next_hi_cost_from_terminal)
        return cost_from_terminal

    def compute_lower_bound(self, lo_costs, hi_costs, is_smooth = False):
        cost_from_terminal = self.backward_run(lo_costs, hi_costs, is_smooth)
        return torch.sum(cost_from_terminal[self.root_indices_])

    def compute_solution_objective(self, lo_costs, hi_costs, solution):
        net_costs = hi_costs - lo_costs
        return torch.sum(solution * net_costs[self.valid_layer_mask()])

    def marginals(self, lo_costs, hi_costs, is_smooth = False):
        cost_from_root = self.forward_run(lo_costs, hi_costs, is_smooth)
        cost_from_terminal = self.backward_run(lo_costs, hi_costs, is_smooth)
        lo_path_costs = torch.empty_like(cost_from_root)
        hi_path_costs = torch.empty_like(cost_from_root)
        for hop_index in range(self.nr_hops()):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, valid_node_mask = self.get_hop_data(hop_index, lo_costs, hi_costs)
            cost_from_root_hop = cost_from_root[start_node:end_node]
            if not is_smooth:
                next_cost_to_lo = cost_from_root_hop[valid_node_mask] + lo_cost_hop[valid_node_mask]
                next_cost_to_hi = cost_from_root_hop[valid_node_mask] + hi_cost_hop[valid_node_mask]
            else:
                next_cost_to_lo = cost_from_root_hop[valid_node_mask] - lo_cost_hop[valid_node_mask]
                next_cost_to_hi = cost_from_root_hop[valid_node_mask] - hi_cost_hop[valid_node_mask]
            hop_indices = torch.arange(start_node, end_node, dtype = torch.long, device = 'cuda')
            hop_indices = hop_indices[valid_node_mask]
            lo_path_costs[hop_indices] = next_cost_to_lo + cost_from_terminal[lo_bdd_node_hop[valid_node_mask]]
            hi_path_costs[hop_indices] = next_cost_to_hi + cost_from_terminal[hi_bdd_node_hop[valid_node_mask]]
        if is_smooth:
            marginal_lo = scatter_logsumexp(lo_path_costs, self.bdd_node_to_layer_map_)
            marginal_hi = scatter_logsumexp(hi_path_costs, self.bdd_node_to_layer_map_)
        else:
            marginal_lo = scatter_min(lo_path_costs, self.bdd_node_to_layer_map_)[0]
            marginal_hi = scatter_min(hi_path_costs, self.bdd_node_to_layer_map_)[0]
        return marginal_lo, marginal_hi

    def smooth_solution(self, lo_costs, hi_costs):
        marginal_lo, marginal_hi = self.marginals(lo_costs, hi_costs, True)
        return torch.nn.Softmax(dim = 1)(torch.stack((marginal_lo, marginal_hi), 1))[:, 1]

    def smooth_solution_logits(self, lo_costs, hi_costs):
        marginal_lo, marginal_hi = self.marginals(lo_costs, hi_costs, True)
        return torch.nn.LogSoftmax(dim = 1)(torch.stack((marginal_lo, marginal_hi), 1))[:, 1]        

    def valid_layer_mask(self):
        mask = torch.ones(self.cuda_solver.nr_layers(), dtype=torch.bool, device = 'cuda')
        mask[self.bdd_node_to_layer_map_[self.bot_sink_indices_]] = False
        return mask
        
    def valid_bdd_node_mask(self):
        mask = torch.ones(self.cuda_solver.nr_bdd_nodes(), dtype=torch.bool, device = 'cuda')
        mask[self.bot_sink_indices_] = False
        return mask
