import torch
from torch_scatter import scatter_sum, scatter_min 
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_cuda_solver

class bdd_torch_base:
    def __init__(self, bdd_ilp_instance):
        # bdd_ilp_instance = ilp_instance_bbd.read_ILP('ilp_path')
        self.cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(bdd_ilp_instance, True, 1.0)
        self.cum_nr_bdd_nodes_per_hop_dist_ = self.cuda_solver.get_cum_nr_bdd_nodes_per_hop_dist()
        self.cum_nr_layers_per_hop_dist_ = self.cuda_solver.get_cum_nr_layers_per_hop_dist()
        self.nr_variables_per_hop_dist_ = self.cuda_solver.get_nr_variables_per_hop_dist()

        self.primal_variable_index_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
        self.bdd_index_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.int32, device = 'cuda')
        self.cuda_solver.primal_variable_index(self.primal_variable_index_.data_ptr())
        self.cuda_solver.bdd_index(self.bdd_index_.data_ptr())
        self.primal_variable_index_ = self.primal_variable_index_.to(torch.long)
        self.bdd_index_ = self.bdd_index_.to(torch.long)

        self.lo_bdd_node_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        self.hi_bdd_node_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        self.cuda_solver.lo_hi_bdd_node_index(self.lo_bdd_node_.data_ptr(), self.hi_bdd_node_.data_ptr())
        self.lo_bdd_node_ = self.lo_bdd_node_.to(torch.long)
        self.hi_bdd_node_ = self.hi_bdd_node_.to(torch.long)

        self.bdd_node_to_layer_map_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.int32, device = 'cuda')
        self.cuda_solver.bdd_node_to_layer_map(self.bdd_node_to_layer_map_.data_ptr())
        self.bdd_node_to_layer_map_ = self.bdd_node_to_layer_map_.to(torch.long)

        self.root_indices_ = torch.empty((self.cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.top_sink_indices_ = torch.empty((self.cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.bot_sink_indices_ = torch.empty((self.cuda_solver.nr_bdds()), dtype = torch.int32, device = 'cuda')
        self.cuda_solver.root_indices(self.root_indices_.data_ptr())
        self.cuda_solver.top_sink_indices(self.top_sink_indices_.data_ptr())
        self.cuda_solver.bot_sink_indices(self.bot_sink_indices_.data_ptr())
        self.root_indices_ = self.root_indices_.to(torch.long)
        self.top_sink_indices_ = self.top_sink_indices_.to(torch.long)
        self.bot_sink_indices_ = self.bot_sink_indices_.to(torch.long)

        self.lo_costs_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.float32, device = 'cuda')
        self.hi_costs_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.float32, device = 'cuda')
        self.def_mm_ = torch.empty((self.cuda_solver.nr_layers()), dtype = torch.float32, device = 'cuda')

        self.cuda_solver.get_solver_costs(self.lo_costs_.data_ptr(), self.hi_costs_.data_ptr(), self.def_mm_.data_ptr())
        self.cost_from_root_ = torch.empty((self.cuda_solver.nr_bdd_nodes()), dtype = torch.float32, device = 'cuda')
        self.cost_from_terminal_ = torch.zeros((self.cuda_solver.nr_bdd_nodes()), dtype = torch.float32, device = 'cuda')
        # Set costs of bot sinks to top to infinity:
        self.cost_from_terminal_[self.bot_sink_indices_] = float("Inf")

    def nr_hops(self):
        return len(self.cum_nr_layers_per_hop_dist_) - 1 # ignores terminal nodes.

    def start_end_bdd_node_indices(self, hop_index):
        if hop_index == 0:
            return 0, self.cum_nr_bdd_nodes_per_hop_dist_[0]
        else:
            return self.cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1], self.cum_nr_bdd_nodes_per_hop_dist_[hop_index]

    def flush_costs_from_root(self):
        self.cost_from_root_[:] = float("Inf")
        self.cost_from_root_[self.root_indices_] = 0.0

    def get_arc_costs_bdd_node(self, hop_index):
        bdd_node_start, bdd_node_end = self.start_end_bdd_node_indices(hop_index)
        node_to_layer_map = self.bdd_node_to_layer_map_[bdd_node_start: bdd_node_end]
        return self.lo_costs_[node_to_layer_map], self.hi_costs_[node_to_layer_map], self.def_mm_[node_to_layer_map]

    def get_hop_data(self, hop_index):
        start_node, end_node = self.start_end_bdd_node_indices(hop_index)
        lo_bdd_node_hop = self.lo_bdd_node_[start_node: end_node]
        hi_bdd_node_hop = self.hi_bdd_node_[start_node: end_node]
        lo_cost_hop, hi_cost_hop, def_mm_hop = self.get_arc_costs_bdd_node(hop_index)
        valid_node_mask = lo_bdd_node_hop >= 0 # No outgoing arcs from terminal nodes.
        return start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, def_mm_hop, valid_node_mask

    def forward_run(self):
        self.flush_costs_from_root()
        for hop_index in range(self.nr_hops()):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, def_mm_hop, valid_node_mask = self.get_hop_data(hop_index)
            cost_from_root_hop = self.cost_from_root_[start_node:end_node]

            next_cost_to_lo = cost_from_root_hop[valid_node_mask] + lo_cost_hop[valid_node_mask]
            next_cost_to_hi = cost_from_root_hop[valid_node_mask] + hi_cost_hop[valid_node_mask]
            
            scatter_min(next_cost_to_lo, lo_bdd_node_hop[valid_node_mask], out = self.cost_from_root_)            
            scatter_min(next_cost_to_hi, hi_bdd_node_hop[valid_node_mask], out = self.cost_from_root_)
        return self.cost_from_root_

    def backward_run(self):
        for hop_index in reversed(range(self.nr_hops())):
            start_node, end_node, lo_bdd_node_hop, hi_bdd_node_hop, lo_cost_hop, hi_cost_hop, def_mm_hop, valid_node_mask = self.get_hop_data(hop_index)

            next_lo_cost_from_terminal = self.cost_from_terminal_[lo_bdd_node_hop[valid_node_mask]] + lo_cost_hop[valid_node_mask]
            next_hi_cost_from_terminal = self.cost_from_terminal_[hi_bdd_node_hop[valid_node_mask]] + hi_cost_hop[valid_node_mask]
            hop_indices = torch.arange(start_node, end_node, dtype = torch.long)
            self.cost_from_terminal_[hop_indices[valid_node_mask]] = torch.minimum(next_lo_cost_from_terminal, next_hi_cost_from_terminal)
        return self.cost_from_terminal_
