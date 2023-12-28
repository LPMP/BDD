import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from torch_geometric.nn import TransformerConv, LayerNorm
import model.solver_utils as sol_utils

class EdgeUpdater(torch.nn.Module):
    # Receives learned var, con, edge features and fixed edge features to predict new learned edge features.
    def __init__(self, num_input_edge_channels, num_hidden_channels, num_output_edge_channels, num_var_channels, num_con_channels, num_hidden_layers = 0, use_celu_activation = False):
        super(EdgeUpdater, self).__init__()
        self.activation = nn.ReLU()
        if use_celu_activation:
            self.activation = nn.Softplus()
        self.var_compress_mlp = [nn.Linear(num_var_channels, num_hidden_channels), self.activation]
        for l in range(num_hidden_layers + 1):
            self.var_compress_mlp.append(nn.Linear(num_hidden_channels, num_hidden_channels))
            self.var_compress_mlp.append(self.activation)       
        self.var_compress_mlp = nn.Sequential(*self.var_compress_mlp)

        self.con_compress_mlp = [nn.Linear(num_con_channels, num_hidden_channels), self.activation]
        for l in range(num_hidden_layers + 1):
            self.con_compress_mlp.append(nn.Linear(num_hidden_channels, num_hidden_channels))
            self.con_compress_mlp.append(self.activation)
        self.con_compress_mlp = nn.Sequential(*self.con_compress_mlp)

        self.edge_mlp = [nn.Linear(num_input_edge_channels + num_hidden_channels + num_hidden_channels, num_hidden_channels), self.activation]
        for l in range(num_hidden_layers):
            self.edge_mlp.append(nn.Linear(num_hidden_channels, num_hidden_channels))
            self.edge_mlp.append(self.activation)
        self.edge_mlp.append(nn.Linear(num_hidden_channels, num_output_edge_channels))
        self.edge_mlp = nn.Sequential(*self.edge_mlp)

    def forward(self, var_f: torch.Tensor, con_f: torch.Tensor, combined_edge_f: torch.Tensor, edge_index_var_con: torch.Tensor):
        var_f_c = var_f.clone()
        con_f_c = con_f.clone()
        out = torch.cat([combined_edge_f, 
                        self.var_compress_mlp(var_f_c)[edge_index_var_con[0], :], 
                        self.con_compress_mlp(con_f_c)[edge_index_var_con[1], :]], 1)
        return self.edge_mlp(out)

def compute_normalized_solver_costs_for_dual(solver_state, batch_index_edge, batch_index_con, con_lp_f, con_lp_f_names, norm_type = 'l2', epsilon = 1e-6, max_v = 1e6):
    net_cost = solver_state['hi_costs'] - solver_state['lo_costs'] + solver_state['def_mm']
    if norm_type == 'l2':
        norm = scatter_sum(torch.square(net_cost), batch_index_edge)
    elif norm_type == 'inf':
        norm = scatter_max(torch.abs(net_cost), batch_index_edge)[0]
    else:
        assert False, f'norm: {norm_type} unknown'
    norm = norm + 1e-6
    norm = torch.clamp(norm, max = max_v, min = -max_v)
    norm_edge = norm[batch_index_edge]
    norm_cost = net_cost / norm_edge
    norm_def_mm = solver_state['def_mm'] / norm_edge
    norm_state = {'norm_cost': norm_cost, 'norm_def_mm': norm_def_mm}
    return norm_state, con_lp_f, norm_edge

class FeatureExtractorLayer(torch.nn.Module):
    def __init__(self, 
                num_var_lp_f, in_var_dim, out_var_dim, 
                num_con_lp_f, in_con_dim, out_con_dim,
                num_edge_lp_f_with_ss, in_edge_dim, out_edge_dim,
                use_layer_norm, use_def_mm = True, use_solver_costs = True,
                use_net_solver_costs = False,
                num_hidden_layers_edge = 0, use_celu_activation = False,
                aggr = 'mean'):
        super(FeatureExtractorLayer, self).__init__()

        self.use_def_mm = use_def_mm
        self.use_solver_costs = use_solver_costs
        self.activation = nn.ReLU()
        if use_celu_activation:
            self.activation = nn.Softplus()

        self.con_updater = TransformerConv((num_var_lp_f + in_var_dim, num_con_lp_f + in_con_dim), 
                                            out_con_dim, 
                                            edge_dim = num_edge_lp_f_with_ss + in_edge_dim, 
                                            aggr = aggr)
                                            
        self.var_updater = TransformerConv((out_con_dim + num_con_lp_f, num_var_lp_f + in_var_dim), 
                                            out_var_dim,
                                            edge_dim = num_edge_lp_f_with_ss + in_edge_dim, 
                                            aggr = aggr)

        self.edge_updater = EdgeUpdater(num_edge_lp_f_with_ss + in_edge_dim, out_edge_dim, out_edge_dim, out_var_dim + num_var_lp_f, out_con_dim + num_con_lp_f, num_hidden_layers_edge, use_celu_activation)
        self.use_net_solver_costs = use_net_solver_costs

        if use_layer_norm:
            self.var_norm = LayerNorm(out_var_dim, affine = True)
            self.con_norm = LayerNorm(out_con_dim, affine = True)
            self.edge_norm = LayerNorm(out_edge_dim, affine = True)
        else:
            self.var_norm = torch.nn.Identity()
            self.con_norm = torch.nn.Identity()
            self.edge_norm = torch.nn.Identity()

    def combine_features(self, f1, f2):
        return torch.cat((f1, f2), 1)

    def forward(self, var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con, batch_index_var = None, batch_index_con = None, batch_index_edge = None):
        # 0. Combine learned and fixed features:
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)
        if self.use_solver_costs:
            if self.use_net_solver_costs:
                if self.use_def_mm:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['norm_cost'].unsqueeze(1), solver_state['norm_def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1)
                else:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['norm_cost'].unsqueeze(1), edge_lp_f_wo_ss), 1)
            else:
                if self.use_def_mm:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1)
                else:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), edge_lp_f_wo_ss), 1)
        else:
            assert(not self.use_def_mm)        
            edge_comb_f = torch.cat((edge_learned_f, edge_lp_f_wo_ss), 1)

        # 1. Update constraint features (var, constraint, edge -> constraint_new):
        con_learned_f = self.activation(self.con_norm(self.con_updater((var_comb_f, con_comb_f), edge_index_var_con, edge_comb_f), batch_index_con))
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)

        # 2. Update variable features (var, constraint_new, edge -> var):
        var_learned_f = self.activation(self.var_norm(self.var_updater((con_comb_f, var_comb_f), edge_index_var_con.flip([0]), edge_comb_f), batch_index_var))
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)

        # 3. Update edges:
        edge_learned_f = self.activation(self.edge_norm(self.edge_updater(var_comb_f, con_comb_f, edge_comb_f, edge_index_var_con), batch_index_edge))

        return var_learned_f, con_learned_f, edge_learned_f

def get_edge_mask_without_terminal_nodes(edge_index_var_con, var_degree):
    var_indices = edge_index_var_con[0]
    non_terminal_vars = var_degree != 0
    non_terminal_edges_mask = non_terminal_vars[var_indices]
    return non_terminal_edges_mask

class FeatureExtractor(torch.nn.Module):
    def __init__(self, num_var_lp_f, out_var_dim, num_con_lp_f, out_con_dim, num_edge_lp_f, out_edge_dim, depth, 
                use_layer_norm = False, use_def_mm = True, num_hidden_layers_edge = 0, use_net_solver_costs = False,
                use_solver_costs = True, takes_learned_features = False, use_celu_activation = False, aggr = 'mean'):
        super(FeatureExtractor, self).__init__()
        self.num_var_lp_f = num_var_lp_f
        self.num_con_lp_f = num_con_lp_f
        self.use_def_mm = use_def_mm
        if use_solver_costs:
            num_edge_lp_f_with_ss = num_edge_lp_f + 2 - int(use_net_solver_costs) + int(use_def_mm) 
        else:
            num_edge_lp_f_with_ss = num_edge_lp_f
            self.use_def_mm = False
        self.takes_learned_features = takes_learned_features
        if takes_learned_features:
            self.in_var_dim = out_var_dim
            self.in_con_dim = out_con_dim
            self.in_edge_dim = out_edge_dim
        else:
            self.in_var_dim = 0
            self.in_con_dim = 0
            self.in_edge_dim = 0

        layers = [
            FeatureExtractorLayer(num_var_lp_f, self.in_var_dim, out_var_dim,
                                num_con_lp_f, self.in_con_dim, out_con_dim,
                                num_edge_lp_f_with_ss, self.in_edge_dim, out_edge_dim,
                                use_layer_norm, use_def_mm, 
                                use_solver_costs = use_solver_costs,
                                num_hidden_layers_edge = num_hidden_layers_edge,
                                use_net_solver_costs = use_net_solver_costs,
                                use_celu_activation = use_celu_activation,
                                aggr = aggr)
        ]
        for l in range(depth - 1):
            layers.append(
                FeatureExtractorLayer(num_var_lp_f, out_var_dim, out_var_dim,
                                    num_con_lp_f, out_con_dim, out_con_dim,
                                    num_edge_lp_f_with_ss, out_edge_dim, out_edge_dim,
                                    use_layer_norm, use_def_mm,
                                    use_solver_costs = use_solver_costs,
                                    num_hidden_layers_edge = num_hidden_layers_edge,
                                    use_net_solver_costs = use_net_solver_costs,
                                    use_celu_activation = use_celu_activation,
                                    aggr = aggr)
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, var_lp_f, con_lp_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con, 
                var_learned_f = None, con_learned_f = None, edge_learned_f = None,
                batch_index_var = None, batch_index_con = None, batch_index_edge = None):
        if self.takes_learned_features:
            assert var_learned_f is not None
            assert con_learned_f is not None
            assert edge_learned_f is not None
        else:
            var_learned_f = torch.zeros((var_lp_f.shape[0], 0), device = var_lp_f.device)
            con_learned_f = torch.zeros((con_lp_f.shape[0], 0), device = con_lp_f.device)
            edge_learned_f = torch.zeros((edge_lp_f_wo_ss.shape[0], 0), device = edge_lp_f_wo_ss.device)

        for l in range(len(self.layers)):
            var_learned_f, con_learned_f, edge_learned_f = self.layers[l](
                var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss,
                edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)
            
        return var_learned_f, con_learned_f, edge_learned_f

class DOGEPredictor(torch.nn.Module):
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, history_num_itr = 20,
                use_layer_norm = False, predict_dist_weights = False, predict_omega = False, num_hidden_layers_edge = 0, 
                use_net_solver_costs = False, use_lstm_var = False, free_update = False, free_update_loss_weight = 0.0,
                denormalize_free_update = False, scale_free_update = False, use_celu_activation = False, aggr = 'mean',
                use_solver_costs = True):
        super(DOGEPredictor, self).__init__()
        self.var_lp_f_names = var_lp_f_names
        self.con_lp_f_names = con_lp_f_names
        self.edge_lp_f_names = edge_lp_f_names
        self.num_var_lp_f = len(var_lp_f_names)
        self.num_con_lp_f = len(con_lp_f_names)
        self.num_edge_lp_f_with_ss = len(edge_lp_f_names) + 3 - int(use_net_solver_costs)
        self.use_net_solver_costs = use_net_solver_costs
        self.use_gnn = depth > 0
        if self.use_gnn:
            self.feature_extractor = FeatureExtractor(self.num_var_lp_f, var_dim, self.num_con_lp_f, con_dim, len(edge_lp_f_names), edge_dim, 
                                                    depth, use_layer_norm, use_def_mm = True, num_hidden_layers_edge = num_hidden_layers_edge,
                                                    use_net_solver_costs = use_net_solver_costs, use_celu_activation = use_celu_activation, aggr = aggr,
                                                    use_solver_costs = use_solver_costs)
            
        self.history_num_itr = history_num_itr
        self.predict_dist_weights = predict_dist_weights
        self.predict_omega = predict_omega
        self.free_update = free_update
        self.free_update_loss_weight = free_update_loss_weight
        self.denormalize_free_update = denormalize_free_update
        self.scale_free_update = scale_free_update
        self.maintain_feasibility = False
        assert free_update_loss_weight >= 0
        assert free_update_loss_weight <= 1.0
        if free_update_loss_weight > 0:
            assert self.free_update
        if free_update_loss_weight == 1.0:
            assert not predict_dist_weights
            assert not predict_omega

        num_outputs = int(predict_dist_weights) + int(free_update) + int(predict_omega)
        assert num_outputs > 0
        in_var_dim = var_dim
        self.var_lstm = None
        if use_lstm_var:
            self.var_lstm = nn.LSTMCell(var_dim, var_dim)
            in_var_dim += var_dim
        self.subgradient_step_size = torch.nn.Parameter(torch.zeros(1) + 5e-4)
        if self.use_gnn:
            self.dist_weights_predictor = EdgeUpdater(self.num_edge_lp_f_with_ss + edge_dim, edge_dim, num_outputs, 
                                                self.num_var_lp_f + in_var_dim, 
                                                self.num_con_lp_f + con_dim,
                                                num_hidden_layers_edge, 
                                                use_celu_activation)
        else:
            self.dist_weights_predictor = EdgeUpdater(self.num_edge_lp_f_with_ss, edge_dim, num_outputs, 
                                                self.num_var_lp_f, 
                                                self.num_con_lp_f,
                                                num_hidden_layers_edge, 
                                                use_celu_activation)

    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_lp_f_wo_ss, 
                omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope, valid_edge_mask, 
                batch_index_var, batch_index_con, batch_index_edge,
                var_objective, var_hidden_states_lstm = None, dist_weights_isotropic = None, randomize_num_itrs = False):
        
        assert(not randomize_num_itrs)
        if self.use_net_solver_costs:
            net_solver_state, con_lp_f, norm_edge = compute_normalized_solver_costs_for_dual(solver_state, batch_index_edge, batch_index_con, con_lp_f, self.con_lp_f_names, 'inf')
        if self.use_gnn:
            if self.use_net_solver_costs:
                var_learned_f, con_learned_f, edge_learned_f = self.feature_extractor(var_lp_f, con_lp_f, net_solver_state, edge_lp_f_wo_ss, edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)
            else:
                var_learned_f, con_learned_f, edge_learned_f = self.feature_extractor(var_lp_f, con_lp_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)

            if self.var_lstm is not None:
                lstm_h, lstm_c = self.var_lstm(var_learned_f, (var_hidden_states_lstm['h'], var_hidden_states_lstm['c']))
                var_hidden_states_lstm['h'] = lstm_h
                var_hidden_states_lstm['c'] = lstm_c
                var_features = torch.cat((var_learned_f, var_lp_f, lstm_h), 1)
            else:
                var_features = torch.cat((var_learned_f, var_lp_f), 1)

            if not self.use_net_solver_costs:
                predictions = self.dist_weights_predictor(
                    var_features, torch.cat((con_learned_f, con_lp_f), 1),
                    torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
                    edge_index_var_con)
            else:
                predictions = self.dist_weights_predictor(
                    var_features, torch.cat((con_learned_f, con_lp_f), 1),
                    torch.cat((edge_learned_f, net_solver_state['norm_cost'].unsqueeze(1), net_solver_state['norm_def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
                    edge_index_var_con)
        else:
            if self.use_net_solver_costs:
                predictions = self.dist_weights_predictor(
                    var_lp_f, con_lp_f,
                    torch.cat((net_solver_state['norm_cost'].unsqueeze(1), net_solver_state['norm_def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
                    edge_index_var_con)
            else:
                predictions = self.dist_weights_predictor(
                    var_lp_f.clone(), con_lp_f.clone(),
                    torch.cat((solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss.clone()), 1), 
                    edge_index_var_con)

        if self.predict_dist_weights:
            dist_weights = predictions[:, 0]
            dist_weights = sol_utils.normalize_distribution_weights_softmax(dist_weights, edge_index_var_con)
        else:
            assert dist_weights_isotropic.shape == solver_state['hi_costs'].shape
            dist_weights = dist_weights_isotropic

        lb_after_dist_free_update = None
        if self.free_update:
            update = predictions[:, int(self.predict_dist_weights)] * torch.abs(self.subgradient_step_size.to(var_lp_f.device))
            update = (update - scatter_mean(update.to(torch.float64), edge_index_var_con[0])[edge_index_var_con[0]]).to(torch.get_default_dtype())
            if 'lb_change_free_update' in self.con_lp_f_names and self.scale_free_update:
                lb_change_per_con = con_lp_f[:, self.con_lp_f_names.index('lb_change_free_update')]
                lb_change_per_edge = lb_change_per_con[edge_index_var_con[1]]
                mean_lb_change_per_var = scatter_mean(lb_change_per_edge, edge_index_var_con[0])
                update = update * mean_lb_change_per_var[edge_index_var_con[0]]
            if self.use_net_solver_costs and self.denormalize_free_update:
                update = norm_edge * update
            solver_state['hi_costs'] = solver_state['hi_costs'] + update
            if self.maintain_feasibility and not update.requires_grad:
                solver_state_dd = sol_utils.distribute_delta(solvers, solver_state)
                net_objective = scatter_sum(solver_state_dd['hi_costs'].to(torch.float64) - solver_state_dd['lo_costs'].to(torch.float64), edge_index_var_con[0]) 
                objective_diff = var_objective.to(net_objective.device).to(torch.float64) - net_objective
                counts = scatter_sum(torch.ones_like(solver_state_dd['hi_costs']), edge_index_var_con[0])
                objective_diff = (objective_diff / counts).to(torch.get_default_dtype())
                solver_state['hi_costs'] = solver_state['hi_costs'] + objective_diff[edge_index_var_con[0]]

            if 'lb_change_free_update' in self.con_lp_f_names:
                con_lp_f[:, self.con_lp_f_names.index('lb_change_free_update')] = (
                    sol_utils.compute_per_bdd_lower_bound(solvers, solver_state) - con_lp_f[:, self.con_lp_f_names.index('lb')])

            if self.free_update_loss_weight > 0 and update.requires_grad:
                solver_state_dd = sol_utils.distribute_delta(solvers, solver_state)
                lb_after_dist_free_update = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state_dd)

        omega_vec = None
        if num_dual_iterations > 0:

            if solver_state['hi_costs'].requires_grad:
                solver_state['hi_costs'].register_hook(lambda x: x.clamp(min=-10, max=10))
            if solver_state['lo_costs'].requires_grad:
                solver_state['lo_costs'].register_hook(lambda x: x.clamp(min=-10, max=10))
            if solver_state['def_mm'].requires_grad:
                solver_state['def_mm'].register_hook(lambda x: x.clamp(min=-10, max=10))
            if dist_weights.requires_grad:
                dist_weights.register_hook(lambda x: x.clamp(min=-10, max=10))

            if self.predict_omega:
                omega_vec = torch.sigmoid(predictions[:, int(self.predict_dist_weights) + int(self.free_update)])
                if omega_vec.requires_grad:
                    omega_vec.register_hook(lambda x: x.clamp(min=-10, max=10))
                # Dual iterations
                solver_state, sol_avg, lb_first_order_avg, lb_sec_order_avg = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, 
                                                                        omega_vec, dual_improvement_slope, grad_dual_itr_max_itr, self.history_num_itr, 0.9, randomize_num_itrs)
            else:
                # Dual iterations
                solver_state, sol_avg, lb_first_order_avg, lb_sec_order_avg = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, 
                                                                        omega, dual_improvement_slope, grad_dual_itr_max_itr, self.history_num_itr, 0.9, randomize_num_itrs)

        if 'prev_dist_weights' in self.edge_lp_f_names:
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_dist_weights')] = dist_weights.clone().detach()
        elif 'dist_weights' in self.edge_lp_f_names:
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('dist_weights')] = dist_weights.clone().detach()
        if 'omega' in self.edge_lp_f_names and self.predict_omega:
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('omega')] = omega_vec.clone().detach()

        if 'mm_diff' in self.edge_lp_f_names:
            if solver_state['lo_costs'].requires_grad:
                solver_state_temp = {'lo_costs': solver_state['lo_costs'].clone(), 'hi_costs': solver_state['hi_costs'].clone(), 'def_mm': solver_state['def_mm'].clone()}
                solver_state_dd = sol_utils.distribute_delta(solvers, solver_state_temp)
            else:
                solver_state_dd = sol_utils.distribute_delta(solvers, solver_state)
            mm_diff = sol_utils.compute_all_min_marginal_diff(solvers, solver_state_dd)
            mm_diff[~valid_edge_mask] = 0
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('mm_diff')] = mm_diff

        with torch.no_grad():
            if 'prev_sol' in self.edge_lp_f_names:
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_sol')] = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')].clone()
            if 'prev_sol_avg' in self.edge_lp_f_names:
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_sol_avg')] = sol_avg
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
            if 'prev_lb' in self.con_lp_f_names:
                con_lp_f[:, self.con_lp_f_names.index('prev_lb')] = con_lp_f[:, self.con_lp_f_names.index('lb')].clone()
            if 'lb_first_order_avg' in self.con_lp_f_names:
                con_lp_f[:, self.con_lp_f_names.index('lb_first_order_avg')] = lb_first_order_avg
            if 'lb_sec_order_avg' in self.con_lp_f_names:
                con_lp_f[:, self.con_lp_f_names.index('lb_sec_order_avg')] = lb_sec_order_avg
            if 'grad_dist_weights' in self.edge_lp_f_names:
                dist_weights_grad, omega_vec_grad = sol_utils.populate_grad_features_dual(solvers, solver_state, dist_weights, omega_vec, 20, batch_index_con)
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('grad_dist_weights')] = dist_weights_grad
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('grad_omega')] = omega_vec_grad
            edge_lp_f_wo_ss = sol_utils.populate_smooth_solution_features(solvers, self.edge_lp_f_names, solver_state, edge_lp_f_wo_ss, valid_edge_mask)
        con_lp_f[:, self.con_lp_f_names.index('lb')] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
        return solver_state, var_lp_f, con_lp_f, edge_lp_f_wo_ss, dist_weights, omega_vec, var_hidden_states_lstm, lb_after_dist_free_update