import torch
import pickle
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_add, scatter_std, scatter_max
import bdd_cuda_torch
import numpy as np

def set_features(f_name, feature_names_to_use, feature_tensor, feature_vector):
    for (i, name) in enumerate(feature_names_to_use):
        if f_name == name:
            feature_tensor[:, i] = feature_vector
    return feature_tensor

def init_solver_and_get_states(batch, device, gt_solution_type, var_lp_f_names, con_lp_f_names, edge_lp_f_names, 
                            num_iterations = 20, improvement_slope = 0.0, omega = 0.5, distribute_deltaa = False, 
                            num_grad_iterations_dual_features = 0, compute_history_for_itrs = 20, avg_sol_beta = 0.9,
                            replay_buffer = None, replay_buffer_start_round = None, smoothing_coeffs = [],
                            use_lstm_var = False, num_learned_var_f = 0):
    batch.edge_index_var_con = batch.edge_index_var_con.to(device) 
    batch.batch_index_var = batch.objective_batch.to(device) # var batch assignment
    batch.batch_index_con = batch.rhs_vector_batch.to(device) # con batch assignment
    batch.batch_index_edge = batch.edge_index_var_con_batch.to(device) # edge batch assignment
    batch.num_cons = batch.num_cons.to(device)
    num_edges = batch.edge_index_var_con.shape[1]
    batch.omega = torch.tensor([omega], device = device)
    lo_costs_out = torch.empty((num_edges), device = device, dtype = torch.get_default_dtype())
    hi_costs_out = torch.empty_like(lo_costs_out)
    def_mm_out = torch.empty_like(lo_costs_out)
    solvers = []
    layer_start = 0
    for b in range(len(batch.solver_data)):
        solver = pickle.loads(batch.solver_data[b])
        batch.solver_data[b] = None # Free-up the memory.
        solver.distribute_delta()
        solver.get_solver_costs(lo_costs_out[layer_start].data_ptr(), hi_costs_out[layer_start].data_ptr(), def_mm_out[layer_start].data_ptr())
        layer_start += solver.nr_layers()
        solvers.append(solver)
    initial_lbs = [solver.lower_bound() for solver in solvers]
    batch.sampled_trajectory_locations = [0] * len(solvers)
    if use_lstm_var:
        batch.var_hidden_states_lstm = {'h': torch.zeros((torch.numel(batch.objective), num_learned_var_f), device = device),
                                        'c': torch.zeros((torch.numel(batch.objective), num_learned_var_f), device = device)}
    else:
        batch.var_hidden_states_lstm = torch.empty((0, 0))

    var_start, layer_start = 0, 0
    for (b, solver) in enumerate(solvers):
        # if replay_buffer_start_round = 0 we are at initial point anyway.
        sampled = False
        num_var_nodes = solver.nr_primal_variables() + 1 # +1 to account for BDD terminal nodes.
        if replay_buffer is not None and batch.file_path[b] in replay_buffer and replay_buffer_start_round > 0:
            sampled_trajectory = replay_buffer[batch.file_path[b]].sample(replay_buffer_start_round)
            if sampled_trajectory:
                if not use_lstm_var:
                    (lo_costs_t, hi_costs_t, def_mm_t), sampled_index = sampled_trajectory
                else:
                    (lo_costs_t, hi_costs_t, def_mm_t, lstm_h_t, lstm_c_t), sampled_index = sampled_trajectory
                    batch.var_hidden_states_lstm['h'][var_start: var_start + num_var_nodes] = lstm_h_t.to(device)
                    batch.var_hidden_states_lstm['c'][var_start: var_start + num_var_nodes] = lstm_c_t.to(device)
                batch.sampled_trajectory_locations[b] = sampled_index
                lo_costs_t = lo_costs_t.to(device).contiguous()
                hi_costs_t = hi_costs_t.to(device).contiguous()
                def_mm_t = def_mm_t.to(device).contiguous()
                solver.set_solver_costs(lo_costs_t.data_ptr(), hi_costs_t.data_ptr(), def_mm_t.data_ptr())
                solver.get_solver_costs(lo_costs_out[layer_start].data_ptr(), hi_costs_out[layer_start].data_ptr(), def_mm_out[layer_start].data_ptr())
                sampled = True
        layer_start += solver.nr_layers()
        var_start += num_var_nodes

    solver_state = {'lo_costs': lo_costs_out, 'hi_costs': hi_costs_out, 'def_mm': def_mm_out}
    if distribute_deltaa:
        solver_state = distribute_delta(solvers, solver_state)
    # dual_feasbility_check(solvers, solver_state, batch.objective.to(device), batch.num_vars, distribute_delta_before = not distribute_deltaa)
    prev_per_bdd_lb = compute_per_bdd_lower_bound(solvers, solver_state)
    prev_per_bdd_sol = compute_per_bdd_solution(solvers, solver_state)
    dist_weights = normalize_distribution_weights(torch.ones_like(lo_costs_out), batch.edge_index_var_con)

    # Run non-learned solver for num_iterations to build history.
    solver_state, sol_avg, lb_first_order_avg, lb_sec_order_avg = dual_iterations(solvers, solver_state, dist_weights, num_iterations, batch.omega, improvement_slope, 
                                            grad_dual_itr_max_itr = None, compute_history_for_itrs = compute_history_for_itrs, avg_sol_beta = avg_sol_beta)

    if distribute_deltaa:
        solver_state = distribute_delta(solvers, solver_state)

    per_bdd_lb = compute_per_bdd_lower_bound(solvers, solver_state)
    per_bdd_sol = compute_per_bdd_solution(solvers, solver_state)
    valid_edge_mask_list = get_valid_edge_mask(batch.edge_index_var_con, solvers)
    if gt_solution_type is not None:
        gt_sol_var, gt_sol_edge = get_valid_target_solution_edge(solvers, batch.num_vars, batch.edge_index_var_con[0], batch.gt_info, valid_edge_mask_list, gt_solution_type)
    else:
        gt_sol_var = None
        gt_sol_edge = None
    valid_edge_mask = torch.cat(valid_edge_mask_list, 0)
    batch.valid_edge_mask = valid_edge_mask
    solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
    batch.gt_sol_edge = gt_sol_edge
    batch.gt_sol_var = gt_sol_var
    batch.initial_lbs = initial_lbs
    batch.solver_state = solver_state
    batch.solvers = solvers
    mm_diff = compute_all_min_marginal_diff(solvers, distribute_delta(solvers, solver_state))
    mm_diff[~batch.valid_edge_mask] = 0
    try:
        assert(torch.all(torch.isfinite(solver_state['lo_costs'])))
        assert(torch.all(torch.isfinite(solver_state['hi_costs'])))
        assert(torch.all(torch.isfinite(solver_state['def_mm'])))
        assert(torch.all(torch.isfinite(mm_diff)))
    except:
        breakpoint()

    # Gradient features:
    # if num_grad_iterations_dual_features > 0:
    #     omega_vec = torch.zeros_like(per_bdd_sol) + omega
    #     dist_weights_grad, omega_vec_grad = populate_grad_features_dual(solvers, solver_state, dist_weights, omega_vec, num_grad_iterations_dual_features, batch.batch_index_con)

    # Variable LP features:
    var_degree = scatter_add(torch.ones((batch.num_edges.sum()), device=device), batch.edge_index_var_con[0])
    var_degree[torch.cumsum(batch.num_vars, 0) - 1] = 0 # Terminal nodes, not corresponding to any primal variable.
    batch.var_lp_f = torch.zeros((torch.numel(batch.objective), len(var_lp_f_names)), device = device, dtype = torch.get_default_dtype())
    batch.var_lp_f = set_features('obj', var_lp_f_names, batch.var_lp_f, batch.objective.to(device))
    batch.var_lp_f = set_features('deg', var_lp_f_names, batch.var_lp_f, var_degree)

    # Constraint LP features:
    con_degree = scatter_add(torch.ones((batch.num_edges.sum()), device=device), batch.edge_index_var_con[1])
    batch.con_lp_f = torch.zeros((torch.numel(con_degree), len(con_lp_f_names)), device = device, dtype = torch.get_default_dtype())
    batch.con_lp_f = set_features('deg', con_lp_f_names, batch.con_lp_f, con_degree)
    batch.con_lp_f = set_features('rhs', con_lp_f_names, batch.con_lp_f, batch.rhs_vector.to(device))
    batch.con_lp_f = set_features('lb', con_lp_f_names, batch.con_lp_f, per_bdd_lb)
    batch.con_lp_f = set_features('prev_lb', con_lp_f_names, batch.con_lp_f, prev_per_bdd_lb)
    if 'lb_change' in con_lp_f_names:
        batch.con_lp_f = set_features('lb_change', con_lp_f_names, batch.con_lp_f, per_bdd_lb - prev_per_bdd_lb)

    if 'lb_first_order_avg' in con_lp_f_names:
        assert compute_history_for_itrs >= 2
        assert num_iterations >= 2

    if 'lb_sec_order_avg' in con_lp_f_names:
        assert compute_history_for_itrs >= 3
        assert num_iterations >= 3

    batch.con_lp_f = set_features('lb_first_order_avg', con_lp_f_names, batch.con_lp_f, lb_first_order_avg)
    batch.con_lp_f = set_features('lb_sec_order_avg', con_lp_f_names, batch.con_lp_f, lb_sec_order_avg)
    batch.con_lp_f = set_features('con_type', con_lp_f_names, batch.con_lp_f, batch.con_type.to(device))
    
    # Edge LP features:
    batch.edge_rest_lp_f = torch.zeros((torch.numel(dist_weights), len(edge_lp_f_names)), device = device, dtype = torch.get_default_dtype())
    batch.edge_rest_lp_f = set_features('sol', edge_lp_f_names, batch.edge_rest_lp_f, per_bdd_sol)
    batch.edge_rest_lp_f = set_features('prev_sol', edge_lp_f_names, batch.edge_rest_lp_f, prev_per_bdd_sol)
    batch.edge_rest_lp_f = set_features('prev_sol_avg', edge_lp_f_names, batch.edge_rest_lp_f, sol_avg)
    batch.edge_rest_lp_f = set_features('dist_weights', edge_lp_f_names, batch.edge_rest_lp_f, dist_weights)
    batch.edge_rest_lp_f = set_features('mm_diff', edge_lp_f_names, batch.edge_rest_lp_f, mm_diff)
    batch.edge_rest_lp_f = set_features('coeff', edge_lp_f_names, batch.edge_rest_lp_f, batch.con_coeff.to(device))
    batch.edge_rest_lp_f = set_features('omega', edge_lp_f_names, batch.edge_rest_lp_f, torch.zeros_like(per_bdd_sol) + omega)
    batch.edge_rest_lp_f = populate_smooth_solution_features(batch.solvers, edge_lp_f_names, batch.solver_state, batch.edge_rest_lp_f, batch.valid_edge_mask)
    # if num_grad_iterations_dual_features > 0:
    #     batch.edge_rest_lp_f = set_features('grad_dist_weights', edge_lp_f_names, batch.edge_rest_lp_f, dist_weights_grad)
    #     batch.edge_rest_lp_f = set_features('grad_omega', edge_lp_f_names, batch.edge_rest_lp_f, omega_vec_grad)
    return batch, dist_weights

def populate_smooth_solution_features(solvers, edge_lp_f_names, solver_state, edge_rest_lp_f, valid_edge_mask):
    for fname in edge_lp_f_names:
        if 'smooth_sol' in fname:
            _, sm_coeff = fname.split("@")
            prob_hi = compute_smoothed_solution(solvers, solver_state, valid_edge_mask, float(sm_coeff) * 100.0)
            edge_rest_lp_f = set_features(fname, edge_lp_f_names, edge_rest_lp_f, prob_hi)
            # torch.set_printoptions(edgeitems = 20)
            # print(fname)
            # print(prob_hi)
            # print('')
    return edge_rest_lp_f

def compute_normalized_solver_costs_for_primal(solver_state, mm_diff, lb, edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge, norm_type = 'inf'):
    net_edge_cost = solver_state['hi_costs'] - solver_state['lo_costs']
    net_var_cost = scatter_sum(net_edge_cost, edge_index_var_con[0])
    if norm_type == 'l2':
        norm = scatter_sum(torch.square(net_var_cost), batch_index_var)
    elif norm_type == 'inf':
        norm = scatter_max(torch.abs(net_var_cost), batch_index_var)[0]
    else:
        assert False, f'norm: {norm_type} unknown'
    norm = norm + 1e-9
    net_var_cost = net_var_cost / norm[batch_index_var]
    norm_con = norm[batch_index_con]
    constant_per_bdd = scatter_sum(solver_state['lo_costs'], edge_index_var_con[1]) / norm_con
    lb = lb / norm_con
    norm_edge = norm[batch_index_edge]
    mm_diff = mm_diff / norm_edge
    net_edge_cost = net_edge_cost / norm_edge
    return net_var_cost, net_edge_cost, mm_diff, lb, constant_per_bdd, norm

def non_learned_iterations(solvers, solver_state, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
    layer_start = 0
    for (b, solver) in enumerate(solvers):  
        solver.set_solver_costs(solver_state['lo_costs'][layer_start].data_ptr(), 
                                solver_state['hi_costs'][layer_start].data_ptr(), 
                                solver_state['def_mm'][layer_start].data_ptr())
        solver.non_learned_iterations(omega, num_iterations, improvement_slope)
        solver.get_solver_costs(solver_state['lo_costs'][layer_start].data_ptr(), 
                                solver_state['hi_costs'][layer_start].data_ptr(), 
                                solver_state['def_mm'][layer_start].data_ptr())
        layer_start += solver.nr_layers()
    return solvers, solver_state

def non_learned_updates(batch, edge_lp_f_names, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
    batch.solvers, batch.solver_state = non_learned_iterations(batch.solvers, batch.solver_state, num_iterations, improvement_slope, omega)
    # Update lower bounds:
    batch.con_lp_f[:, 0] = compute_per_bdd_lower_bound(batch.solvers, batch.solver_state) 
    # Update LP feature information so that GNN can be run afterwards.
    per_bdd_sol = compute_per_bdd_solution(batch.solvers, batch.solver_state)
    dist_weights = normalize_distribution_weights(torch.ones_like(batch.solver_state['lo_costs']), batch.edge_index_var_con)

    batch.edge_rest_lp_f = set_features('sol', edge_lp_f_names, batch.edge_rest_lp_f, per_bdd_sol)
    batch.edge_rest_lp_f = set_features('prev_dist_weights', edge_lp_f_names, batch.edge_rest_lp_f, dist_weights)

    batch.solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
    return batch

# Find edges which correspond to valid primal variables (as each BDD contains 1 invalid primal variable each.)
def get_valid_target_solution_edge(solvers, num_vars, var_indices, gt_info, valid_edge_mask_list, solution_type):
    assert(solution_type == 'lp_stats' or solution_type == 'ilp_stats')
    gt_sol_var = []
    gt_sol_edge = []
    layer_start = 0
    var_offset = 0
    for b in range(len(num_vars)):
        sol = gt_info[solution_type]['sol'][b]
        if sol is not None:
            current_var_sol = torch.from_numpy(sol).to(torch.get_default_dtype()).to(var_indices.device)
            gt_sol_var.append(current_var_sol)
            current_var_indices = var_indices[layer_start: layer_start + solvers[b].nr_layers()] - var_offset
            current_edge_sol = current_var_sol[current_var_indices]
            current_edge_sol_valid = current_edge_sol[valid_edge_mask_list[b]]
            gt_sol_edge.append(current_edge_sol_valid)
        else:
            gt_sol_var.append(None)
            gt_sol_edge.append(None)
        layer_start += solvers[b].nr_layers()
        var_offset += num_vars[b]
    return gt_sol_var, gt_sol_edge 

def get_valid_edge_mask(edge_index_var_con, solvers):
    masks_list = [] #torch.ones(batch.num_edges, dtype=torch.bool)
    for (b, solver) in enumerate(solvers):
        terminal_indices = torch.empty((solver.nr_bdds()), device = edge_index_var_con.device, dtype = torch.int32)
        solver.terminal_layer_indices(terminal_indices.data_ptr())
        current_mask = torch.ones(solver.nr_layers(), dtype = torch.bool)
        current_mask[terminal_indices.to(torch.long)] = False
        masks_list.append(current_mask)
    return masks_list

def compute_per_bdd_lower_bound(solvers, solver_state):
    return bdd_cuda_torch.ComputeLowerBoundperBDD.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'])

def compute_per_bdd_solution(solvers, solver_state):
    sol = bdd_cuda_torch.ComputePerBDDSolutions(solvers, solver_state['lo_costs'], solver_state['hi_costs'])
    return sol

def distribute_delta(solvers, solver_state):
    lo_costs, hi_costs = bdd_cuda_torch.DistributeDeferredDelta.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'])
    return {'lo_costs': lo_costs, 'hi_costs': hi_costs, 'def_mm': torch.zeros_like(solver_state['def_mm'])}

def dual_iterations(solvers, solver_state, dist_weights, num_iterations, omega, improvement_slope = 1e-6, 
                    grad_dual_itr_max_itr = None, compute_history_for_itrs = 20, avg_sol_beta = 0.9, randomize_num_itrs = False):
    # Following two lines use solver written fully in pytorch and so uses pytorch autodiff to compute gradients. 
    # Not using it by default as it is memory intensive for large instances especially with num_iterations >> 1.
    # if solver_state['hi_costs'].requires_grad or omega.requires_grad or dist_weights.requires_grad:
    #     return dual_iterations_torch(solvers, solver_state, dist_weights, num_iterations, omega, [0], 'cuda')
    assert num_iterations >= compute_history_for_itrs
    if grad_dual_itr_max_itr is None:
        grad_dual_itr_max_itr = num_iterations
    num_caches = min(int(grad_dual_itr_max_itr / 3), 25)
    lo_costs, hi_costs, def_mm, sol_avg, lb_first_order_avg, lb_sec_order_avg = bdd_cuda_torch.DualIterations.apply(solvers, 
                                                                    solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'], 
                                                                    dist_weights, num_iterations, omega, grad_dual_itr_max_itr, improvement_slope, num_caches,
                                                                    compute_history_for_itrs, avg_sol_beta, randomize_num_itrs)
    solver_state['lo_costs'] = lo_costs
    solver_state['hi_costs'] = hi_costs
    solver_state['def_mm'] = def_mm
    # Return updated solver_state, average of last 'compute_history_for_itrs'-many per BDD solutions, 
    # smoothed first order difference of lb and smoothed second order difference of lb.
    return solver_state, sol_avg, lb_first_order_avg, lb_sec_order_avg

def dual_iterations_torch(cuda_solvers, solver_state, dist_weights, num_iterations, omega, lb_shape, device, smoothing = None):
    sol_avg = torch.zeros_like(solver_state['lo_costs'])
    lb_first_order_avg = torch.zeros((lb_shape), device = 'cuda')
    lb_sec_order_avg = torch.zeros((lb_shape), device = 'cuda')

    layer_start = 0
    lo_costs_in = solver_state['lo_costs'].to(device)
    hi_costs_in = solver_state['hi_costs'].to(device)
    def_mm_in = solver_state['def_mm'].to(device)
    dist_weights_in = dist_weights.to(device)
    if omega.numel() == 1:
        omega_in = torch.zeros_like(hi_costs_in) + omega.item()
    else:
        omega_in = omega.to(device)
    lo_costs_out = torch.empty_like(lo_costs_in)
    hi_costs_out = torch.empty_like(hi_costs_in)
    def_mm_out = torch.empty_like(def_mm_in)
    for (b, solver) in enumerate(cuda_solvers): 
        layer_end = layer_start + solver.nr_layers()
        torch_solver = bdd_cuda_torch.bdd_torch_learned_mma(solver, device)
        cur_lo, cur_hi, cur_def_mm = torch_solver.iterations(
            lo_costs_in[layer_start:layer_end], hi_costs_in[layer_start:layer_end], def_mm_in[layer_start:layer_end],
            dist_weights_in[layer_start:layer_end], omega_in[layer_start:layer_end], num_iterations, 
            smoothing = smoothing)
        lo_costs_out[layer_start:layer_end] = cur_lo
        hi_costs_out[layer_start:layer_end] = cur_hi
        def_mm_out[layer_start:layer_end] = cur_def_mm
        layer_start += solver.nr_layers()
    solver_state['lo_costs'] = lo_costs_out.to('cuda')
    solver_state['hi_costs'] = hi_costs_out.to('cuda')
    solver_state['def_mm'] = def_mm_out.to('cuda')
    return solver_state, sol_avg, lb_first_order_avg, lb_sec_order_avg

def compute_all_min_marginal_diff(solvers, solver_state):
    mm_diff = bdd_cuda_torch.ComputeAllMinMarginalsDiff.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'])
    return mm_diff

def normalize_distribution_weights(dist_weights, edge_index_var_con):
# Normalize distribution weights so that they sum upto 1 for each variable.
    var_indices = edge_index_var_con[0, :]
    dist_weights_sum = scatter_sum(dist_weights.to(torch.get_default_dtype()), var_indices)[var_indices]
    return (dist_weights / dist_weights_sum).to(torch.get_default_dtype())

def normalize_distribution_weights_softmax(dist_weights, edge_index_var_con):
    var_indices = edge_index_var_con[0, :]
    softmax_scores = scatter_softmax(dist_weights.to(torch.get_default_dtype()), var_indices)
    return softmax_scores.to(torch.get_default_dtype())

def perturb_primal_costs(lo_costs, hi_costs, lo_pert, hi_pert, dist_weights, edge_index_var_con):
    var_indices = edge_index_var_con[0]
    lo_costs = lo_costs + lo_pert[var_indices] * dist_weights
    hi_costs = hi_costs + hi_pert[var_indices] * dist_weights
    return lo_costs, hi_costs

# Assumes that dual update is performed using only hi_costs and lo_costs remain 0.
# Finds weights w s.t. dual_cost_ij = w_ij * c_i => w_ij = dual_cost_ij / c_i for variable i and constraint j.
def compute_overall_dist_weights(var_objective, dual_costs, edge_index_var_con):
    var_indices = edge_index_var_con[0]
    edge_objective = var_objective[var_indices]
    return dual_costs / (edge_objective + 1e-6)

def normalize_objective_batch(objective_batch, batch_indices):
    norm_batch = torch.sqrt(scatter_sum(torch.square(objective_batch), batch_indices))
    normalized_objective = objective_batch / norm_batch[batch_indices]
    return normalized_objective, norm_batch

def primal_rounding_non_learned(num_rounds, solvers, norm_solver_state, obj_multiplier, obj_offset, num_itr_lb, improvement_slope_lb, omega, edge_index_var_con, dist_weights, init_delta = 1.0, delta_growth_rate = 1.2):
    assert len(solvers) == 1, 'batch size 1 supported for now.'
    hi_costs = norm_solver_state['hi_costs'] / obj_multiplier[0].item() + obj_offset[0].item()
    lo_costs = norm_solver_state['lo_costs'] / obj_multiplier[0].item() + obj_offset[0].item()
    def_mm = torch.zeros_like(lo_costs)
    solution_cpu = bdd_cuda_torch.ComputePrimalSolution(solvers, lo_costs, hi_costs, def_mm, init_delta, delta_growth_rate, 500)[0]
    dummy_mm_diff = torch.zeros_like(hi_costs)
    var_indices = edge_index_var_con[0]
    if len(solution_cpu) > 0:
        solution_cpu.append(0) # for terminal node.
        dummy_mm_diff = -torch.FloatTensor(solution_cpu).to(hi_costs.device) + 0.5
        dummy_mm_diff = dummy_mm_diff[var_indices]
    logs = []
    logs.append({'r' : 0, 'all_mm_diff': dummy_mm_diff})
    return dummy_mm_diff, None, logs

# def primal_rounding_non_learned(num_rounds, solvers, norm_solver_state, obj_multiplier, obj_offset, num_itr_lb, improvement_slope_lb, omega, edge_index_var_con, dist_weights, init_delta = 1.0, delta_growth_rate = 1.2):
#     assert len(solvers) == 1, 'batch size 1 supported for now.'
#     hi_costs = norm_solver_state['hi_costs'] / obj_multiplier[0].item() + obj_offset[0].item()
#     lo_costs = norm_solver_state['lo_costs'] / obj_multiplier[0].item() + obj_offset[0].item()
#     solver_state = {'hi_costs': hi_costs, 'lo_costs': lo_costs, 'def_mm': torch.zeros_like(lo_costs)}
#     logs = []
#     for round_index in range(num_rounds):
#         var_indices = edge_index_var_con[0]
#         current_delta = min(init_delta * np.power(delta_growth_rate, round_index), 1e6)

#         mm_diff = compute_all_min_marginal_diff(solvers, solver_state)
#         logs.append({'r' : round_index, 'all_mm_diff': mm_diff.detach()})

#         hi_assignments = mm_diff < -1e-6
#         lo_assignments = mm_diff > 1e-6

#         undecided_assigments = torch.logical_and(~hi_assignments, ~lo_assignments)
#         var_hi = scatter_mean(hi_assignments.to(torch.get_default_dtype()), var_indices) >= 1.0 - 1e-6
#         var_lo = scatter_mean(lo_assignments.to(torch.get_default_dtype()), var_indices) >= 1.0 - 1e-6
#         var_lo[-1] = True # terminal node.
#         if (var_hi + var_lo).min() >= 1.0 - 1e-6: # Solution found
#             return mm_diff, var_hi, logs
        
#         var_undecided = scatter_mean(undecided_assigments.to(torch.get_default_dtype()), var_indices) >= 1.0 - 1e-6
#         var_inconsistent = torch.logical_and(torch.logical_and(~var_hi, ~var_lo), ~var_undecided)

#         rand_perturbation = 2.0 * current_delta * torch.rand(var_undecided.shape, dtype = torch.get_default_dtype(), device = mm_diff.device) - current_delta
#         var_undecided_pert = var_undecided * rand_perturbation

#         mm_diff_sum_sign = torch.sign(scatter_sum(mm_diff, var_indices))
#         var_inconsistent_pert = var_inconsistent * mm_diff_sum_sign * torch.abs(rand_perturbation)
#         var_cost_pert = -(var_hi * current_delta) + (var_lo * current_delta) + (var_undecided_pert) + (var_inconsistent_pert)
#         var_cost_pert[-1] = 0 # dummy primal variable for terminal nodes.
#         solver_state['lo_costs'], solver_state['hi_costs'] = perturb_primal_costs(solver_state['lo_costs'], 
#                                                                                 solver_state['hi_costs'], 
#                                                                                 torch.relu(-var_cost_pert),
#                                                                                 torch.relu(var_cost_pert),
#                                                                                 dist_weights,
#                                                                                 edge_index_var_con)
#         solvers, solver_state = non_learned_iterations(solvers, solver_state, num_itr_lb, improvement_slope_lb, omega)
#         solver_state = distribute_delta(solvers, solver_state)
#         # print(f'one min-marg diff.: {var_hi.sum():.0f}, '
#         #     f'zero min-marg diff.: {var_lo.sum():.0f}, '
#         #     f'equal min-marg diff.: {var_undecided.sum():.0f}, '
#         #     f'inconsistent min-marg diff.: {var_inconsistent.sum():.0f}')
#     return mm_diff, None, logs

def compute_smoothed_solution(solvers, solver_state, valid_edge_mask, smoothing_coeff):
    lo_costs = solver_state['lo_costs'].detach().clone()
    hi_costs = solver_state['hi_costs'].detach().clone()
    solver_state = distribute_delta(solvers, solver_state)
    smooth_sol = bdd_cuda_torch.GetMarginalProbability(solvers, lo_costs * smoothing_coeff, hi_costs * smoothing_coeff)
    smooth_sol[~valid_edge_mask] = 0
    return smooth_sol

def dual_feasbility_check(solvers, solver_state, orig_primal_obj, num_vars_per_instance, distribute_delta_before = False, tolerance = 1e-3):
    var_start = 0
    layer_start = 0
    computed_primal_obj = torch.zeros_like(orig_primal_obj)
    if distribute_delta_before:
        solver_state = distribute_delta(solvers, solver_state)
    else:
        assert torch.abs(solver_state['def_mm']).max() < tolerance, "deferred min-marginals should be 0."

    for (b, solver) in enumerate(solvers): 
        var_end = num_vars_per_instance[b] + var_start - 1
        solver.set_solver_costs(solver_state['lo_costs'][layer_start].data_ptr(), 
                                solver_state['hi_costs'][layer_start].data_ptr(), 
                                solver_state['def_mm'][layer_start].data_ptr())
        
        solver.get_primal_objective_vector(computed_primal_obj[var_start].data_ptr())
        layer_start += solver.nr_layers()
        var_start = var_end + 1 # To account for terminal node.
    diff = torch.abs(orig_primal_obj - computed_primal_obj)
    max_difference = diff.max()
    mean_difference = diff.mean()
    if max_difference > tolerance:
        print(f"dual feasibility check: difference: {max_difference} > {tolerance}")
    return max_difference, mean_difference