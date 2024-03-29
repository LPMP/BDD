import torch
torch.set_default_dtype(torch.float32)
import time, os
import numpy as np
import bdd_cuda_torch.bdd_torch_base as bdd_torch_base
from bdd_cuda_torch.bdd_cuda_torch import ComputePerBDDSolutions
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_cuda_solver
# from bdd_cuda_torch import SmoothSolutionGradientCheckpoint

one_simplex = """Minimize
1 x_1 + 2 x_2 + 1 x_3
Subject To
x_1 + x_2 + x_3 = 1
End"""

two_simplex = """Minimize
2 x_1 + 3 x_2 + 4 x_3
+1 x_4 + 2 x_5 - 1 x_6
Subject To  
x_1 + x_2 + x_3 = 2
x_4 + x_5 + x_6 = 1
End"""

matching_3x3 = """Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End"""

short_chain_shuffled = """Minimize
+ 1 mu_2_1 + 1 mu_10 + 0 mu_1_1 + 0 mu_11
-1 mu_1_0 + 1 mu_00 + 2 mu_01 + 2 mu_2_0
Subject To
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_00 + mu_10 + mu_01 + mu_11 = 1
mu_1_0 - mu_00 - mu_01 = 0
mu_1_1 - mu_10 - mu_11 = 0
mu_2_0 - mu_00 - mu_10 = 0
mu_2_1 - mu_01 - mu_11 = 0
End"""

def get_opt_solution(ilp_path, bdd_instance):
    import gurobipy as gp
    ilp_gurobi = gp.read(ilp_path)
    ilp_gurobi.Params.TimeLimit = 60
    ilp_gurobi.optimize()
    solution = torch.zeros(len(ilp_gurobi.getVars()))
    for var in ilp_gurobi.getVars():
        solution[bdd_instance.get_var_index(var.VarName)] = var.X
    return solution

def get_costs(cuda_solver):
    lo_costs = torch.empty((cuda_solver.nr_layers()), dtype = torch.get_default_dtype(), device = 'cuda')
    hi_costs = torch.empty((cuda_solver.nr_layers()), dtype = torch.get_default_dtype(), device = 'cuda')
    def_mm = torch.empty((cuda_solver.nr_layers()), dtype = torch.get_default_dtype(), device = 'cuda')
    cuda_solver.get_solver_costs(lo_costs.data_ptr(), hi_costs.data_ptr(), def_mm.data_ptr())
    return lo_costs, hi_costs, def_mm

def test_forward_run(instance, tolerance = 1e-5):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_base(cuda_solver)
    start = time.time()
    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    valid_mask = torch_solver.valid_bdd_node_mask()
    torch_cost_from_root = torch_solver.forward_run(lo_costs, hi_costs)
    print(torch_cost_from_root)
    smoothing = torch.tensor([1e-6], device = 'cuda')
    torch_cost_from_root_smooth = torch_solver.forward_run(lo_costs, hi_costs, smoothing)
    assert torch.all(torch.abs(torch_cost_from_root[valid_mask] - torch_cost_from_root_smooth[valid_mask]) < tolerance)
    cuda_cost_from_root = torch.empty_like(torch_cost_from_root)
    cuda_solver.cost_from_root(cuda_cost_from_root.data_ptr())
    assert torch.all(torch.abs(torch_cost_from_root[valid_mask] - cuda_cost_from_root[valid_mask]) < 1e-9)

def test_backward_run(instance, tolerance = 1e-5):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_base(cuda_solver)
    start = time.time()
    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    torch_cost_from_terminal = torch_solver.backward_run(lo_costs, hi_costs)
    print(torch_cost_from_terminal)
    smoothing = torch.tensor([1e-6], device = 'cuda')
    torch_cost_from_terminal_smooth = torch_solver.backward_run(lo_costs, hi_costs, smoothing)
    valid_mask = torch_solver.valid_bdd_node_mask()
    assert torch.all(torch.abs(torch_cost_from_terminal[valid_mask] - torch_cost_from_terminal_smooth[valid_mask]) < tolerance)
    cuda_cost_from_terminal = torch.empty_like(torch_cost_from_terminal)
    cuda_solver.cost_from_terminal(cuda_cost_from_terminal.data_ptr())
    assert torch.all(torch.abs(torch_cost_from_terminal[valid_mask] - cuda_cost_from_terminal[valid_mask]) < 1e-9)

def test_lower_bounds(instance, tolerance = 1e-10):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_base(cuda_solver)
    start = time.time()
    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    valid_mask = torch_solver.valid_bdd_node_mask()
    torch_cost_from_terminal = torch_solver.backward_run(lo_costs, hi_costs)
    torch_per_bdd_lb = torch_solver.per_bdd_lower_bound(torch_cost_from_terminal)
    cuda_per_bdd_lb = torch.empty_like(torch_per_bdd_lb)
    cuda_solver.lower_bound_per_bdd(cuda_per_bdd_lb.data_ptr())
    assert torch.all(torch.abs(cuda_per_bdd_lb - torch_per_bdd_lb) < tolerance)

def test_marginals(instance, tolerance = 1e-5):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_base(cuda_solver)
    valid_mask = torch_solver.valid_layer_mask()
    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    cuda_mm_difference = torch.empty_like(lo_costs)
    torch_solver.cuda_solver.all_min_marginal_differences(cuda_mm_difference.data_ptr())

    torch_mm_lo, torch_mm_hi = torch_solver.marginals(lo_costs, hi_costs)
    torch_mm_diff = torch_mm_hi - torch_mm_lo
    assert torch.all(torch.abs(torch_mm_diff[valid_mask] - cuda_mm_difference[valid_mask]) < tolerance)

    smoothing = torch.tensor([1e-6], device = 'cuda')
    torch_mm_lo_smooth, torch_mm_hi_smooth = torch_solver.marginals(lo_costs, hi_costs, smoothing)
    assert torch.all(torch.abs(torch_mm_lo[valid_mask] - torch_mm_lo_smooth[valid_mask]) < tolerance)
    assert torch.all(torch.abs(torch_mm_hi[valid_mask] - torch_mm_hi_smooth[valid_mask]) < tolerance)

# def test_sum_marginals(instance, tolerance = 1e-4):
#     cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
#     # cuda_solver.non_learned_iterations(0.5, 1000, 1e-6, 3600)
#     torch_solver = bdd_torch_base(cuda_solver)
#     start = time.time()
#     lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
#     torch_sum_marginal_lo, torch_sum_marginal_hi = torch_solver.marginals(lo_costs, hi_costs, True)
#     print(torch_sum_marginal_lo)
#     print(torch_sum_marginal_hi)
#     print(f'torch sm time: {time.time() - start}')
#     cuda_sm_lo = torch.empty_like(torch_sum_marginal_lo)
#     cuda_sm_hi = torch.empty_like(torch_sum_marginal_lo)
#     start = time.time()
#     torch_solver.cuda_solver.sum_marginals(cuda_sm_lo.data_ptr(), cuda_sm_hi.data_ptr(), True)
#     print(f'cuda sm time: {time.time() - start}')
#     cuda_sm_diff = cuda_sm_hi - cuda_sm_lo
#     valid_mask = torch_solver.valid_layer_mask()
#     torch_sm_difference = torch_sum_marginal_hi - torch_sum_marginal_lo
#     assert torch.all(torch.abs(torch_sm_difference[valid_mask] - cuda_sm_diff[valid_mask]) < tolerance)

def test_smooth_solution(instance, tolerance = 1e-6):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_base(cuda_solver)
    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    smoothing = torch.tensor([1.0], device = 'cuda')
    torch_smooth_solution = torch_solver.smooth_solution(lo_costs, hi_costs, smoothing)
    cuda_smooth_solution = torch.empty_like(torch_smooth_solution)
    torch_solver.cuda_solver.smooth_solution_per_bdd(cuda_smooth_solution.data_ptr())
    valid_mask = torch_solver.valid_layer_mask()
    cuda_solution = torch.empty_like(torch_smooth_solution)
    torch_solver.cuda_solver.solution_per_bdd(cuda_solution.data_ptr())
    diff = torch.abs(torch_smooth_solution[valid_mask] - cuda_smooth_solution[valid_mask])
    print(f'max diff: {diff.max().item():.3f}')
    assert torch.all(diff < tolerance)

def compute_hamming_loss(torch_solver, lo_costs, hi_costs, target_sol, valid_mask):
    pred_sol = ComputePerBDDSolutions([torch_solver.cuda_solver], lo_costs, hi_costs)[valid_mask]
    # pred_sol = torch_solver.smooth_solution(lo_costs, hi_costs)[valid_mask]
    return torch.abs(pred_sol - target_sol).sum()

def convert_solution_to_bdd(torch_solver, sol):
    primal_index = torch_solver.get_variable_index()
    valid_layer_mask = torch_solver.valid_layer_mask()
    return sol[primal_index[valid_layer_mask]]

def test_smooth_solution_grad(instance, target_sol = None, tolerance = 1e-4):
    if torch.get_default_dtype() == torch.float64:
        cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma_double(instance, True, 1.0)
    else:
        cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_base(cuda_solver)
    start = time.time()
    lo_costs_orig, hi_costs_orig, def_mm = get_costs(torch_solver.cuda_solver)
    lo_costs = lo_costs_orig.clone()
    lo_costs.requires_grad = True
    hi_costs = hi_costs_orig.clone()
    hi_costs.requires_grad = True
    opt = torch.optim.Adam([lo_costs, hi_costs], 1e-2)
    if target_sol is not None:
        target_sol = target_sol.to('cuda').to(torch.get_default_dtype())
        target_sol = convert_solution_to_bdd(torch_solver, target_sol)
    valid_mask = torch_solver.valid_layer_mask()
    st = time.time()
    for itr in range(10):
        torch_smooth_solution = torch_solver.smooth_solution_logits(lo_costs, hi_costs)[valid_mask]
        # torch_smooth_solution = SmoothSolutionGradientCheckpoint.apply(torch_solver.cuda_solver, lo_costs, hi_costs, True)[valid_mask]
        if target_sol is None:
            target_sol = torch.ones_like(torch_smooth_solution)
        loss = torch.nn.CrossEntropyLoss()(torch_smooth_solution, target_sol)
        loss.backward()
        opt.step()
        with torch.no_grad():
            print(f'itr: {itr}, loss: {loss.item()}, hamming loss: {compute_hamming_loss(torch_solver, lo_costs, hi_costs, target_sol, valid_mask)}')
    print(f'\t test_smooth_solution_grad time: {time.time() - st}')

def test_mle(instance, target_sol = None):
    if torch.get_default_dtype() == torch.float64:
        cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma_double(instance, True, 1.0)
    else:
        cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    cuda_solver.non_learned_iterations(0.5, 1000, 1e-6, 60)
    torch_solver = bdd_torch_base(cuda_solver)
    start = time.time()
    lo_costs_orig, hi_costs_orig, def_mm = get_costs(torch_solver.cuda_solver)
    lo_costs = lo_costs_orig.clone()
    lo_costs.requires_grad = True
    hi_costs = hi_costs_orig.clone()
    hi_costs.requires_grad = True
    opt = torch.optim.Adam([lo_costs, hi_costs], 1e1)
    valid_mask = torch_solver.valid_layer_mask()
    if target_sol is not None:
        target_sol = target_sol.to('cuda').to(torch.get_default_dtype())
        target_sol = convert_solution_to_bdd(torch_solver, target_sol)
    else:
        target_sol = torch.ones_like(lo_costs[valid_mask])
    st = time.time()
    for itr in range(100):
        A = -torch_solver.compute_lower_bound(lo_costs, hi_costs, True)
        obj = -torch_solver.compute_solution_objective(lo_costs, hi_costs, target_sol)
        loss = A - obj
        # torch_smooth_solution = SmoothSolutionGradientCheckpoint.apply(torch_solver.cuda_solver, lo_costs, hi_costs, True)[valid_mask]
        loss.backward()
        opt.step()
        with torch.no_grad():
            print(f'itr: {itr}, loss: {loss.item()}, hamming loss: {compute_hamming_loss(torch_solver, lo_costs, hi_costs, target_sol, valid_mask)}')
    print(f'\t test_mle time: {time.time() - st}')

test_forward_run(ilp_instance_bbd.parse_ILP(one_simplex))
test_forward_run(ilp_instance_bbd.parse_ILP(two_simplex))
test_forward_run(ilp_instance_bbd.parse_ILP(matching_3x3))
test_forward_run(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_backward_run(ilp_instance_bbd.parse_ILP(one_simplex))
test_backward_run(ilp_instance_bbd.parse_ILP(two_simplex))
test_backward_run(ilp_instance_bbd.parse_ILP(matching_3x3))
test_backward_run(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_lower_bounds(ilp_instance_bbd.parse_ILP(one_simplex))
test_lower_bounds(ilp_instance_bbd.parse_ILP(two_simplex))
test_lower_bounds(ilp_instance_bbd.parse_ILP(matching_3x3))
test_lower_bounds(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_marginals(ilp_instance_bbd.parse_ILP(one_simplex))
test_marginals(ilp_instance_bbd.parse_ILP(two_simplex))
test_marginals(ilp_instance_bbd.parse_ILP(matching_3x3))
test_marginals(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

# test_sum_marginals(ilp_instance_bbd.parse_ILP(one_simplex))
# test_sum_marginals(ilp_instance_bbd.parse_ILP(two_simplex))
# test_sum_marginals(ilp_instance_bbd.parse_ILP(matching_3x3))
# test_sum_marginals(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_smooth_solution(ilp_instance_bbd.parse_ILP(one_simplex))
test_smooth_solution(ilp_instance_bbd.parse_ILP(two_simplex))
test_smooth_solution(ilp_instance_bbd.parse_ILP(matching_3x3))
test_smooth_solution(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_1000_n_cols_2000_density_0.05_max_coeff_200_seed_1/instances/100_0.051.lp')
test_lower_bounds(instance)

# test_smooth_solution_grad(ilp_instance_bbd.parse_ILP(one_simplex), [0, 1, 0])
# test_mle(ilp_instance_bbd.parse_ILP(one_simplex), [0, 1, 0])
# sm_instance = ilp_instance_bbd.read_ILP('/BS/discrete_opt/nobackup/qaplib_lp/small/lipa30b.lp')
# sm_instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/learnDBCA/airline_crew_scheduling_presolved_correct//train_split/instances/NW_320_week2.lp')

# sm_instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/average_sm.lp')
# test_forward_run(sm_instance)
# test_backward_run(sm_instance)
# test_min_marginals(sm_instance)
# test_sum_marginals(sm_instance)
# opt_sol = get_opt_solution('/home/ahabbas/data/learnDBCA/airline_crew_scheduling_presolved_correct//train_split/instances/NW_320_week2.lp', sm_instance)
# test_smooth_solution_grad(sm_instance, opt_sol)
# test_mle(sm_instance, opt_sol)
# for name in os.listdir('/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_1000_n_cols_2000_density_0.05_max_coeff_200_seed_1/instances/'):
#     if not name.endswith('.lp'):
#         continue
#     if "79_0.069.lp" not in name:
#         continue
#     print(f'Testing: {name}')
#     instance = ilp_instance_bbd.read_ILP(os.path.join('/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_1000_n_cols_2000_density_0.05_max_coeff_200_seed_1/instances/', name))
#     test_smooth_solution(instance)
#     print('')
