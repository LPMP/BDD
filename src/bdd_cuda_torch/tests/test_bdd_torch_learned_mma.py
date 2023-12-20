import torch
torch.autograd.set_detect_anomaly(True) 
# torch.set_default_dtype(torch.float64)
import time, os
import numpy as np
import bdd_cuda_torch.bdd_torch_learned_mma as bdd_torch_learned_mma
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_cuda_solver
import bdd_cuda_torch
from torch_scatter import scatter_softmax, scatter_mean
import pickle

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

grid_graph_3x3 = """Minimize
2 mu_0_0 - 1 mu_0_1 + 3 mu_1_0 - 1 mu_1_1
+ 3 mu_2_0 + 2 mu_2_1 - 1 mu_3_0 - 2 mu_3_1
- 2 mu_4_0 - 1 mu_4_1 + 3 mu_5_0 - 1 mu_5_1
+ 1 mu_6_0 + 1 mu_6_1 - 3 mu_7_0 + 2 mu_7_1
+ 0 mu_8_0 + 2 mu_8_1
+ 1 mu_01_00 - 2 mu_01_01 + 2 mu_01_10 - 1 mu_01_11
+ 0 mu_12_00 + 1 mu_12_01 + 1 mu_12_10 + 0 mu_12_11
- 1 mu_03_00 + 2 mu_03_01 + 0 mu_03_10 - 2 mu_03_11
+ 2 mu_14_00 + 0 mu_14_01 + 2 mu_14_10 + 2 mu_14_11
+ 1 mu_25_00 - 2 mu_25_01 - 3 mu_25_10 - 1 mu_25_11
+ 0 mu_34_00 + 1 mu_34_01 + 1 mu_34_10 + 1 mu_34_11
- 1 mu_45_00 - 2 mu_45_01 + 4 mu_45_10 - 2 mu_45_11
- 2 mu_36_00 + 0 mu_36_01 + 1 mu_36_10 + 3 mu_36_11
+ 3 mu_47_00 - 2 mu_47_01 - 2 mu_47_10 - 1 mu_47_11
+ 0 mu_58_00 + 1 mu_58_01 + 1 mu_58_10 + 1 mu_58_11
- 1 mu_67_00 + 2 mu_67_01 - 1 mu_67_10 - 1 mu_67_11
+ 2 mu_78_00 + 0 mu_78_01 + 2 mu_78_10 + 2 mu_78_11
Subject To
mu_0_0 + mu_0_1 = 1
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_3_0 + mu_3_1 = 1
mu_4_0 + mu_4_1 = 1
mu_5_0 + mu_5_1 = 1
mu_6_0 + mu_6_1 = 1
mu_7_0 + mu_7_1 = 1
mu_8_0 + mu_8_1 = 1
mu_01_00 + mu_01_10 + mu_01_01 + mu_01_11 = 1
mu_12_00 + mu_12_10 + mu_12_01 + mu_12_11 = 1
mu_03_00 + mu_03_10 + mu_03_01 + mu_03_11 = 1
mu_14_00 + mu_14_10 + mu_14_01 + mu_14_11 = 1
mu_25_00 + mu_25_10 + mu_25_01 + mu_25_11 = 1
mu_34_00 + mu_34_10 + mu_34_01 + mu_34_11 = 1
mu_45_00 + mu_45_10 + mu_45_01 + mu_45_11 = 1
mu_36_00 + mu_36_10 + mu_36_01 + mu_36_11 = 1
mu_47_00 + mu_47_10 + mu_47_01 + mu_47_11 = 1
mu_58_00 + mu_58_10 + mu_58_01 + mu_58_11 = 1
mu_67_00 + mu_67_10 + mu_67_01 + mu_67_11 = 1
mu_78_00 + mu_78_10 + mu_78_01 + mu_78_11 = 1
mu_0_0 - mu_01_00 - mu_01_01 = 0
mu_0_1 - mu_01_10 - mu_01_11 = 0
mu_0_0 - mu_03_00 - mu_03_01 = 0
mu_0_1 - mu_03_10 - mu_03_11 = 0
mu_1_0 - mu_01_00 - mu_01_10 = 0
mu_1_1 - mu_01_01 - mu_01_11 = 0
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_1_0 - mu_14_00 - mu_14_01 = 0
mu_1_1 - mu_14_10 - mu_14_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_2_0 - mu_25_00 - mu_25_01 = 0
mu_2_1 - mu_25_10 - mu_25_11 = 0
mu_3_0 - mu_03_00 - mu_03_10 = 0
mu_3_1 - mu_03_01 - mu_03_11 = 0
mu_3_0 - mu_34_00 - mu_34_01 = 0
mu_3_1 - mu_34_10 - mu_34_11 = 0
mu_3_0 - mu_36_00 - mu_36_01 = 0
mu_3_1 - mu_36_10 - mu_36_11 = 0
mu_4_0 - mu_14_00 - mu_14_10 = 0
mu_4_1 - mu_14_01 - mu_14_11 = 0
mu_4_0 - mu_34_00 - mu_34_10 = 0
mu_4_1 - mu_34_01 - mu_34_11 = 0
mu_4_0 - mu_45_00 - mu_45_01 = 0
mu_4_1 - mu_45_10 - mu_45_11 = 0
mu_4_0 - mu_47_00 - mu_47_01 = 0
mu_4_1 - mu_47_10 - mu_47_11 = 0
mu_5_0 - mu_25_00 - mu_25_10 = 0
mu_5_1 - mu_25_01 - mu_25_11 = 0
mu_5_0 - mu_45_00 - mu_45_10 = 0
mu_5_1 - mu_45_01 - mu_45_11 = 0
mu_5_0 - mu_58_00 - mu_58_01 = 0
mu_5_1 - mu_58_10 - mu_58_11 = 0
mu_6_0 - mu_36_00 - mu_36_10 = 0
mu_6_1 - mu_36_01 - mu_36_11 = 0
mu_6_0 - mu_67_00 - mu_67_01 = 0
mu_6_1 - mu_67_10 - mu_67_11 = 0
mu_7_0 - mu_47_00 - mu_47_10 = 0
mu_7_1 - mu_47_01 - mu_47_11 = 0
mu_7_0 - mu_67_00 - mu_67_10 = 0
mu_7_1 - mu_67_01 - mu_67_11 = 0
mu_7_0 - mu_78_00 - mu_78_01 = 0
mu_7_1 - mu_78_10 - mu_78_11 = 0
mu_8_0 - mu_58_00 - mu_58_10 = 0
mu_8_1 - mu_58_01 - mu_58_11 = 0
mu_8_0 - mu_78_00 - mu_78_10 = 0
mu_8_1 - mu_78_01 - mu_78_11 = 0
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

def normalize_distribution_weights_softmax(dist_weights, var_indices):
    softmax_scores = scatter_softmax(dist_weights.to(torch.get_default_dtype()), var_indices)
    return softmax_scores.to(torch.get_default_dtype())

def normalize_smoothing(smoothing, bdd_indices):
    return scatter_mean(smoothing, bdd_indices)

def cuda_lower_bound(cuda_solver, num_iterations):
    cuda_solver.non_learned_iterations(0.5, num_iterations, 0.0, 100)
    return cuda_solver.lower_bound()

def test_iterations(instance, num_iterations = 50, tolerance = 1e-6):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    torch_solver = bdd_torch_learned_mma(cuda_solver)
    start = time.time()
    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    
    alpha = torch_solver.get_isotropic_alpha()
    omega = torch.zeros_like(alpha) + 0.5

    st = time.time()
    cuda_lb = cuda_lower_bound(cuda_solver, num_iterations)
    cuda_time = time.time() - st

    st = time.time()
    # lo_costs, hi_costs, def_mm, _, _, _, _ = bdd_cuda_torch.DualIterations.apply([cuda_solver], lo_costs, hi_costs, def_mm, alpha, 20, torch.tensor([0.5], device = 'cuda'), 5, 0.0, 5, 0, 0.9, False, 0)
    lo_costs, hi_costs, def_mm = torch_solver.iterations(lo_costs, hi_costs, def_mm, alpha, omega, num_iterations)
    torch_lb = torch_solver.compute_lower_bound(lo_costs, hi_costs, None, def_mm).item()
    torch_time = time.time() - st
    print(f'torch LB: {torch_lb}, cuda LB: {cuda_lb}')
    print(f'CUDA time: {cuda_time}')
    print(f'torch time: {torch_time}')
    assert np.abs(torch_lb - cuda_lb) < tolerance

def test_smooth_iterations(instance, starting_itr = 50, smooth_num_iterations = 50, smoothing = 1e-4):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    print(f'Starting LB: {cuda_lower_bound(cuda_solver, starting_itr)}')

    torch_solver = bdd_torch_learned_mma(cuda_solver)

    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    lo_costs, hi_costs = bdd_cuda_torch.DistributeDeferredDelta.apply([torch_solver.cuda_solver], lo_costs, hi_costs, def_mm)
    def_mm = def_mm * 0

    primal_variable_index = torch_solver.get_variable_index()
    alpha_feas = torch_solver.get_isotropic_alpha()
    omega_feas = torch.ones_like(alpha_feas) - 0.5
    smoothing_feas = torch.tensor([smoothing], device = 'cuda')
    for outer_itr in range(smooth_num_iterations // 5):
        st = time.time()
        lo_costs, hi_costs, def_mm = torch_solver.iterations(lo_costs, hi_costs, def_mm, alpha_feas, omega_feas, 5, smoothing = smoothing_feas)
        print(f'without grad time: {time.time() - st}, num_iterations: {(outer_itr + 1) * 5}')
        torch_lb = torch_solver.compute_lower_bound(lo_costs, hi_costs)
        print(f'LB: {torch_lb.item()}')

def test_grad_iterations(instance, starting_itr = 1, num_iterations = 3, tolerance = 1e-6):
    cuda_solver = bdd_cuda_solver.bdd_cuda_learned_mma(instance, True, 1.0)
    print(f'Starting LB: {cuda_lower_bound(cuda_solver, starting_itr)}')

    torch_solver = bdd_torch_learned_mma(cuda_solver)

    lo_costs, hi_costs, def_mm = get_costs(torch_solver.cuda_solver)
    lo_costs, hi_costs = bdd_cuda_torch.DistributeDeferredDelta.apply([torch_solver.cuda_solver], lo_costs, hi_costs, def_mm)
    def_mm = def_mm * 0

    primal_variable_index = torch_solver.get_variable_index()
    bdd_index = torch_solver.get_bdd_index()
    alpha = torch_solver.get_isotropic_alpha()
    omega = torch.zeros_like(alpha)
    smoothing_per_bdd = torch.ones(cuda_solver.nr_bdds(), device = 'cuda') * 1e-2
    # smoothing_per_bdd = torch.ones(1, device = 'cuda') * 1e-2
    # alpha.requires_grad = True
    # omega.requires_grad = True
    smoothing_per_bdd.requires_grad = True
    # opt = torch.optim.Adam([{'params': alpha}, {'params': omega}], lr=1.0)
    opt = torch.optim.Adam([{'params': smoothing_per_bdd, 'lr': 1e-3}], lr=1.0)
    for itr in range(50):
        omega_feas = torch.sigmoid(omega)
        alpha_feas = normalize_distribution_weights_softmax(alpha, primal_variable_index)
        smoothing_feas = torch.abs(smoothing_per_bdd) #+ 1e-6
        st = time.time()
        lo_costs_new, hi_costs_new, def_mm_new = torch_solver.iterations(lo_costs, hi_costs, def_mm, alpha_feas, omega_feas, num_iterations, smoothing = smoothing_feas)
        print(f'with grad time: {time.time() - st}, num_iterations: {num_iterations}')
        # lo_costs_new, hi_costs_new, def_mm_new, _ = torch_solver.iterations(lo_costs, hi_costs, def_mm, alpha_feas, omega_feas, num_iterations, smoothing = smoothing_feas)
        # lo_costs_new, hi_costs_new, def_mm_new, _, _, _, _ = bdd_cuda_torch.DualIterations.apply([torch_solver.cuda_solver], lo_costs, hi_costs, def_mm, alpha_feas, num_iterations, omega_feas, num_iterations, 0.0, 5, 0, 0.9, False, 0)
        torch_lb = torch_solver.compute_lower_bound(lo_costs_new, hi_costs_new)
        loss = -torch_lb
        print(f'smoothing_feas: min: {smoothing_feas.min().item()}, max: {smoothing_feas.max().item()}, mean: {smoothing_feas.mean().item()}')
        loss.backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            print(f'itr: {itr}, LB: {torch_lb.item()}')

    with torch.no_grad():
        omega_feas = torch.sigmoid(omega)
        alpha_feas = normalize_distribution_weights_softmax(alpha, primal_variable_index)
        num_iterations_f = 100
        # lo_costs_new, hi_costs_new, def_mm_new, cost_from_terminal_new = torch_solver.iterations(lo_costs, hi_costs, def_mm, alpha_feas, omega_feas, cost_from_terminal, num_iterations_f)
        lo_costs_new, hi_costs_new, def_mm_new, _, _, _, _ = bdd_cuda_torch.DualIterations.apply([torch_solver.cuda_solver], lo_costs, hi_costs, def_mm, alpha_feas, num_iterations_f, omega_feas, num_iterations_f, 0.0, 5, 0, 0.9, False, 0)
        torch_lb = torch_solver.compute_lower_bound(lo_costs_new, hi_costs_new, False, def_mm_new)
        print(f'Final LB: {torch_lb.item()}')
    alpha_init = normalize_distribution_weights_softmax(alpha, primal_variable_index)
    diff_alpha = alpha_feas - alpha_init
    diff_omega = omega_feas - torch.sigmoid(torch.zeros_like(omega))
    breakpoint()

# test_iterations(ilp_instance_bbd.parse_ILP(one_simplex))
# test_iterations(ilp_instance_bbd.parse_ILP(two_simplex))
# test_iterations(ilp_instance_bbd.parse_ILP(matching_3x3))
# test_iterations(ilp_instance_bbd.parse_ILP(short_chain_shuffled))
# test_grad_iterations(ilp_instance_bbd.parse_ILP(matching_3x3))
# test_grad_iterations(ilp_instance_bbd.parse_ILP(short_chain_shuffled), 1)
# test_grad_iterations(ilp_instance_bbd.parse_ILP(grid_graph_3x3), 1)

# sm_instance = ilp_instance_bbd.read_ILP('/BS/discrete_opt/nobackup/qaplib_lp/small/lipa30b.lp')
# sm_instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/instances/flywing_100_1.lp')
# test_iterations(sm_instance)

# sm_instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_1000_n_cols_2000_density_0.05_max_coeff_200_seed_1/instances/0_0.047.lp')
sm_instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_1000_n_cols_2000_density_0.05_max_coeff_200_seed_1/instances/100_0.051.lp')
# sm_instance = ilp_instance_bbd.read_ILP('/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_10000_n_cols_20000_density_0.05_max_coeff_200_seed_1/instances/0_0.044.lp')
# bdd_repr_path = '/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_1000_n_cols_2000_density_0.05_max_coeff_200_seed_1/instances/0_0.047_bdd_repr.pkl'
# bdd_repr_path = '/home/ahabbas/data/learnDBCA/set_cover_random_doge/SetCover_n_rows_10000_n_cols_20000_density_0.05_max_coeff_200_seed_1/instances/0_0.044_bdd_repr.pkl'
# bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
# solver = pickle.loads(bdd_repr['solver_data'])
# test_smooth_iterations(sm_instance, 1000, 100, 1e-4)
test_grad_iterations(sm_instance, 100)
# test_grad_iterations(solver, 1)
