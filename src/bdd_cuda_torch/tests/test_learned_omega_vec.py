import torch
import BDD.bdd_cuda_learned_mma_py
import BDD.ILP_instance_py
from torch_scatter import scatter_sum
from bdd_cuda_torch import DualIterations, DistributeDeferredDelta, ComputeAllMinMarginalsDiff, PerturbPrimalCosts

two_simplex = """Minimize
1 x_1 + 2 x_2 + 1 x_3
+2 x_4 + 1 x_5 + 2 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + x_5 + x_6 = 2
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

long_chain = """Minimize
2 mu_0_0 - 1 mu_0_1 + 3 mu_1_0 - 1 mu_1_1
+ 3 mu_2_0 + 2 mu_2_1 - 1 mu_3_0 - 2 mu_3_1
- 2 mu_4_0 - 1 mu_4_1 + 1 mu_5_0 - 1 mu_5_1
+ 1 mu_6_0 + 1 mu_6_1 - 3 mu_7_0 + 2 mu_7_1
+ 0 mu_8_0 + 2 mu_8_1
+ 1 mu_01_00 - 2 mu_01_01 + 2 mu_01_10 - 1 mu_01_11
+ 0 mu_12_00 - 1 mu_12_01 + 1 mu_12_10 + 0 mu_12_11
- 1 mu_23_00 + 2 mu_23_01 + 1 mu_23_10 - 2 mu_23_11
+ 2 mu_34_00 + 0 mu_34_01 + 2 mu_34_10 + 2 mu_34_11
+ 1 mu_45_00 - 2 mu_45_01 - 3 mu_45_10 - 1 mu_45_11
- 2 mu_56_00 + 0 mu_56_01 + 1 mu_56_10 + 3 mu_56_11
- 1 mu_67_00 - 2 mu_67_01 - 1 mu_67_10 - 1 mu_67_11
+ 2 mu_78_00 + 0 mu_78_01 + 2 mu_78_10 + 3 mu_78_11
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
mu_23_00 + mu_23_10 + mu_23_01 + mu_23_11 = 1
mu_34_00 + mu_34_10 + mu_34_01 + mu_34_11 = 1
mu_45_00 + mu_45_10 + mu_45_01 + mu_45_11 = 1
mu_56_00 + mu_56_10 + mu_56_01 + mu_56_11 = 1
mu_67_00 + mu_67_10 + mu_67_01 + mu_67_11 = 1
mu_78_00 + mu_78_10 + mu_78_01 + mu_78_11 = 1
mu_0_0 - mu_01_00 - mu_01_01 = 0
mu_0_1 - mu_01_10 - mu_01_11 = 0
mu_1_0 - mu_01_00 - mu_01_10 = 0
mu_1_1 - mu_01_01 - mu_01_11 = 0
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_2_0 - mu_23_00 - mu_23_01 = 0
mu_2_1 - mu_23_10 - mu_23_11 = 0
mu_3_0 - mu_23_00 - mu_23_10 = 0
mu_3_1 - mu_23_01 - mu_23_11 = 0
mu_3_0 - mu_34_00 - mu_34_01 = 0
mu_3_1 - mu_34_10 - mu_34_11 = 0
mu_4_0 - mu_34_00 - mu_34_10 = 0
mu_4_1 - mu_34_01 - mu_34_11 = 0
mu_4_0 - mu_45_00 - mu_45_01 = 0
mu_4_1 - mu_45_10 - mu_45_11 = 0
mu_5_0 - mu_45_00 - mu_45_10 = 0
mu_5_1 - mu_45_01 - mu_45_11 = 0
mu_5_0 - mu_56_00 - mu_56_01 = 0
mu_5_1 - mu_56_10 - mu_56_11 = 0
mu_6_0 - mu_56_00 - mu_56_10 = 0
mu_6_1 - mu_56_01 - mu_56_11 = 0
mu_6_0 - mu_67_00 - mu_67_01 = 0
mu_6_1 - mu_67_10 - mu_67_11 = 0
mu_7_0 - mu_67_00 - mu_67_10 = 0
mu_7_1 - mu_67_01 - mu_67_11 = 0
mu_7_0 - mu_78_00 - mu_78_01 = 0
mu_7_1 - mu_78_10 - mu_78_11 = 0
mu_8_0 - mu_78_00 - mu_78_10 = 0
mu_8_1 - mu_78_01 - mu_78_11 = 0
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

def project_dist_weights(solver, dist_weights):
    primal_indices = torch.empty_like(dist_weights, dtype = torch.int32)
    solver.primal_variable_index(primal_indices.data_ptr())
    primal_indices = primal_indices.to(torch.int64)
    dist_weights_sum = scatter_sum(dist_weights, primal_indices)[primal_indices]
    return dist_weights / dist_weights_sum

def compute_expected_results(instance_string, device):
    instance = BDD.ILP_instance_py.parse_ILP(instance_string)
    solver = BDD.bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(instance)
    solver.non_learned_iterations(0.5, 200, 1e-6)
    expected_mm_diff = torch.empty(solver.nr_layers(), device = device, dtype = torch.float32)
    solver.all_min_marginal_differences(expected_mm_diff.data_ptr())
    return expected_mm_diff, solver.lower_bound()

def run_instance(instance_string, num_solver_itr, num_learning_itr, omega_scalar, device):
    expected_mm_diff, expected_lb = compute_expected_results(instance_string, device)
    instance = BDD.ILP_instance_py.parse_ILP(instance_string)
    solver = BDD.bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(instance)
    dist_weights = torch.ones(solver.nr_layers(), device = device, dtype = torch.float32)
    dist_weights = project_dist_weights(solver, dist_weights)
    orig_lo_costs = torch.empty_like(dist_weights)
    orig_hi_costs = torch.empty_like(dist_weights)
    orig_def_mm = torch.empty_like(dist_weights)
    solver.get_solver_costs(orig_lo_costs.data_ptr(), orig_hi_costs.data_ptr(), orig_def_mm.data_ptr())
    avg_loss_improvement_per_itr = 0
    omega_vec = torch.ones_like(dist_weights) * omega_scalar
    for i in range(num_learning_itr):
        omega_vec_g = omega_vec.detach().clone()
        omega_vec_g.requires_grad = True
        lo_costs_out, hi_costs_out, def_mm_out = DualIterations.apply([solver], orig_lo_costs, orig_hi_costs, orig_def_mm, dist_weights, num_solver_itr, omega_vec_g, num_solver_itr, 1e-6)
        lo_costs_out, hi_costs_out = DistributeDeferredDelta.apply([solver], lo_costs_out, hi_costs_out, def_mm_out)
        mm_diff = ComputeAllMinMarginalsDiff.apply([solver], lo_costs_out, hi_costs_out)
        loss = torch.abs(expected_mm_diff - mm_diff).sum()
        if i > 0:
            avg_loss_improvement_per_itr += (prev_loss - loss.item())
        prev_loss = loss.item()
        print(f"Grad itr: {i}, Loss: {loss.item():.4f}, LB: {solver.lower_bound():.4f}, Max. possible LB: {expected_lb:.4f}, omega_vec mean: {torch.mean(omega_vec).item():.4f}, omega var: {torch.std(omega_vec):.4f}")
        loss.backward()
        omega_vec = omega_vec - 1e-1 * omega_vec_g.grad
        omega_vec = torch.clamp(omega_vec, min = 0.0, max = 1.0)
    assert(avg_loss_improvement_per_itr > 0)

device = torch.device("cuda:0") 

omega_initial = 0.1
num_dual_itr = 250
num_learning_itr = 10
run_instance(two_simplex, num_learning_itr, num_dual_itr, omega_initial, device)
run_instance(matching_3x3, num_learning_itr, num_dual_itr, omega_initial, device)
run_instance(short_chain_shuffled, num_learning_itr, num_dual_itr, omega_initial, device)
run_instance(long_chain, num_learning_itr, num_dual_itr, omega_initial, device)
run_instance(grid_graph_3x3, num_learning_itr, num_dual_itr, omega_initial, device)
