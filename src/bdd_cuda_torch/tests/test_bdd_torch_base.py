import torch
import time
import numpy as np
import bdd_cuda_torch.bdd_torch_base as bdd_torch_base
import BDD.ILP_instance_py as ilp_instance_bbd

one_simplex = """Minimize
1 x_1 + 2 x_2 + 1 x_3
Subject To
x_1 + x_2 + x_3 = 1
End"""

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

def test_forward_run(instance, tolerance = 1e-10):
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    lo_costs = torch_solver.lo_costs_.clone()
    hi_costs = torch_solver.hi_costs_.clone()
    torch_cost_from_root = torch_solver.forward_run(lo_costs, hi_costs, False)
    torch_cost_from_root_smooth = torch_solver.forward_run(lo_costs, hi_costs, True)
    print(f'torch forward time: {time.time() - start}')
    cuda_cost_from_root = torch.empty_like(torch_cost_from_root)
    start = time.time()
    torch_solver.cuda_solver.cost_from_root(cuda_cost_from_root.data_ptr())
    print(f'cuda forward time: {time.time() - start}')
    valid_mask = torch_solver.valid_bdd_node_mask()
    assert torch.all(torch.abs(torch_cost_from_root[valid_mask] - cuda_cost_from_root[valid_mask]) < tolerance)

def test_backward_run(instance, tolerance = 1e-10):
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    lo_costs = torch_solver.lo_costs_.clone()
    hi_costs = torch_solver.hi_costs_.clone()
    torch_cost_from_terminal = torch_solver.backward_run(lo_costs, hi_costs, False)
    torch_cost_from_terminal_smooth = torch_solver.backward_run(lo_costs, hi_costs, True)
    print(f'torch backward time: {time.time() - start}')
    cuda_cost_from_terminal = torch.empty_like(torch_cost_from_terminal)
    start = time.time()
    torch_solver.cuda_solver.cost_from_terminal(cuda_cost_from_terminal.data_ptr())
    print(f'cuda backward time: {time.time() - start}')
    valid_mask = torch_solver.valid_bdd_node_mask()
    assert torch.all(torch.abs(torch_cost_from_terminal[valid_mask] - cuda_cost_from_terminal[valid_mask]) < tolerance)

def test_min_marginals(instance, tolerance = 1e-10):
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    lo_costs = torch_solver.lo_costs_.clone()
    hi_costs = torch_solver.hi_costs_.clone()
    torch_min_marginal_lo, torch_min_marginal_hi = torch_solver.marginals(lo_costs, hi_costs, False)
    print(f'torch mm time: {time.time() - start}')
    cuda_mm_difference = torch.empty_like(torch_min_marginal_lo)
    start = time.time()
    torch_solver.cuda_solver.all_min_marginal_differences(cuda_mm_difference.data_ptr())
    print(f'cuda mm time: {time.time() - start}')
    valid_mask = torch_solver.valid_layer_mask()
    torch_mm_difference = torch_min_marginal_hi - torch_min_marginal_lo
    assert torch.all(torch.abs(torch_mm_difference[valid_mask] - cuda_mm_difference[valid_mask]) < tolerance)

def test_sum_marginals(instance, tolerance = 1e-4):
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    lo_costs = torch_solver.lo_costs_.clone()
    hi_costs = torch_solver.hi_costs_.clone()
    torch_sum_marginal_lo, torch_sum_marginal_hi = torch_solver.marginals(lo_costs, hi_costs, True)
    print(f'torch sm time: {time.time() - start}')
    cuda_sm_lo = torch.empty_like(torch_sum_marginal_lo)
    cuda_sm_hi = torch.empty_like(torch_sum_marginal_lo)
    start = time.time()
    torch_solver.cuda_solver.sum_marginals(cuda_sm_lo.data_ptr(), cuda_sm_hi.data_ptr(), True)
    print(f'cuda sm time: {time.time() - start}')
    cuda_sm_diff = cuda_sm_hi - cuda_sm_lo
    valid_mask = torch_solver.valid_layer_mask()
    torch_sm_difference = torch_sum_marginal_hi - torch_sum_marginal_lo
    assert torch.all(torch.abs(torch_sm_difference[valid_mask] - cuda_sm_diff[valid_mask]) < tolerance)

def test_smooth_solution(instance, tolerance = 1e-4):
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    lo_costs = torch_solver.lo_costs_.clone()
    hi_costs = torch_solver.hi_costs_.clone()
    torch_smooth_solution = torch_solver.smooth_solution(lo_costs, hi_costs)
    print(f'torch smooth solution time: {time.time() - start}')
    cuda_smooth_solution = torch.empty_like(torch_smooth_solution)
    start = time.time()
    torch_solver.cuda_solver.smooth_solution_per_bdd(cuda_smooth_solution.data_ptr())
    print(f'cuda smooth solution time: {time.time() - start}')
    valid_mask = torch_solver.valid_layer_mask()
    cuda_solution = torch.empty_like(torch_smooth_solution)
    start = time.time()
    torch_solver.cuda_solver.solution_per_bdd(cuda_solution.data_ptr())
    assert torch.all(torch.abs(torch_smooth_solution[valid_mask] - cuda_smooth_solution[valid_mask]) < tolerance)

def test_smooth_solution_grad(instance, target_sol, tolerance = 1e-4):
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    lo_costs = torch_solver.lo_costs_.clone()
    lo_costs.requires_grad = True
    hi_costs = torch_solver.hi_costs_.clone()
    hi_costs.requires_grad = True
    opt = torch.optim.Adam([lo_costs, hi_costs], 1e-2)
    target_sol = torch.tensor(target_sol).to('cuda').to(torch.float32)
    valid_mask = torch_solver.valid_layer_mask()
    for itr in range(100):
        torch_smooth_solution = torch_solver.smooth_solution(lo_costs, hi_costs, True)[valid_mask]
        loss = torch.nn.CrossEntropyLoss()(torch_smooth_solution, target_sol)
        loss.backward()
        opt.step()
        with torch.no_grad():
            print(f'itr: {itr}, loss: {loss.item()}, smooth solution: {torch_solver.smooth_solution(lo_costs, hi_costs)[valid_mask]}, net_costs: {hi_costs.detach() - lo_costs.detach()}')

test_forward_run(ilp_instance_bbd.parse_ILP(one_simplex))
test_forward_run(ilp_instance_bbd.parse_ILP(two_simplex))
test_forward_run(ilp_instance_bbd.parse_ILP(matching_3x3))
test_forward_run(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_backward_run(ilp_instance_bbd.parse_ILP(one_simplex))
test_backward_run(ilp_instance_bbd.parse_ILP(two_simplex))
test_backward_run(ilp_instance_bbd.parse_ILP(matching_3x3))
test_backward_run(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_min_marginals(ilp_instance_bbd.parse_ILP(one_simplex))
test_min_marginals(ilp_instance_bbd.parse_ILP(two_simplex))
test_min_marginals(ilp_instance_bbd.parse_ILP(matching_3x3))
test_min_marginals(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_sum_marginals(ilp_instance_bbd.parse_ILP(one_simplex))
test_sum_marginals(ilp_instance_bbd.parse_ILP(two_simplex))
test_sum_marginals(ilp_instance_bbd.parse_ILP(matching_3x3))
test_sum_marginals(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_smooth_solution(ilp_instance_bbd.parse_ILP(one_simplex))
test_smooth_solution(ilp_instance_bbd.parse_ILP(two_simplex))
test_smooth_solution(ilp_instance_bbd.parse_ILP(matching_3x3))
test_smooth_solution(ilp_instance_bbd.parse_ILP(short_chain_shuffled))

test_smooth_solution_grad(ilp_instance_bbd.parse_ILP(one_simplex), [0, 1, 0])
# sm_instance = ilp_instance_bbd.read_ILP('/BS/discrete_opt/nobackup/qaplib_lp/small/lipa30b.lp')
# test_forward_run(sm_instance)
# test_backward_run(sm_instance)
# test_min_marginals(sm_instance)
# test_sum_marginals(sm_instance)
# test_sum_marginals_grad(sm_instance)
