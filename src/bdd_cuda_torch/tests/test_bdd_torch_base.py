import torch
import time
import bdd_cuda_torch.bdd_torch_base as bdd_torch_base
import BDD.ILP_instance_py as ilp_instance_bbd

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

def test_forward_run(instance_string, tolerance = 1e-10):
    instance = ilp_instance_bbd.parse_ILP(instance_string)
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    torch_cost_from_root = torch_solver.forward_run()
    print(f'torch forward time: {time.time() - start}')
    cuda_cost_from_root = torch.empty_like(torch_cost_from_root)
    start = time.time()
    torch_solver.cuda_solver.cost_from_root(cuda_cost_from_root.data_ptr())
    print(f'cuda forward time: {time.time() - start}')
    assert torch.all(torch.abs(torch_cost_from_root - cuda_cost_from_root) < tolerance)

def test_backward_run(instance_string, tolerance = 1e-10):
    instance = ilp_instance_bbd.parse_ILP(instance_string)
    torch_solver = bdd_torch_base(instance)
    start = time.time()
    torch_cost_from_terminal = torch_solver.backward_run()
    print(f'torch backward time: {time.time() - start}')
    cuda_cost_from_terminal = torch.empty_like(torch_cost_from_terminal)
    start = time.time()
    torch_solver.cuda_solver.cost_from_terminal(cuda_cost_from_terminal.data_ptr())
    print(f'cuda backward time: {time.time() - start}')
    valid_mask = torch.isfinite(cuda_cost_from_terminal)
    assert torch.all(torch.abs(torch_cost_from_terminal[valid_mask] - cuda_cost_from_terminal[valid_mask]) < tolerance)

test_forward_run(two_simplex)
test_forward_run(matching_3x3)
test_forward_run(short_chain_shuffled)

test_backward_run(two_simplex)
test_backward_run(matching_3x3)
test_backward_run(short_chain_shuffled)