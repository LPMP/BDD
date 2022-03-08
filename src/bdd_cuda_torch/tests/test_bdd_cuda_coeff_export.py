from cmath import inf
import torch
import BDD.bdd_cuda_learned_mma_py
import BDD.ILP_instance_py

instance_lp = """Minimize
1 x_1 + 2 x_2 + 1 x_3
+2 x_4 + 1 x_5 + 2 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + 2 x_5 - x_6 = 2
End"""

expected_coeffs = [[1], [1], [1], [1, 1], [inf, 2], [inf, -1], [0, 0]] # Last two zeros for terminal nodes.
instance = BDD.ILP_instance_py.parse_ILP(instance_lp)
solver = BDD.bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(instance)

coeffs = solver.constraint_matrix_coefficients(instance)
bdd_to_constraint_map = solver.bdd_to_constraint_map()

primal_indices = torch.empty(solver.nr_layers(), dtype = torch.int32, device = 'cuda')
solver.primal_variable_index(primal_indices.data_ptr())

bdd_indices = torch.empty(solver.nr_layers(), dtype = torch.int32, device = 'cuda')
solver.bdd_index(bdd_indices.data_ptr())

num_vars = solver.nr_primal_variables()

for l in range(solver.nr_layers()):
    v = primal_indices[l]
    c = bdd_indices[l]
    coeff = coeffs[l]
    expected_coeff = expected_coeffs[v][c]
    assert expected_coeff == coeff, f'Mismatch for layer: {l}, v: {v}, c: {c}, coeff: {coeff}, expected coeff: {expected_coeff}'