import BDD.bdd_cuda_learned_mma_py as bdd_cuda_learned_mma_py
import BDD.ILP_instance_py as ILP_instance_py
import pickle

def test_pickle(solver):
    prev_lb = solver.lower_bound()
    print(solver)
    solver_data = pickle.dumps(solver, -1)
    solver_from_pickle = pickle.loads(solver_data)
    print(solver_from_pickle)
    assert(prev_lb == solver_from_pickle.lower_bound())

one_simplex_problem = '''Minimize
-2 x_11 - 1 x_12 - 1 x_13
Subject To
x_11 + x_12 + x_13 = 1
End''' 

two_simplex_problem = '''Minimize
2 x_1 + 1 x_2 + 1 x_3
+1 x_4 + 2 x_5 - 1 x_6
Subject To
x_1 + x_2 + x_3 = 1
x_4 + x_5 + x_6 = 2
End''' 

two_simplex_diff_size_problem = '''Minimize
2 x_1 + 1 x_2 + 1 x_3
+2 x_4 + 2 x_5 + 3 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + x_5 + x_6 = 2
End''' 

matching_3x3 = '''Minimize
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
End''' 

ilp_instance = ILP_instance_py.parse_ILP(one_simplex_problem)
bdd_cuda_solver = bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(ilp_instance)
assert(bdd_cuda_solver.nr_primal_variables() == 3)
assert(bdd_cuda_solver.nr_bdds() == 1)
assert(bdd_cuda_solver.lower_bound() == -2)
test_pickle(bdd_cuda_solver)

ilp_instance = ILP_instance_py.parse_ILP(two_simplex_problem)
bdd_cuda_solver = bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(ilp_instance)
assert(bdd_cuda_solver.nr_primal_variables() == 6)
assert(bdd_cuda_solver.nr_bdds() == 2)
assert(bdd_cuda_solver.lower_bound() == 1)
test_pickle(bdd_cuda_solver)

ilp_instance = ILP_instance_py.parse_ILP(two_simplex_diff_size_problem)
bdd_cuda_solver = bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(ilp_instance)
assert(bdd_cuda_solver.nr_primal_variables() == 6)
assert(bdd_cuda_solver.nr_bdds() == 2)
assert(bdd_cuda_solver.lower_bound() == 4)
test_pickle(bdd_cuda_solver)

ilp_instance = ILP_instance_py.parse_ILP(matching_3x3)
bdd_cuda_solver = bdd_cuda_learned_mma_py.bdd_cuda_learned_mma(ilp_instance)
assert(bdd_cuda_solver.nr_primal_variables() == 9)
assert(bdd_cuda_solver.nr_bdds() == 6)
assert(bdd_cuda_solver.lower_bound() == -6)
test_pickle(bdd_cuda_solver)