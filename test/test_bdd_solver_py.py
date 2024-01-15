import BDD.ILP_instance_py as ilp_instance
from BDD.bdd_solver_py import bdd_solver as bdd_solver
from BDD.bdd_solver_py import bdd_solver_options as bdd_solver_options

def create_toy_problem():
    objective = {'x1': 2.0, 'x2': 1.0, 'x3': -1.0, 'x4': 1.0, 'x5': -2.0, 'x6': -5.0}
    constraints = [
        ['C1', {'x1': 1.0, 'x2': 1.0, 'x3': 1.0}, '=', 1.0],
        ['C2', {'x3': 1.0, 'x4': 1.0, 'x5': 1.0}, '<', 1.0],
        ['C3', {'x4': 1.0, 'x5': 1.0, 'x6': 1.0}, '>', 1.0]
    ]
    ilp = ilp_instance.ILP_instance()
    for var_name, coeff in objective.items():
        ilp.add_new_variable_with_obj(var_name, coeff)

    for con in constraints:
        con_name = con[0]
        con_vars = list(con[1].keys())
        con_coeffs = list(con[1].values())
        con_coeffs = [int(c) for c in con_coeffs] # Only integer coefficients are supported.
        con_sense = con[2]
        rhs_value = int(con[3]) # Only integer right hand side value is supported.
        if con_sense == '=':
            ineq_type = ilp_instance.equal
        elif con_sense == '<':
            ineq_type = ilp_instance.smaller_equal
        else:
            assert(con_sense == '>')
            ineq_type = ilp_instance.greater_equal
        ilp.add_new_constraint(con_name, con_vars, con_coeffs, rhs_value, ineq_type)
    return ilp

# Create ILP in python:
# ilp = create_toy_problem()
# opts = bdd_solver_options(ilp)
# Or directly read .lp file from disk as:
opts = bdd_solver_options("PATH_TO_LP_FILE.lp")
opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.lbfgs_cuda_mma # Using GPU FastDOG + LBFGS (default).

# Dual solver options (do not need much tweaking):
opts.dual_max_iter = 10000 # maximum allowed number of iterations, solver can terminate early due to convergence criteria below:
opts.dual_tolerance = 1e-9
opts.dual_improvement_slope = 1e-6
opts.dual_time_limit = 3600 # seconds

# Primal solver options (heuristic, might need tweaking).
opts.incremental_primal_rounding = True # do rounding based on FastDOG paper.
opts.incremental_initial_perturbation = 1.1; opts.incremental_growth_rate = 1.1

# Alternatively try:
# opts.incremental_initial_perturbation = 0.1; opts.incremental_growth_rate = 1.2

# Enable for very long constraints:
# opts.cuda_split_long_bdds = True; opts.cuda_split_long_bdds_length = 1000

# Initialize solver:
solver = bdd_solver(opts)

# Solve dual problem:
solver.solve_dual()

# Run primal heuristic:
obj, sol = solver.round()