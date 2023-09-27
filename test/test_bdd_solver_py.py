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

# ilp = create_toy_problem()
# opts = bdd_solver_options(ilp)
# Or read .lp file from disk as:
# opts = bdd_solver_options("/home/ahabbas/data/sm_avg_learned_features.lp")
opts = bdd_solver_options("/BS/discrete_opt/nobackup/shape_matching_aaai23/lpfiles/100/FAUST/smcombsiggraph/tr_reg_087-tr_reg_088.lp")

# set options:
opts.dual_max_iter = 2000 # maximum allowed number of iterations, solver can terminate early due to convergence criteria below:
opts.dual_tolerance = 1e-6
opts.dual_improvement_slope = 1e-6
opts.dual_time_limit = 360 # seconds
opts.incremental_primal_rounding = True # do rounding based on FastDOG paper.
opts.incremental_initial_perturbation = 1.1
opts.incremental_growth_rate = 1.1
opts.incremental_primal_num_itr_lb = 100 # number of iterations of dual solver
opts.incremental_primal_num_rounds = 100 # overall max. number of rounding iterations. Solver terminates when first solution is found.

# Set solver type. Available options are:
#    lbfgs_cuda_mma -> FastDOG GPU + LBFGS solver, 
#    lbfgs_parallel_mma -> FastDOG CPU + LBFGS solver, 
#    mma_cuda -> FastDOG GPU solver, 
#    parallel_mma -> FastDOG CPU solver, 
#    sequential_mma -> CPU sequential, 
#    hybrid_parallel_mma -> CPU/GPU hybrid

# opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.mma_cuda
opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.lbfgs_cuda_mma

# LBFGS specific parameters (only used if solver type is lbfgs_cuda_mma or lbfgs_parallel_mma):
opts.lbfgs_step_size = 1e-8
opts.lbfgs_history_size = 5
opts.lbfgs_required_relative_lb_increase = 1e-6
opts.lbfgs_step_size_decrease_factor = 0.8
opts.lbfgs_step_size_increase_factor = 1.1
opts.cuda_split_long_bdds = True
# opts.cuda_split_long_bdds_length = 1000
# Initialize solver:
solver = bdd_solver(opts)

# Solve dual problem:
solver.solve_dual()

# Run primal heuristic:
obj, sol = solver.round()


opts = bdd_solver_options("/BS/discrete_opt/nobackup/shape_matching_aaai23/lpfiles/100/FAUST/smcombsiggraph/tr_reg_087-tr_reg_088.lp")

# set options:
opts.dual_max_iter = 2000 # maximum allowed number of iterations, solver can terminate early due to convergence criteria below:
opts.dual_tolerance = 1e-6
opts.dual_improvement_slope = 1e-6
opts.dual_time_limit = 360 # seconds
opts.incremental_primal_rounding = True # do rounding based on FastDOG paper.
opts.incremental_initial_perturbation = 1.1
opts.incremental_growth_rate = 1.1
opts.incremental_primal_num_itr_lb = 100 # number of iterations of dual solver
opts.incremental_primal_num_rounds = 100 # overall max. number of rounding iterations. Solver terminates when first solution is found.

# Set solver type. Available options are:
#    lbfgs_cuda_mma -> FastDOG GPU + LBFGS solver, 
#    lbfgs_parallel_mma -> FastDOG CPU + LBFGS solver, 
#    mma_cuda -> FastDOG GPU solver, 
#    parallel_mma -> FastDOG CPU solver, 
#    sequential_mma -> CPU sequential, 
#    hybrid_parallel_mma -> CPU/GPU hybrid

# opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.mma_cuda
opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.lbfgs_cuda_mma

# LBFGS specific parameters (only used if solver type is lbfgs_cuda_mma or lbfgs_parallel_mma):
opts.lbfgs_step_size = 1e-8
opts.lbfgs_history_size = 5
opts.lbfgs_required_relative_lb_increase = 1e-6
opts.lbfgs_step_size_decrease_factor = 0.8
opts.lbfgs_step_size_increase_factor = 1.1
opts.cuda_split_long_bdds = False
# opts.cuda_split_long_bdds_length = 1000
# Initialize solver:
solver = bdd_solver(opts)

# Solve dual problem:
solver.solve_dual()

# Run primal heuristic:
obj2, sol2 = solver.round()
breakpoint()
