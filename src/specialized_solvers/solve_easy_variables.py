import BDD.ILP_instance_py as ilp_instance
from BDD.bdd_solver_py import bdd_solver as bdd_solver
from BDD.bdd_solver_py import bdd_solver_options as bdd_solver_options
import gurobipy as gp
import numpy as np
import argparse

def solve_fastdog(ilp_path, num_primal_rounds = 100):
    opts = bdd_solver_options(ilp_path)

    # set options:
    opts.dual_max_iter = 10000 # maximum allowed number of iterations, solver can terminate early due to convergence criteria below:
    opts.dual_tolerance = 1e-6
    opts.dual_improvement_slope = 1e-6
    opts.dual_time_limit = 360 # seconds
    opts.incremental_primal_rounding = True # do rounding based on FastDOG paper.
    opts.incremental_initial_perturbation = 1.1
    opts.incremental_growth_rate = 1.2
    opts.incremental_primal_num_itr_lb = 100 # number of iterations of dual solver inside primal rounding.
    opts.bdd_solver_type = bdd_solver_options.bdd_solver_types.lbfgs_cuda_mma
    opts.lbfgs_step_size = 1e-6
    opts.lbfgs_history_size = 5
    opts.lbfgs_required_relative_lb_increase = 1e-6
    opts.lbfgs_step_size_decrease_factor = 0.8
    opts.lbfgs_step_size_increase_factor = 1.1
    opts.cuda_split_long_bdds = False
    opts.incremental_primal_num_rounds = num_primal_rounds

    # Initialize solver:
    solver = bdd_solver(opts)
    
    # Solve dual problem:
    solver.solve_dual()
    lb = solver.lower_bound()
    # Run primal heuristic:
    obj, sol = solver.round()
    mm_info = {"lb": lb, "sol": sol, "obj": obj}
    if len(sol) == 0: # no solution found. Compute additional information for fixation.
        var_names, mm_lo, mm_hi = solver.min_marginals_with_var_names()
        mm_info.update({"var_names": var_names, "mm_lo": mm_lo, "mm_hi": mm_hi})
    return mm_info

def read_ilp_and_bounds(ilp_path):
    ilp_gurobi = gp.read(ilp_path)
    lbs = np.array(ilp_gurobi.getAttr("lb", ilp_gurobi.getVars()))
    ubs = np.array(ilp_gurobi.getAttr("ub", ilp_gurobi.getVars()))
    return ilp_gurobi, lbs, ubs

def compute_mm_type(var_names, mm_diff, threshold):
    mm_type = {}
    for i in range(len(var_names)):
        var_name = var_names[i]
        cur_mm_diff = mm_diff[i]
        if cur_mm_diff >= threshold:
            direction = 'zero'
        elif cur_mm_diff >= threshold:
            direction = 'one'
        else:
            direction = 'undecided'
        existing_direction = mm_type.get(var_name, None)
        if existing_direction is None:
            mm_type[var_name] = direction
        elif existing_direction != direction:
            mm_type[var_name] = 'undecided'
    return mm_type

def fix_consistent_variables_solve_gurobi(ilp_gurobi, mm_type):
    number_fixations = 0
    for var_name, var_mm_type in mm_type.items():
        if var_mm_type == 'undecided':
            continue
        var_gur = ilp_gurobi.getVarByName(var_name)
        if var_mm_type == 'zero':
            var_gur.lb = 0
            var_gur.ub = 0
        elif var_mm_type == 'one':
            var_gur.lb = 1
            var_gur.ub = 1
        else:
            assert(False)
        number_fixations += 1
    print(f'Fixed: {number_fixations} / {len(mm_type)} = {100.0 * number_fixations / len(mm_type)}% of variables.', flush = True)
    ilp_gurobi.update()
    ilp_gurobi.optimize()
    return ilp_gurobi

parser = argparse.ArgumentParser()
parser.add_argument("ilp_path", type = str)
parser.add_argument("num_primal_rounds", default=100, type = int,
    help = "Number of iteration of primal rounding to execute in FastDOG before fixation.")
parser.add_argument("--fix_percentiles", default=[5], nargs="+", type = int,
    help = "Percentages of variable to keep unfixed from fastdog. Larger value can thus lead to better solution but lower runtimes"
    "as a larger problem would be delegated to Gurobi.")
args = parser.parse_args()

print("Command Line Args:", flush = True)
print(args, flush = True)

num_primal_rounds = args.num_primal_rounds
percentiles = args.fix_percentiles
ilp_path = args.ilp_path
mm_info = solve_fastdog(ilp_path, num_primal_rounds)
lower_bound = mm_info['lb']
print(f'FastDOG lower bound: {lower_bound}.')
fastdog_sol = mm_info['sol']
if len(fastdog_sol) > 0:
    obj_val = mm_info['obj']
    print(f'Directly solved {ilp_path} via FastDOG. Lower bound: {lower_bound}, Upper bound: {obj_val}')
else:
    # Try by fixing highly certain variables and solve rest by gurobi.
    ilp_gurobi, var_lbs, var_ubs = read_ilp_and_bounds(ilp_path)
    var_names = mm_info['var_names']
    mm_diff = np.array(mm_info['mm_hi']) - np.array(mm_info['mm_lo'])
    abs_mm_diff = np.abs(mm_diff)
    solved = False
    for perc in sorted(percentiles): # Try from smaller values to larger ones, as smaller lead to faster runtime.
        mm_th = np.percentile(abs_mm_diff, perc)
        print(f'\n\nSolving for percentile: {perc}, with mm-diff threshold: {mm_th}', flush = True)
        mm_types = compute_mm_type(var_names, mm_diff, mm_th)
        ilp_gurobi = fix_consistent_variables_solve_gurobi(ilp_gurobi, mm_types)
        if ilp_gurobi.status == 3:
            print(f'Infeasible. Continuing.', flush = True)
            continue
        sol = np.array(ilp_gurobi.getAttr("X", ilp_gurobi.getVars()))
        obj_val = ilp_gurobi.objVal
        lower_bound = mm_info['lb']
        solved = True
        print(f'Solved {ilp_path} via FastDOG + Gurobi. Lower bound: {lower_bound}, Upper bound: {obj_val}')
        break # Can continue if we want to explore better solutions with larger percentage unfixed. 
    if not solved:
        print(f'Unable to solve {ilp_path}. Try Gurobi directly without fixation. ')
