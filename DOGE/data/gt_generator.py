import gurobipy as gp
import time
import numpy as np

def get_solution(model):
    vars = model.getVars()
    solution = {}
    obj_value = 0.0
    for var in vars:
        solution[var.VarName] = var.X
        obj_value += var.X * var.Obj # Does not account constant term in objective vector same as BDD solver.
    return solution, obj_value

def generate_gt_gurobi(ilp_path, need_ilp_gt = True):
    """Generate the following using gurobi:
    1. LP relaxation solution, objective and time need to produce it.
    2. original ILP solution, objective and time need to produce it. """
    ilp_gurobi = gp.read(ilp_path)
    num_binary = ilp_gurobi.getAttr('NumBinVars')
    num_vars = ilp_gurobi.getAttr('NumVars')
    # if num_binary != num_vars:
    #     return None, None
    lp_relaxation = ilp_gurobi.relax()
    lp_relaxation.Params.Method = 1 # Dual simplex.
    start_time = time.time()
    lp_relaxation.optimize()
    lp_relaxation_time = time.time() - start_time
    lp_solution, lp_obj_value = get_solution(lp_relaxation)
    lp_stats = {'time': lp_relaxation_time, 'obj': lp_obj_value + lp_relaxation.ObjCon, 'sol_dict': lp_solution}
    ilp_stats = None
    if need_ilp_gt:
        start_time = time.time()
        ilp_gurobi.optimize()
        ilp_time = time.time() - start_time
        ilp_solution, ilp_obj_value = get_solution(ilp_gurobi)
        ilp_stats = {'time': ilp_time, 'obj': ilp_obj_value + ilp_gurobi.ObjCon, 'sol_dict': ilp_solution}
    return lp_stats, ilp_stats


# generate_gt('/home/ahabbas/data/learnDBCA/random_instances_with_lp/IndependentSet_n_nodes_500_edge_probability_0.25_affinity_4_seed_1/ilp_instance_0.lp')