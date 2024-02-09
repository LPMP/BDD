import BDD.ILP_instance_py as ilp_instance_bbd
import gurobipy as gp

def create_bdd_instance_from_gurobi_model(ilp_gurobi):
    obj_offset = ilp_gurobi.ObjCon
    obj_multiplier = 1.0
    if ilp_gurobi.ModelSense == -1: # Maximization
        print(f'In file: {ilp_path}, detected maximization instead of minimization. Creating BDD representation with minimzation objective.')
        obj_multiplier = -1.0
    variables = ilp_gurobi.getVars()
    cons = ilp_gurobi.getConstrs()
    ilp_bdd = ilp_instance_bbd.ILP_instance()
    objs = []
    var_names = []
    for var in variables:
        assert var.VType == 'B', f'Variable {var} is not binary and is instead of type {var.VType}.'
        objs.append(var.Obj)
        var_names.append(var.VarName) 

    for i in range(len(objs)):
        ilp_bdd.add_new_variable_with_obj(var_names[i], obj_multiplier * float(objs[i]))

    for con in cons:
        rhs_value = con.RHS
        if con.Sense == '=':
            ineq_type = ilp_instance_bbd.equal
        elif con.Sense == '<':
            ineq_type = ilp_instance_bbd.smaller_equal
        else:
            ineq_type = ilp_instance_bbd.greater_equal        
        constraint_var_names = []; constraint_var_coefficients =  []
        constraint_exp = ilp_gurobi.getRow(con)
        for i in range(constraint_exp.size()):
            var = constraint_exp.getVar(i).VarName
            coeff = constraint_exp.getCoeff(i)
            assert int(coeff) == coeff, f'fractional lhs coefficient in constraint {con}.'
            assert int(rhs_value) == rhs_value, f'fractional rhs in constraint {con}.'
            constraint_var_names.append(str(var))
            constraint_var_coefficients.append(int(coeff))
        ilp_bdd.add_new_constraint(con.ConstrName, constraint_var_names, constraint_var_coefficients, int(rhs_value), ineq_type)
    return ilp_bdd, obj_multiplier, obj_offset

def map_bdd_solution(ilp_gurobi, solution_from_bdd, ilp_bdd):
    sol = {}
    for var in variables:
        index_in_bdd_sol = ilp_bdd.get_var_index(var.VarName)
        sol[var.VarName] = solution_from_bdd[index_in_bdd_sol]
    return sol

ilp_path = 'SOME_FILE.lp'
ilp_gurobi = gp.read(ilp_path)
# Any gurobi related tasks here e.g., presolving.
ilp_bdd, obj_multiplier, obj_offset = create_bdd_instance_from_gurobi_model(ilp_gurobi)

# Run BDD solver here giving primal solution 'solution_from_bdd'

# Afterwards map the solution to variable names.
sol = map_bdd_solution(ilp_gurobi, solution_from_bdd, ilp_bdd)
