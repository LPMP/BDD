import torch
import torch_geometric
import numpy as np
import torch.utils.data
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_solver
import pickle
import gurobipy as gp

def create_normalized_bdd_instance(ilp_path):
    ilp_gurobi = gp.read(ilp_path)
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
        # GM can contain continuous variables even though they will ultimately take a binary value.
        # if (var.VType != 'B'): #, f'Variable {var} is not binary in file {ilp_path} and instead of type {var.VType}'
        #     print(f'Non-binary variable.')
        #     return None, None, None
        objs.append(var.Obj)
        var_names.append(var.VarName) 

    # Scale the objective vector to [-1, 1]. (This will change the predicted objective value).
    obj_multiplier = obj_multiplier / (1e-6 + np.abs(np.array(objs)).max())
    for i in range(len(objs)):
        ilp_bdd.add_new_variable_with_obj(var_names[i], obj_multiplier * float(objs[i]))

    max_coeff = 0
    for con in cons:
        multiplier = 1.0
        rhs_value = con.RHS
        if con.Sense == '=':
            ineq_type = ilp_instance_bbd.equal
        elif con.Sense == '<':
            ineq_type = ilp_instance_bbd.smaller_equal
        else:
            assert(con.Sense == '>')
            # Converting all constraints to <= or = representation for GNN.
            multiplier = -1.0
            ineq_type = ilp_instance_bbd.smaller_equal
        
        constraint_var_names = []; constraint_var_coefficients =  []
        constraint_exp = ilp_gurobi.getRow(con)
        for i in range(constraint_exp.size()):
            var = constraint_exp.getVar(i).VarName
            coeff = constraint_exp.getCoeff(i)
            if int(coeff) != coeff or int(rhs_value) != rhs_value: #TODO: Handle fractional values in constraints. 
                print(f'Fractional value in constraint.')
                return None, None, None

            constraint_var_names.append(str(var))
            constraint_var_coefficients.append(int(coeff * multiplier))
            max_coeff = max(max_coeff, coeff)
        ilp_bdd.add_new_constraint(con.ConstrName, constraint_var_names, constraint_var_coefficients, int(rhs_value * multiplier), ineq_type)
    return ilp_bdd, obj_multiplier, obj_offset

def map_solution_order(solution_dict, bdd_ilp_instance):
    sol = np.zeros(len(solution_dict)) - 1
    for var_name, value in solution_dict.items():
        var_index = bdd_ilp_instance.get_var_index(var_name)
        sol[var_index] = value
    assert(sol.min() > - 1)
    return sol

def create_bdd_repr(instance_path, gt_info, load_constraint_coeffs = False):
    if instance_path.endswith('.lp'):
        bdd_ilp_instance, obj_multiplier, obj_offset = create_normalized_bdd_instance(instance_path)
        if torch.get_default_dtype() == torch.float32:
            print('solver_type_fp32')
            solver = bdd_solver.bdd_cuda_learned_mma(bdd_ilp_instance, load_constraint_coeffs, 1.0)
        elif torch.get_default_dtype() == torch.float64:
            print('solver_type_fp64')
            solver = bdd_solver.bdd_cuda_learned_mma_double(bdd_ilp_instance, load_constraint_coeffs, 1.0)
        else:
            print(f'Unsupported dtype {torch.get_default_dtype()}.')
    elif instance_path.endswith('.uai'):
        bdd_ilp_instance = ilp_instance_bbd.read_MRF_UAI(instance_path)
        objective = bdd_ilp_instance.objective()
        obj_multiplier = 1.0 / (1e-6 + np.abs(np.array(objective)).max())
        obj_offset = 0.0
        if torch.get_default_dtype() == torch.float32:
            print('solver_type_fp32')
            solver = bdd_solver.bdd_cuda_learned_mma(bdd_ilp_instance, load_constraint_coeffs, obj_multiplier)
        elif torch.get_default_dtype() == torch.float64:
            print('solver_type_fp64')
            solver = bdd_solver.bdd_cuda_learned_mma_double(bdd_ilp_instance, load_constraint_coeffs, obj_multiplier)
        else:
            print(f'Unsupported dtype {torch.get_default_dtype()}.')
    else:
        assert False

    if bdd_ilp_instance is None:
        return None, None

    num_vars = solver.nr_primal_variables() + 1 # +1 due to terminal node.
    num_cons = solver.nr_bdds()
    num_layers = solver.nr_layers()
    var_indices = torch.empty((solver.nr_layers()), dtype = torch.int32, device = 'cuda')
    solver.primal_variable_index(var_indices.data_ptr())
    con_indices = torch.empty((solver.nr_layers()), dtype = torch.int32, device = 'cuda')
    solver.bdd_index(con_indices.data_ptr())
    var_indices = var_indices.cpu().numpy()
    con_indices = con_indices.cpu().numpy()

    num_extra_vars_decomp = solver.nr_primal_variables() - bdd_ilp_instance.nr_variables()
    objective = np.concatenate((bdd_ilp_instance.objective(), [0] * num_extra_vars_decomp, [0])) # Decomposition variables can be present with obj 0.
    # Map gt solution variables indices according to BDD variable order:
    for stat in ['lp_stats', 'ilp_stats']:
        if stat in gt_info and gt_info[stat]['sol_dict'] is not None:
            assert(len(gt_info[stat]['sol_dict']) == bdd_ilp_instance.nr_variables())
            gt_info[stat]['sol'] = map_solution_order(gt_info[stat]['sol_dict'], bdd_ilp_instance)
            gt_obj = gt_info[stat]['obj']
            computed_obj = np.sum(gt_info[stat]['sol'] * objective[:bdd_ilp_instance.nr_variables()]) / obj_multiplier + obj_offset
            assert np.abs(computed_obj - gt_obj) < 1e-3, f'GT objectives mismatch for {instance_path}. GT obj: {gt_obj}, Computed obj: {computed_obj}.'

    assert(objective.shape[0] == num_vars)

    if load_constraint_coeffs:
    # Encode constraints as features assuming that constraints are linear:
        assert solver.nr_primal_variables() == bdd_ilp_instance.nr_variables(), f'Found {solver.nr_primal_variables()} variables in solver and {bdd_ilp_instance.nr_variables()} in ILP read by BDD solver.'
        assert solver.nr_bdds() == bdd_ilp_instance.nr_constraints(), f'Found {solver.nr_bdds()} BDDs in solver and {bdd_ilp_instance.nr_constraints()} constraints in ILP {instance_path} read by BDD solver.'
        try:
            coefficients = solver.constraint_matrix_coeffs(bdd_ilp_instance)
        except:
            breakpoint()
        coefficients = np.array(coefficients)
        con_bounds = bdd_ilp_instance.variable_constraint_bounds()[num_vars - 1:, :]
        # Convert bounds w.r.t constraints to bounds w.r.t BDDs:
        bdd_to_constraint_map = solver.bdd_to_constraint_map()
        bdd_con_bounds = con_bounds[bdd_to_constraint_map, :]
        # # Constraint features:
        # # bounds containst value of lb, ub meaning: lb <= constraint <= ub.
        bounds = torch.as_tensor(bdd_con_bounds)
        lb_cons = bounds[:, 0]
        ub_cons = bounds[:, 1]
        # lb <=(geq type) a^{T}x <=(leq type) ub. (lb can be equal to ub then the constraint is both leq and geq.)
        leq_cons = lb_cons <= np.iinfo(np.intc).min
        geq_cons = ub_cons >= np.iinfo(np.intc).max
        assert(~torch.any(torch.logical_and(leq_cons, geq_cons)))
        leq_type = torch.ones((num_cons))
        geq_type = torch.ones((num_cons))
        leq_type[geq_cons] = 0
        geq_type[leq_cons] = 0
        if torch.abs((1 - leq_type) * geq_type).max() > 0:
            raise ValueError('all constraints should be <= or = type')
        rhs_vector = lb_cons
        rhs_vector[leq_cons] = ub_cons[leq_cons]
        rhs_vector = rhs_vector.numpy()
        leq_type = leq_type.numpy()
    else:
        rhs_vector = np.zeros((num_cons)) # To create batch index for constraint nodes.
        coefficients = np.zeros((num_layers))
        leq_type = np.zeros((num_cons))
    bdd_repr = {
                    "solver_data": pickle.dumps(solver, -1), # bytes representation of bdd cuda solver internal variables.
                    "num_vars": num_vars, "num_cons": num_cons, "num_layers": num_layers,
                    "var_indices": var_indices, "con_indices": con_indices,
                    "objective": objective, 
                    "coeffs": coefficients, "rhs_vector": rhs_vector,
                    "constraint_type": leq_type, # Contains 1 for <= constraint and 0 for equality, where >= constraints should not be present.
                    "obj_multiplier": obj_multiplier, 
                    "obj_offset": obj_offset
                }
    return bdd_repr, gt_info

def solve_dual_bdd(bdd_repr, improvement_slope, num_iterations, omega):
    solver = pickle.loads(bdd_repr['solver_data'])
    solver.non_learned_iterations(omega, num_iterations, improvement_slope)
    solver.distribute_delta() # make deferred min-marginals zero.
    bdd_repr['solver_data'] = pickle.dumps(solver, -1) # Overwrite bdd representation with update costs.
    return bdd_repr

def create_graph_from_bdd_repr(bdd_repr, gt_info, file_path):
    graph = BipartiteVarConDataset(num_vars = bdd_repr['num_vars'], num_cons = bdd_repr['num_cons'], num_layers = bdd_repr['num_layers'],
                                var_indices = torch.from_numpy(bdd_repr['var_indices']).to(torch.long), 
                                con_indices = torch.from_numpy(bdd_repr['con_indices']).to(torch.long), 
                                objective = torch.from_numpy(bdd_repr['objective']).to(torch.get_default_dtype()),
                                con_coeffs = torch.from_numpy(bdd_repr['coeffs']).to(torch.get_default_dtype()), 
                                rhs_vector = torch.from_numpy(bdd_repr['rhs_vector']).to(torch.get_default_dtype()),
                                con_type = torch.from_numpy(bdd_repr['constraint_type']).to(torch.get_default_dtype()),
                                obj_multiplier = bdd_repr['obj_multiplier'],
                                obj_offset = bdd_repr['obj_offset'])

    graph.num_nodes = graph.num_vars + graph.num_cons
    
    # Append 0 to solution vector.
    for stat in ['lp_stats', 'ilp_stats']:
        if stat in gt_info and gt_info[stat]['sol_dict'] is not None:
            assert('sol_dict' in gt_info[stat])
            # assert(gt_info[stat]['sol'].size + 1 == bdd_repr['num_vars']) # not true for miplib.
            gt_info[stat]['sol'] = np.concatenate((gt_info[stat]['sol'], np.array([0.0])), 0)
            gt_info[stat]['sol_dict'] = None # pyg tries to collate it very slowly.

    graph.gt_info = gt_info
    graph.solver_data = bdd_repr['solver_data']
    graph.file_path = file_path
    return graph

class BipartiteVarConDataset(torch_geometric.data.Data):
    def __init__(self, num_vars, num_cons, num_layers, var_indices, con_indices, objective, con_coeffs, rhs_vector, con_type, obj_multiplier, obj_offset):
        super(BipartiteVarConDataset, self).__init__()
        # super().__init__()
        self.num_vars = num_vars
        self.num_cons = num_cons
        self.num_layers = num_layers
        self.objective = objective
        self.con_coeff = con_coeffs
        self.rhs_vector = rhs_vector
        self.con_type = con_type 
        self.obj_multiplier = obj_multiplier
        self.obj_offset =  obj_offset
        if var_indices is not None:
            assert(con_coeffs.shape == var_indices.shape)
            assert(con_coeffs.shape == con_indices.shape)
            assert(torch.numel(self.rhs_vector) == self.num_cons)
            self.edge_index_var_con = torch.stack((var_indices, con_indices))
            self.num_edges = torch.numel(var_indices)
        else:
            self.edge_index_var_con = None
            self.num_edges = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_var_con':
            return torch.tensor([[self.num_vars], [self.num_cons]])
        # if key == 'var_indices':
        #     return torch.tensor([self.num_vars])
        else:
            return super().__inc__(key, value, *args, **kwargs)