import ecole, os
from tqdm import tqdm
import gurobipy as gp

# For training data:
num_instances_train = 240
n_nodes_train = 10000
edge_probability_train = 0.25
affinity_train = 4

# For testing data:
num_instances_test = 60
n_nodes_test = 50000
edge_probability_test = 0.25
affinity_test = 8

seed = 1311

def CreateInstances(num_instances, n_nodes, edge_probability, affinity, out_dir):
    os.makedirs(out_dir, exist_ok = True)
    random_instance_generator = ecole.instance.IndependentSetGenerator(n_nodes = n_nodes, edge_probability = edge_probability, affinity = affinity)
    random_instance_generator.seed(seed)
    for n in tqdm(range(num_instances)):
        out_path = os.path.join(out_dir, f'{n}.lp')
        if not os.path.exists(out_path):
            ilp_ecole = next(random_instance_generator)
            ilp_scip = ilp_ecole.as_pyscipopt()
            ilp_scip.writeProblem(out_path)

def convert_minimization(root_dir):
    for path, subdirs, files in os.walk(root_dir):
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp'):
                continue
            ilp_path = os.path.join(path, instance_name)
            ilp_gurobi = gp.read(ilp_path)
            if ilp_gurobi.ModelSense == -1: # Maximization
                print(f'In file: {ilp_path}, detected maximization instead of minimization. Changing to minimzation objective.')
                variables = ilp_gurobi.getVars()
                for var in variables:
                    var.Obj = -1.0 * var.Obj
                ilp_gurobi.ModelSense = 1
                ilp_gurobi.update()
                ilp_gurobi.write(ilp_path)

def presolve_conservative(root_dir):
    for name in os.listdir(root_dir):
        if not name.endswith('.lp'):
            continue

        in_path = os.path.join(root_dir, name)
        out_path = in_path
        ilp_gurobi = gp.read(in_path)
        ilp_gurobi.setParam('Presolve', 1) # conservative.
        ilp_gurobi = ilp_gurobi.presolve()
        ilp_gurobi.update()
        num_binary = ilp_gurobi.getAttr('NumBinVars')
        num_vars = ilp_gurobi.getAttr('NumVars')
        assert(num_binary == num_vars)
        ilp_gurobi.write(out_path)

train_path = 'datasets/MIS/train_split/instances/'
test_path  = 'datasets/MIS/test_split/instances/'
CreateInstances(num_instances_train, n_nodes_train, edge_probability_train, affinity_train, train_path)
CreateInstances(num_instances_test, n_nodes_test, edge_probability_test, affinity_test, test_path)

convert_minimization(train_path)
convert_minimization(test_path)

presolve_conservative(train_path)
presolve_conservative(test_path)
