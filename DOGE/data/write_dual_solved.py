import pickle, os, tqdm, argparse

def solve_dual_bdd(bdd_repr, improvement_slope, num_iterations, omega):
    solver = pickle.loads(bdd_repr['solver_data'])
    solver.non_learned_iterations(omega, num_iterations, improvement_slope)
    solver.distribute_delta() # make deferred min-marginals zero.
    bdd_repr['solver_data'] = pickle.dumps(solver, -1) # Overwrite bdd representation with update costs.
    return bdd_repr

def generate_ilps(root_dir, suffix, improvement_slope, num_iterations, omega):
    file_list = []
    for path, directories, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.pkl'):
                continue
            file_list.append([path, file])

    for bdd_rep_path, filename in tqdm.tqdm(file_list):
        ext = os.path.splitext(filename)[1]
        out_filename = filename.replace(ext, suffix + ext)
        bdd_repr = pickle.load(open(os.path.join(bdd_rep_path, filename), 'rb'))
        solve_dual_bdd(bdd_repr, improvement_slope, num_iterations, omega)
        pickle.dump(open(os.path.join(bdd_rep_path, out_filename), 'wb'))

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path to config root dir containing .lp files in any child dirs.")
parser.add_argument("output_suffix", default='_dual_solved')
parser.add_argument("num_iterations", default=50000)
parser.add_argument("improvement_slope", default=1e-6)
parser.add_argument("omega", default=0.5)
args = parser.parse_args()

root_dir = args.input_dir
output_suffix = args.output_suffix
generate_ilps(root_dir, output_suffix, args.improvement_slope, args.num_iterations, args.omega)
