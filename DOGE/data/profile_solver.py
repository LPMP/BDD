import pickle, os
import bdd_cuda_torch
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_add, scatter_std
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_solver
import torch
import numpy as np
import time

num_itr = 10
instance_path = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/instances/flywing_245_bdd_repr.pkl'
bdd_repr = pickle.load(open(instance_path, 'rb'))
solver = pickle.loads(bdd_repr['solver_data'])

lo_costs = torch.empty((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
hi_costs = torch.empty_like(lo_costs)
def_mm = torch.zeros_like(lo_costs)
    
omega = torch.ones((solver.nr_layers()), device = 'cuda', dtype = torch.float32) - 0.5
dist_weights = torch.ones((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
var_indices = torch.from_numpy(bdd_repr['var_indices']).to(omega.device).to(torch.long)
con_indices = torch.from_numpy(bdd_repr['con_indices']).to(omega.device).to(torch.long)
dist_weights = scatter_softmax(dist_weights, var_indices)

start_time = time.time()
solver.set_solver_costs(lo_costs.data_ptr(), hi_costs.data_ptr(), def_mm.data_ptr())
end_time = time.time()
print(f"Costs copy time: {end_time - start_time}")

start_time = time.time()
solver.iterations(dist_weights.data_ptr(), num_itr, 1.0, 0.0, omega.data_ptr(), True)
end_time = time.time()
print(f"Learned solver time: {end_time - start_time}")

start_time = time.time()
solver.non_learned_iterations(0.5, num_itr, 0.0)
end_time = time.time()
print(f"FastDOG solver time: {end_time - start_time}")