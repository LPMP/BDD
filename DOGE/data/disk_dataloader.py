import os
import torch
import torch_geometric
import pickle
from tqdm import tqdm
from data.ilp_converters import create_bdd_repr, create_graph_from_bdd_repr, solve_dual_bdd
from data.gt_generator import generate_gt_gurobi

class ILPDiskDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_root_dir, files_to_load, read_dual_converged, 
                    need_gt, need_ilp_gt, need_bdd_constraint_features, 
                    load_in_memory, skip_dual_solved = False, extension = '.lp', use_double_precision = False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_root_dir = data_root_dir
        self.files_to_load = files_to_load
        self.read_dual_converged = read_dual_converged
        self.need_gt = need_gt
        self.need_ilp_gt = need_ilp_gt
        self.use_double_precision = use_double_precision
        self.need_bdd_constraint_features = need_bdd_constraint_features
        self.skip_dual_solved = skip_dual_solved
        self.load_in_memory = load_in_memory
        self.extension = extension
        self.memory_map = []
        self.process_custom()

    @classmethod
    def from_config(cls, cfg, data_name, con_features, skip_dual_solved, use_double_precision = False):
        params = cfg.DATA[data_name + '_PARAMS']
        data_root_dir = params.root_dir
        files_to_load = params.files_to_load
        read_dual_converged = params.read_dual_converged
        need_gt = False
        need_ilp_gt = False
        load_in_memory = False
        extension = '.lp'
        need_bdd_constraint_features = 'con_type' in con_features
        if 'need_gt' in params:
            need_gt = params.need_gt
        if 'need_ilp_gt' in params:
            need_ilp_gt = params.need_ilp_gt
        if 'load_in_memory' in params:
            load_in_memory = params.load_in_memory
        if 'extension' in params:
            extension = params.extension
        return cls(
            data_root_dir = data_root_dir,
            files_to_load = files_to_load,
            read_dual_converged = read_dual_converged,
            need_gt = need_gt,
            need_ilp_gt = need_ilp_gt,
            need_bdd_constraint_features = need_bdd_constraint_features,
            skip_dual_solved = skip_dual_solved,
            load_in_memory = load_in_memory,
            extension = extension,
            use_double_precision = use_double_precision)

    def process_custom(self):
        self.file_list = []
        for path, subdirs, files in os.walk(self.data_root_dir):
            for instance_name in sorted(files):
                if not instance_name.endswith(self.extension) or 'nan' in instance_name or 'normalized' in instance_name or 'slow_bdd' in instance_name or '_one_con' in instance_name or 'oom' in instance_name or 'too_easy' in instance_name:
                    continue
                    
                instance_path = os.path.join(path, instance_name)
                if 'error_bdd' in instance_name:
                    instance_name = instance_name.replace('_error_bdd' + self.extension, self.extension)
                    os.rename(instance_path, os.path.join(path, instance_name))
                    instance_path = os.path.join(path, instance_name)
                sol_name = instance_name.replace(self.extension, '.pkl')
                if 'dual_solved' in instance_name:
                    if self.skip_dual_solved:
                        continue
                    sol_name = sol_name.replace('_dual_solved', '')

                if len(self.files_to_load) > 0 and instance_name not in self.files_to_load:
                    continue

                sol_path = os.path.join(path.replace('instances', 'solutions'), sol_name)
                if not os.path.exists(sol_path):
                    os.makedirs(os.path.dirname(sol_path), exist_ok=True)
                    empty_sol = {'time': None, 'obj': None, 'sol_dict': None, 'sol': None}
                    if self.need_gt:
                        assert '.lp' in self.extension
                        lp_stats, ilp_stats = generate_gt_gurobi(instance_path, self.need_ilp_gt)
                        if ilp_stats == None:
                            ilp_stats = empty_sol
                        gt_info = {"lp_stats": lp_stats, "ilp_stats": ilp_stats}
                        os.makedirs(os.path.dirname(sol_path), exist_ok = True)
                        pickle.dump(gt_info, open(sol_path, "wb"))
                    else:
                        gt_info = {"lp_stats": empty_sol, "ilp_stats": empty_sol}
                        pickle.dump(gt_info, open(sol_path, "wb"))
                else:
                    gt_info = pickle.load(open(sol_path, 'rb'))

                orig_dtype = torch.get_default_dtype()
                if torch.get_default_dtype() == torch.float32 and not self.use_double_precision:
                    bdd_repr_path = instance_path.replace(self.extension, '_bdd_repr.pkl')
                    bdd_repr_conv_path = instance_path.replace(self.extension, '_bdd_repr_dual_converged.pkl')
                    sol_processed_path = sol_path
                    torch.set_default_dtype(torch.float32)
                else:
                    bdd_repr_path = instance_path.replace(self.extension, '_bdd_repr_double.pkl')
                    bdd_repr_conv_path = instance_path.replace(self.extension, '_bdd_repr_dual_converged_double.pkl')
                    sol_processed_name = sol_name.replace('.pkl', '_double.pkl')
                    sol_processed_path = os.path.join(path.replace('instances', 'solutions'), sol_processed_name)
                    torch.set_default_dtype(torch.float64)
                if not os.path.exists(bdd_repr_path):
                    print(f'Creating BDD repr of instance: {instance_path}')
                    bdd_repr, gt_info = create_bdd_repr(instance_path, gt_info, self.need_bdd_constraint_features)
                    if bdd_repr is None:
                        print(f'Removing {instance_path}.')
                        os.rename(instance_path, os.path.join(path, instance_name.replace(self.extension, '_error_bdd' + self.extension)))
                        continue
                    print(f'Saving bdd_repr: {bdd_repr_path}')
                    pickle.dump(bdd_repr, open(bdd_repr_path, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
                    pickle.dump(gt_info, open(sol_processed_path, "wb"))
                if self.read_dual_converged:
                    if not os.path.exists(bdd_repr_conv_path):
                        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
                        print(f'Solving BDD dual of instance: {instance_path}')
                        bdd_repr = solve_dual_bdd(bdd_repr, 1e-6, 50000, 0.5)
                        print(f'Saving converged bdd_repr: {bdd_repr_conv_path}')
                        pickle.dump(bdd_repr, open(bdd_repr_conv_path, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
                    bdd_repr_path = bdd_repr_conv_path # Read dual converged instead.
                self.file_list.append({'instance_path': instance_path, 
                                    'bdd_repr_path': bdd_repr_path,
                                    'sol_path': sol_processed_path, 
                                    'lp_size': os.path.getsize(instance_path)})
                torch.set_default_dtype(orig_dtype) # Revert back to original data type.
        def get_size(elem):
            return elem['lp_size']
        # Sort by size so that largest instances automatically go to end indices.
        self.file_list.sort(key = get_size)

        if self.load_in_memory:
            print('Loading dataset in memory.')
            for index in tqdm(range(self.len())):
                item = self._get_from_disk(index)
                self.memory_map.append(item)

    def _get_from_disk(self, index):
        lp_path = self.file_list[index]['instance_path']
        bdd_repr_path = self.file_list[index]['bdd_repr_path']
        sol_path = self.file_list[index]['sol_path']
        gt_info = pickle.load(open(sol_path, 'rb'))
        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
        return bdd_repr, gt_info, lp_path

    def _get_from_memory(self, index):
        return self.memory_map[index]

    def len(self):
        return len(self.file_list)

    def get(self, index):
        orig_dtype = torch.get_default_dtype()
        if self.use_double_precision:
            torch.set_default_dtype(torch.float64)
        if self.load_in_memory:
            bdd_repr, gt_info, lp_path = self._get_from_memory(index)
        else:
            bdd_repr, gt_info, lp_path = self._get_from_disk(index)
        graph = create_graph_from_bdd_repr(bdd_repr, gt_info, lp_path)
        torch.set_default_dtype(orig_dtype)
        if not self.need_gt:
            gt_info['lp_stats'] = {'time': None, 'obj': None, 'sol_dict': None, 'sol': None}  
        if not self.need_ilp_gt:
            gt_info['ilp_stats'] = {'time': None, 'obj': None, 'sol_dict': None, 'sol': None}
        return graph

