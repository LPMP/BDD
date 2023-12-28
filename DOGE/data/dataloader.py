import torch 
from data.disk_dataloader import ILPDiskDataset
from torch_geometric.loader import DataLoader

def get_ilp_gnn_loaders(cfg, skip_dual_solved = False, test_only = False, test_precision_double = False, test_on_train = False, train_precision_double = False):
    combined_train_loader = None
    val_loaders = []
    val_datanames = []
    if not test_only:
        all_train_datasets = []
        for data_name, val_fraction in zip(cfg.DATA.DATASETS, cfg.DATA.VAL_FRACTION):
            full_dataset = ILPDiskDataset.from_config(cfg, data_name, cfg.MODEL.CON_LP_FEATURES, skip_dual_solved, use_double_precision = train_precision_double)

            val_size = int(val_fraction * len(full_dataset))
            train_size = len(full_dataset) - val_size
            if val_size > 0 and train_size > 0:
                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            elif train_size > 0:
                train_dataset = full_dataset
                val_dataset = None 
            else:
                train_dataset = None 
                val_dataset = full_dataset
            
            if train_dataset is not None:
                all_train_datasets.append(train_dataset)
            if val_dataset is not None:
                # test datasets are not combined, they are kept separate to compute per dataset eval metrics.
                val_loaders.append(DataLoader(val_dataset, 
                                        batch_size=cfg.TEST.VAL_BATCH_SIZE, 
                                        shuffle=False, 
                                        follow_batch = ['objective', 'rhs_vector', 'edge_index_var_con'], 
                                        num_workers = cfg.DATA.NUM_WORKERS))
                val_datanames.append(data_name)
        if len(all_train_datasets) > 0:
            combined_train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
            combined_train_loader = DataLoader(combined_train_dataset, 
                                            batch_size=cfg.TRAIN.BATCH_SIZE, 
                                            shuffle=True, 
                                            follow_batch = ['objective', 'rhs_vector', 'edge_index_var_con'], 
                                            num_workers = cfg.DATA.NUM_WORKERS)

    test_loaders = []
    test_data_names = []
    if not test_on_train:
        for test_data_name in cfg.TEST.DATA.DATASETS:
            test_dataset = ILPDiskDataset.from_config(cfg.TEST, test_data_name, cfg.MODEL.CON_LP_FEATURES, skip_dual_solved = False, use_double_precision = test_precision_double)
            test_loaders.append(DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    follow_batch = ['objective', 'rhs_vector', 'edge_index_var_con'], 
                                    num_workers = cfg.DATA.NUM_WORKERS))
            test_data_names.append(test_data_name)
    else:
        for data_name in cfg.DATA.DATASETS:
            test_dataset = ILPDiskDataset.from_config(cfg, data_name, cfg.MODEL.CON_LP_FEATURES, skip_dual_solved = False, use_double_precision = test_precision_double)
            test_loaders.append(DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    follow_batch = ['objective', 'rhs_vector', 'edge_index_var_con'], 
                                    num_workers = cfg.DATA.NUM_WORKERS))
            test_data_names.append(data_name)

    return combined_train_loader, val_loaders, val_datanames, test_loaders, test_data_names