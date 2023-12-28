from yacs.config import CfgNode as CN
from configs.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

DATA_DIR = 'datasets/MIS'

cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1
cfg.TRAIN.NUM_DUAL_ITERATIONS = 20
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 20
cfg.TRAIN.NUM_ROUNDS = 20
cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT = 0.0
cfg.TRAIN.MAX_NUM_EPOCHS = 600
cfg.TRAIN.NUM_JOURNEYS = 6
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.USE_REPLAY_BUFFER = True

cfg.MODEL.PREDICT_OMEGA = True
cfg.MODEL.PREDICT_DIST_WEIGHTS = True
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = True

cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 0.0
cfg.TEST.NUM_ROUNDS = 500
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 50
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 1000000


cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'mm_diff', 'prev_sol_avg', 'smooth_sol@0.1', 'smooth_sol@1.0', 'smooth_sol@10.0', 'smooth_sol@100.0', 'smooth_sol@1000.0']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'mm_diff', 'prev_sol_avg', 'smooth_sol@0.1', 'smooth_sol@1.0', 'smooth_sol@10.0', 'smooth_sol@100.0', 'smooth_sol@1000.0']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']

cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['MIS_TRAIN', 'MIS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.MIS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': f'{DATA_DIR}/train_split/', 'read_dual_converged': False}) 
cfg.DATA.MIS_VAL_PARAMS = CN({'files_to_load': ['0.lp', '1.lp'], 'root_dir': f'{DATA_DIR}/train_split/', 'read_dual_converged' : False}) 

test_datasets, test_params = get_all_lp_instances(
    root_dir = f'{DATA_DIR}/test_split/', data_name = 'MIS', keyword = None, read_converged = False, need_gt = True)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUT_REL_DIR = 'MIS/v1/'