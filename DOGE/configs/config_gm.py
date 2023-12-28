from yacs.config import CfgNode as CN
from configs.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

DATA_DIR = 'datasets/full_inst/' # TODO: Please set accordingly.

cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1
cfg.TRAIN.NUM_DUAL_ITERATIONS = 20
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 20
cfg.TRAIN.NUM_ROUNDS = 20
cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT = 0.0
cfg.TRAIN.MAX_NUM_EPOCHS = 400
cfg.TRAIN.NUM_JOURNEYS = 4
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.USE_REPLAY_BUFFER = True

cfg.MODEL.PREDICT_OMEGA = True
cfg.MODEL.PREDICT_DIST_WEIGHTS = True
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = True

cfg.TEST.NUM_ROUNDS = 50
cfg.TEST.NUM_DUAL_ITERATIONS = 200
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10000

cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'mm_diff', 'prev_sol_avg', 'smooth_sol@0.1', 'smooth_sol@1.0', 'smooth_sol@10.0', 'smooth_sol@100.0']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'mm_diff', 'prev_sol_avg', 'smooth_sol@0.1', 'smooth_sol@1.0', 'smooth_sol@10.0', 'smooth_sol@100.0']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']

cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
WORM_TRAIN = [f'worm0{i}-16-03-11-1745.lp' for i in range(10)]
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': WORM_TRAIN, 'root_dir': DATA_DIR, 'read_dual_converged' : False}) 
cfg.DATA.GM_WORMS_VAL_PARAMS = CN({'files_to_load': ['worm01-16-03-11-1745.lp'], 'root_dir': DATA_DIR, 'read_dual_converged' : False}) 

cfg.TEST.DATA.DATASETS = ['WORM_TEST']
WORM_TEST = [f'worm{i}-16-03-11-1745.lp' for i in range(10, 31)]
cfg.TEST.DATA.WORM_TEST_PARAMS = CN({'files_to_load': WORM_TEST, 'root_dir': DATA_DIR, 'read_dual_converged' : False, 'need_gt' : True}) 

cfg.OUTPUT_ROOT_DIR = 'output_logs/'
cfg.OUT_REL_DIR = 'WORMS/v1/'
