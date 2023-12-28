from yacs.config import CfgNode as CN
from configs.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

DATA_DIR = 'datasets/full_inst/' # TODO: Please set accordingly.

cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1
cfg.TRAIN.NUM_DUAL_ITERATIONS = 1
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 1
cfg.TRAIN.NUM_ROUNDS = 400
cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT = 0.0
cfg.TRAIN.MAX_NUM_EPOCHS = 1025
cfg.TRAIN.NUM_JOURNEYS = 10
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.GRAD_CLIP_VAL = 50.0
cfg.TRAIN.USE_REPLAY_BUFFER = True
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 25

cfg.MODEL.PREDICT_OMEGA = False
cfg.MODEL.PREDICT_DIST_WEIGHTS = False
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = True
cfg.MODEL.USE_SEPARATE_MODEL_LATER_STAGE = True

cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 0.0
cfg.TEST.NUM_ROUNDS = 500
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 1000000 # Validate after every n epoch (can be less than 1).

cfg.MODEL.VAR_LP_FEATURES = ['deg', 'obj']
cfg.MODEL.VAR_LP_FEATURES_INIT = ['deg', 'obj']

cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'mm_diff', 'smooth_sol@1.0', 'smooth_sol@10.0', 'smooth_sol@100.0', 'smooth_sol@1000.0'] #, 'prev_sol_avg']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'mm_diff', 'smooth_sol@1.0', 'smooth_sol@10.0', 'smooth_sol@100.0', 'smooth_sol@1000.0'] #, 'prev_sol_avg']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_change']

cfg.DATA.NUM_WORKERS = 1
cfg.DATA.DATASETS = ['CT_TRAIN', 'CT_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

CT_TRAIN_FILES = ['flywing_100_1.lp', 'flywing_100_2.lp']

cfg.DATA.CT_TRAIN_PARAMS = CN({'files_to_load': CT_TRAIN_FILES, 'root_dir': DATA_DIR, 'read_dual_converged' : False, 'need_gt': False, 'load_in_memory': True})
cfg.DATA.CT_VAL_PARAMS = CN({'files_to_load': CT_TRAIN_FILES, 'root_dir': DATA_DIR, 'read_dual_converged' : False, 'need_gt': False, 'load_in_memory': True})

cfg.TEST.DATA.DATASETS = ['CT_TEST']
CT_TEST_FILES =['flywing_245.lp']
cfg.TEST.DATA.CT_TEST_PARAMS = CN({'files_to_load': CT_TEST_FILES, 'root_dir': DATA_DIR, 'read_dual_converged' : False, 'need_gt': True})

cfg.OUTPUT_ROOT_DIR = 'output_logs/'
cfg.OUT_REL_DIR = 'CT_LARGE/v1/'