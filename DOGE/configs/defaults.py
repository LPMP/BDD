from yacs.config import CfgNode as CN
import os

def get_all_lp_instances(root_dir, data_name, keyword = None, read_converged = False, need_gt = False):
    datasets = [data_name]
    files_to_load = []
    for path, subdirs, files in os.walk(root_dir):
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
                continue
            
            if keyword is None or not keyword in instance_name:
                continue

            files_to_load.append(instance_name)

    all_params = {}
    all_params[data_name + '_PARAMS'] = CN({
            'root_dir': root_dir, 
            'files_to_load': files_to_load,
            'read_dual_converged' : read_converged,
            'need_gt': need_gt,
            'need_ilp_gt': False})
    return datasets, all_params

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.PREDICT_OMEGA = True
cfg.MODEL.PREDICT_DIST_WEIGHTS = True
cfg.MODEL.VAR_FEATURE_DIM = 16
cfg.MODEL.CON_FEATURE_DIM = 16
cfg.MODEL.EDGE_FEATURE_DIM = 8
cfg.MODEL.FEATURE_EXTRACTOR_DEPTH = 1
cfg.MODEL.DUAL_PRED_DEPTH = 1
cfg.MODEL.CKPT_PATH = None
cfg.MODEL.OMEGA_INITIAL = 0.5
cfg.MODEL.USE_LAYER_NORM = True
cfg.MODEL.VAR_LP_FEATURES = ['obj', 'deg']
cfg.MODEL.VAR_LP_FEATURES_INIT = ['obj', 'deg']
cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']
cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff'] 
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff']
cfg.MODEL.NUM_HIDDEN_LAYERS_EDGE = 2
cfg.MODEL.USE_NET_SOLVER_COSTS = True
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = False
cfg.MODEL.DENORM_FREE_UPDATE = False
cfg.MODEL.SCALE_FREE_UPDATE = False
cfg.MODEL.USE_CELU_ACTIVATION = False
cfg.MODEL.USE_SEPARATE_MODEL_LATER_STAGE = True
cfg.MODEL.MP_AGGR = 'mean'
cfg.MODEL.USE_SOLVER_COST_AS_FEATURE = True

cfg.DATA = CN()
# Number of workers for data loader
cfg.DATA.DISK_DATA_ROOT = '/home/ahabbas/data/learnDBCA/cv_structure_pred/'
cfg.DATA.NUM_WORKERS = 0

cfg.DATA.DATASETS = []
cfg.DATA.VAL_FRACTION = []

cfg.TRAIN = CN()
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.MAX_NUM_EPOCHS = 300
cfg.TRAIN.OPTIMIZER = "Adam"
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False
cfg.TRAIN.NUM_JOURNEYS = 10

cfg.TRAIN.NUM_ROUNDS = 1 # Max. possible number of dual iteration rounds.
cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1 # Number of rounds in which gradients are backpropagated.
cfg.TRAIN.NUM_DUAL_ITERATIONS = 10
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 10 # Gradient of dual iterations would be backpropagated for a maximum of last min(GRAD_DUAL_ITR_MAX_ITR, NUM_DUAL_ITERATIONS) many iterations.

cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 0.0
cfg.TRAIN.LOSS_DISCOUNT_FACTOR = 1.0
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 25
cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT = 0.0
cfg.TRAIN.GRAD_CLIP_VAL = 50.0
cfg.TRAIN.USE_REPLAY_BUFFER = True

cfg.TEST = CN()
cfg.TEST.NUM_DUAL_ITERATIONS = 10
cfg.TEST.NUM_ROUNDS = 1 # How many times dual iterations.
cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 1e-9
cfg.TEST.VAL_BATCH_SIZE = 1
cfg.TEST.PERIOD = 50 # Validate after every n epoch (can be less than 1).
cfg.TEST.VAL_PERIOD = 10000000
cfg.LOG_EVERY = 100

cfg.TEST.DATA = CN() # Stores dataset params used for testing only.
cfg.TEST.DATA.DATASETS = ['DATASET_TEST']
cfg.TEST.DATA.QAP_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '', 'read_dual_converged' : False, 'need_gt': False})

cfg.SEED = 1
cfg.OUTPUT_ROOT_DIR = 'output_logs/'
cfg.OUT_REL_DIR = 'v1/'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`