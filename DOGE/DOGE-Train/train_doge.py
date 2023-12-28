from ensurepip import version
from genericpath import isfile
import os, argparse
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*if you want to see logs for the training epoch.*")
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
torch.use_deterministic_algorithms(False)
from configs.defaults import get_cfg_defaults
from data.dataloader import get_ilp_gnn_loaders
from doge import DOGE

def get_final_config(args):
    cfg = get_cfg_defaults()
    if (hasattr(args, 'config_file')) and os.path.exists(args.config_file):
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.config_file)
    orig_root_dir = cfg.OUTPUT_ROOT_DIR
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_ROOT_DIR, cfg.OUT_REL_DIR)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, 'w') as yaml_file:
        cfg.dump(stream = yaml_file, default_flow_style=False)

    print('USING FOLLOWING CONFIG:')
    print(cfg)
    print("Wrote config file at: {}".format(path))
    return cfg, output_dir, orig_root_dir

def save_best_ckpt_cfg(cfg, output_dir, best_ckpt_path):
    cfg.defrost()
    cfg.MODEL.CKPT_PATH = best_ckpt_path
    cfg.freeze()
    path = os.path.join(output_dir, "config_best.yaml")
    with open(path, 'w') as yaml_file:
        cfg.dump(stream = yaml_file, default_flow_style=False)

def find_ckpt(root_dir, ckpt_rel_path, find_best_ckpt = False):
    if ckpt_rel_path is not None and os.path.isfile(ckpt_rel_path):
        return ckpt_rel_path
    if ckpt_rel_path is not None and os.path.isfile(os.path.join(root_dir, ckpt_rel_path)):
        return os.path.join(root_dir, ckpt_rel_path)
    versions = os.path.join(root_dir, 'default')
    if not os.path.isdir(versions):
        return None
    for folder in sorted(os.listdir(versions)):
        ckpt_folder = os.path.join(versions, folder, 'checkpoints')
        if not find_best_ckpt:
            possible_path = os.path.join(ckpt_folder, 'last.ckpt')
            if os.path.isfile(possible_path):
                print(f'Found checkpoint: {possible_path}')
                return possible_path
        else:
            for ckpt in os.listdir(ckpt_folder):
                if ckpt.endswith('.ckpt') and 'epoch' in ckpt:
                    return os.path.join(ckpt_folder, ckpt)
    return None

def main(args):
    print(datetime.now().time())
    cfg, output_dir, orig_output_dir = get_final_config(args)   
    seed_everything(cfg.SEED)
    # gpus = 0
    # if cfg.DEVICE == 'gpu':
        # gpus = [0]
        # gpu_id = get_freer_gpu()
        # if gpu_id >= 0:
        #     print(f'Using GPU: {gpu_id}')
        #     gpus = [gpu_id]

    # wandb.tensorboard.patch(root_logdir = output_dir)
    # wandb.init(project=os.path.basename(output_dir), sync_tensorboard=True)
    tb_logger = TensorBoardLogger(output_dir, default_hp_metric=False, max_queue = 1000, flush_secs = 60)
    ckpt_path = None
    if cfg.MODEL.CKPT_PATH is not None or (args.eval_only and not args.only_test_non_learned):
        ckpt_path = find_ckpt(cfg.OUTPUT_ROOT_DIR, cfg.MODEL.CKPT_PATH, args.eval_best_ckpt)
        if ckpt_path is None:
            ckpt_path = find_ckpt(orig_output_dir, cfg.MODEL.CKPT_PATH, args.eval_best_ckpt)

    assert ckpt_path is None or os.path.isfile(ckpt_path), f'CKPT: {ckpt_path} not found.'
    checkpoint_callback = ModelCheckpoint(save_last = True, save_on_train_epoch_end = True, mode = 'max', save_top_k = 1, monitor = 'train_last_round_lb', verbose = True)
    patience_mult = 4 if cfg.TRAIN.USE_REPLAY_BUFFER else 2 # train for more iterations if using replay buffer since samples from trajectories are 'off-policy'.
    early_stopping = EarlyStopping('train_last_round_lb', 
                                patience = patience_mult * cfg.TRAIN.MAX_NUM_EPOCHS // cfg.TRAIN.NUM_JOURNEYS, 
                                check_on_train_epoch_end = True, 
                                mode = 'max')
    num_sanity_val_steps = 0
    # if args.test_non_learned:
    #     num_sanity_val_steps = -1
    trainer = Trainer(deterministic=False,  # due to https://github.com/pyg-team/pytorch_geometric/issues/3175#issuecomment-1047886622
                    accelerator  = 'gpu',
                    max_epochs = cfg.TRAIN.MAX_NUM_EPOCHS, 
                    default_root_dir=output_dir,
                    check_val_every_n_epoch = cfg.TEST.VAL_PERIOD,
                    logger = tb_logger, 
                    num_sanity_val_steps = num_sanity_val_steps, 
                    log_every_n_steps=cfg.LOG_EVERY,
                    gradient_clip_val=cfg.TRAIN.GRAD_CLIP_VAL,
                    callbacks=[checkpoint_callback, early_stopping],
                    detect_anomaly = False)

    combined_train_loader, val_loaders, val_datanames, test_loaders, test_datanames = get_ilp_gnn_loaders(cfg, 
                                                                                        skip_dual_solved = True, 
                                                                                        test_only = args.eval_only, 
                                                                                        test_precision_double = not args.test_precision_float,
                                                                                        test_on_train = args.test_on_train,
                                                                                        train_precision_double = args.train_precision_double)
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ['SLURM_JOB_ID']
        job_file_path = f'out_dual/slurm_new/{job_id}.out'
        if os.path.isfile(job_file_path):
            os.symlink(os.path.abspath(job_file_path), os.path.join(output_dir, f'{job_id}.out'))
    if ckpt_path is not None:
        print(f'Loading checkpoint and hyperparameters from: {ckpt_path}')
        model = DOGE.load_from_checkpoint(ckpt_path,
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            non_learned_updates_test = args.test_non_learned,
            only_test_non_learned = args.only_test_non_learned)
    else:
        print(f'Initializing from scratch.')
        model = DOGE.from_config(cfg, 
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            non_learned_updates_test = args.test_non_learned,
            only_test_non_learned = args.only_test_non_learned)
    if not args.eval_only:
        if args.train_precision_double:
            torch.set_default_dtype(torch.float64)
            model = model.to(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
            model = model.to(torch.float32)

        trainer.fit(model, combined_train_loader, val_loaders)
        save_best_ckpt_cfg(cfg, output_dir, checkpoint_callback.best_model_path)
        # Use the checkpoint with the best training performance for testing.
        model = DOGE.load_from_checkpoint(checkpoint_callback.best_model_path,
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            non_learned_updates_test = args.test_non_learned)
        combined_train_loader = None
        val_loaders = None
        if args.test_precision_float:
            model = model.to(torch.float32)
            torch.set_default_dtype(torch.float32)
        else:
            model = model.to(torch.float64)
            torch.set_default_dtype(torch.float64)
        model.eval()
        trainer.test(model, dataloaders = test_loaders)
    else:
        model.eval()
        if not args.test_precision_float:
            model = model.to(torch.float64)
            torch.set_default_dtype(torch.float64)
        trainer.test(model, dataloaders = test_loaders)
    print(datetime.now().time())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval-best-ckpt", action="store_true", help="perform evaluation on best ckpt instead of last")
    parser.add_argument("--test-non-learned", action="store_true", help="Runs FastDOG updates.")
    parser.add_argument("--only-test-non-learned", action="store_true", help="Only runs FastDOG updates.")
    parser.add_argument("--test-precision-float", action="store_true", help="Performs testing in FP32 format. Recommended to not set due to numerical issues in FP32.")
    parser.add_argument("--test-on-train", action="store_true", help="Performs testing on training data.")
    parser.add_argument('--train-precision-double', action="store_true", help="double precision training.")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. ",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:")
    print(args)
    main(args)