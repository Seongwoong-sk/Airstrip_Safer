import sys
sys.path.append('/content/drive/MyDrive/Air_PY')
from env import dataset

import os
import gc
import cv2
import sys
import time
import copy
import mmcv
import torch
import random
import argparse
import numpy as np
from glob import glob
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv import collect_env

import mmdet
from mmdet.utils import (get_device,update_data_root,setup_multi_processes)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector 
from mmdet.apis import init_detector, inference_detector, set_random_seed,train_detector 


import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)
from mmcv.runner.hooks.lr_updater import CosineRestartLrUpdaterHook
from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)
import wandb



# Argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work_dir', help='Path to save checkpoint')
    parser.add_argument('--max_epochs', type = int,help='Max epochs when training')
    parser.add_argument('--checkpoint', help='the file to load vanila model weights')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Decide whether to evaluate the dataset during training')
    parser.add_argument(
        '--auto_scale_lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.
    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    '''
    Build Dataset이 수행하는 것
    0. Dataset을 위한 Config 설정 (data_root, ann_file, img_prefix)
    1. CustomDataset 객체를 MMDetection Framework에 등록
    2. Config에 설정된 주요 값으로 CustomDataset 객체 생성
    '''
    dataset = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), 
            test_cfg=cfg.get('test_cfg'))
    
    meta = dict()
    meta['exp_name'] = osp.basename(args.config)

    distributed = False
    validate = args.validate
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)
    update_data_root(cfg)
    setup_multi_processes(cfg)
    seed_everything(args.seed)
    set_random_seed(args.seed)

    # Create work_dir
    cfg.work_dir = args.work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Setting max epochs
    cfg.runner.max_epochs = args.max_epochs

    # Applying augmentation to Train Pipeline  args.aug_pipeline_list #

    aug_four = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='AutoAugment',
            policies=[[{
                'type':
                'Resize',
                'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                'multiscale_mode':
                'value',
                'keep_ratio':
                True
            }],
                    [{
                        'type': 'Resize',
                        'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
                        'multiscale_mode': 'value',
                        'keep_ratio': True
                    }, {
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (384, 600),
                        'allow_negative_crop': True
                    }, {
                        'type':
                        'Resize',
                        'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                                        (576, 1333), (608, 1333), (640, 1333),
                                        (672, 1333), (704, 1333), (736, 1333),
                                        (768, 1333), (800, 1333)],
                        'multiscale_mode':
                        'value',
                        'override':
                        True,
                        'keep_ratio':
                        True
                    },
                    {
                        'type' : 'Rotate',
                        'prob' : 0.25,
                        'level' : 0.2,
                        'max_rotate_angle' : 15

                    },
                    {
                        'type' : 'BrightnessTransform',
                        'level': 0.2,
                        'prob' : 0.25

                    },
                    ]]),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='MixUp', flip_ratio=0.2),
        dict(type='Mosaic', prob=0.25),
        dict(type='Pad', size_divisor=1),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    cfg.train_pipeline = aug_four # AUG Selection
    cfg.data.train.pipeline = cfg.train_pipeline
    print(f"✨[Info msg] Applied Aug --> {cfg.train_pipeline}")
    # ######################################################################


# prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # ✨ Auto Scale LR ✨
    # args.auto_scale_lr = False
    # if args.auto_scale_lr:
    #     auto_scale_lr(cfg, distributed, logger)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get('momentum_config', None),
            custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
    print(f"✨[Info msg] validate : {args.validate}")
    if validate:
        val_dataloader_default_args = dict(
                samples_per_gpu=1,
                workers_per_gpu=2,
                dist=distributed,
                shuffle=False,
                persistent_workers=False)

        val_dataloader_args = {
                **val_dataloader_default_args,
                **cfg.data.get('val_dataloader', {})
            }
            # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:

                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    if args.resume_from :
        print(f"✨[Info msg] load_from : {args.checkpoint}")
        print(f"✨[Info msg] resume : {args.resume_from}")
    # cfg.load_from = args.checkpoint
    # cfg.resume_from = args.resume_from

    print(f"✨[Info msg] cfg.load_from : {cfg.load_from}")
    print(f"✨[Info msg] cfg.resume_from : {cfg.resume_from}")
    
    # resume_from = args.resume_from
    # if cfg.resume_from is None and cfg.get('auto_resume'):
    #         resume_from = find_latest_checkpoint(cfg.work_dir)
    # if cfg.resume_from is not None:
    #         cfg.resume_from = resume_from

    if cfg.resume_from and cfg.load_from is None:
            print(f"✨ [Info msg] Resume Training will be starting")
            runner.resume(cfg.resume_from)
    elif cfg.load_from and cfg.resume_from is None:
            print(f"✨ [Info msg] Vanila Training will be starting")
            runner.load_checkpoint(cfg.load_from)

    # print(cfg.pretty_text)
    
    # errored: RuntimeError('CUDA out of memory 방지)
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✨[Info msg] cfg.lr_config : {cfg.lr_config}")

    runner.run(data_loaders, cfg.workflow)

if __name__ == '__main__':
    main()