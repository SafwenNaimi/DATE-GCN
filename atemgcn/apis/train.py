# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv.engine import multi_gpu_test
from mmcv.engine import single_gpu_test
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, get_dist_info
from mmcv.runner import Runner
#from ..core import DistEvalHook
from ..core import DistEvalHook
from ..datasets import build_dataloader, build_dataset
from ..utils import cache_checkpoint, get_root_logger

torch.autograd.set_detect_anomaly(True)

def squeeze_input(batch):
    if isinstance(batch, dict):
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() == 5:
                batch[key] = value.squeeze(0)
    elif isinstance(batch, (tuple, list)):
        batch = [item.squeeze(0) if torch.is_tensor(item) and item.dim() == 5 else item for item in batch]
    elif torch.is_tensor(batch) and batch.dim() == 5:
        batch = batch.squeeze(0)
    return batch


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.get('log_level', 'INFO'))

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    for i, data_loader in enumerate(data_loaders):
        batch = next(iter(data_loader))
        if isinstance(batch, dict):
            for key, value in batch.items():
                if torch.is_tensor(value):
                    logger.info(f"Data loader {i}, Input '{key}' shape: {value.shape}")
        elif isinstance(batch, (tuple, list)):
            for j, item in enumerate(batch):
                if torch.is_tensor(item):
                    logger.info(f"Data loader {i}, Input {j} shape: {item.shape}")
        else:
            logger.info(f"Data loader {i}, Input shape: {batch.shape}")


    # Move the model to GPU
    device = torch.cuda.current_device()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Total params", total_params)
    print("Trainable parameters:", trainable_params)
    
    ##logger.info(f'Total parameters: {total_params:,}')
    ##logger.info(f'Trainable parameters: {trainable_params:,}')
    ##logger.info(f'Non-trainable parameters: {total_params - trainable_params:,}')

    # Build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # Build runner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # Set timestamp
    runner.timestamp = timestamp

    if 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # Register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    eval_hook = None
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        cfg.load_from = cache_checkpoint(cfg.load_from)
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            assert eval_hook is not None
            best_ckpt_path = None
            ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
            ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                epoch_ids = [
                    int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
                ]
                best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
            else:
                best_ckpt_path = ckpt_paths[0]
            if best_ckpt_path:
                best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        tmpdir = cfg.get('evaluation', {}).get('tmpdir', osp.join(cfg.work_dir, 'tmp'))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))

        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []

        if test['test_last']:
            names.append('last')
            ckpts.append(None)
        if test['test_best']:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                runner.load_checkpoint(ckpt)

            outputs = single_gpu_test(runner.model, test_dataloader, tmpdir)
            if name == 'last':
                pred_filename = f'{name}_pred.pkl'
            else:
                pred_filename = f'{name}_pred.pth'

            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(cfg.work_dir, pred_filename)
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start',
                        'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    logger.info(f'{metric_name}: {val:.04f}')
