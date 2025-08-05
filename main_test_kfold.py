import glob
import time

import torch

from main.config import ex
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.strategies import DDPStrategy
import os
from main.datamodules import TestData
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules import Model, Model_Pre
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from lightning.pytorch.tuner import Tuner
from main.modules.get_mu_std import Mu_Std
from torch.distributed.elastic.multiprocessing.errors import record
from main.modules import LightningUNet

@record
@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config['seed'])
    print(_config)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    k = _config['kfold']
    # for k in range(0, _config['kfold']):
    version = f"version_{_config['kfold_test']}"
    # if k==0 or k==1: #resume
    #     continue
    rank_zero_info(f'Using k fold: now is {k}')
    exp_name = f'{_config["exp_name"]}'
    # model = Mu_Std(_config)

    logger_path = _config["log_dir"]
    rank_zero_info(f'logger_path: {logger_path}')
    os.makedirs(logger_path, exist_ok=True)
    name = f'{exp_name}_{_config["lr_policy"]}_{_config["model_arch"]}_{_config["loss_function"]}'
    if _config['extra_name'] is not None:
        name = f'{_config["extra_name"]}_{_config["lr_policy"]}_{_config["model_arch"]}_{_config["optim"]}'
    if _config['fft_only'] is True:
        name += '_fft_only'
    elif _config['time_only'] is True:
        name += '_time_only'
    elif _config['mode'] == 'pretrain':
        name += '_pretrain'
    else:
        name += '_' + _config['mode']
    if _config['all_time'] is not None:
        name += '_all_time'
    if _config['use_pooling'] is not None:
        name += _config['use_pooling']
    if _config['Use_FPN'] is not None:
        name += '_' + _config['Use_FPN']
    if _config['use_fpfn'] is not None:
        name += '_' + _config['use_fpfn']
    if _config['expert'] is not None:
        name += '_' + _config['expert']
    if _config['EDF_Mode'] is not None:
        name += '_' + _config['EDF_Mode']
    if _config['subset'] is not None:
        name += '_data_' + str(_config['subset'])
    ckpt_path = os.path.join(_config['output_dir'], f'{name}/fold_{k}/{version}')
    rank_zero_info(f'ckpt_path: {ckpt_path}')
    ckpt_path_list = glob.glob(ckpt_path + '/*')
    ckpt_path_list = sorted(ckpt_path_list, reverse=True)[:10]
    if len(ckpt_path_list) == 0:
        ckpt_path_list = [_config['load_path']]
    print(f'ckpt_path_list : {ckpt_path_list}')
    logger = pl.loggers.TensorBoardLogger(
        logger_path,
        name=name,
    )
    summary = ModelSummary(
        max_depth=-1)
    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval="step")
    callbacks = [lr_callback, summary]
    accum_iter = _config['accum_iter']
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    if _config['dist_on_itp'] is True:
        distributed_strategy = 'ddp'
    elif _config['deepspeed']:
        distributed_strategy = 'deepspeed'
    else:
        distributed_strategy = None
    if distributed_strategy is None:
        trainer = pl.Trainer(
            profiler="simple",
            devices=_config["num_gpus"],
            precision=_config["precision"],
            accelerator=_config["device"],
            strategy="auto",
            deterministic=True,
            # benchmark=True,
            max_epochs=_config["max_epoch"],
            max_steps=max_steps,
            # callbacks=callbacks,
            logger=logger,
            accumulate_grad_batches=accum_iter,
            log_every_n_steps=1,
            val_check_interval=_config["val_check_interval"],
            limit_val_batches=_config['limit_val_batches']
        )
    else:
        trainer = pl.Trainer(
            num_nodes=_config["num_nodes"],
            devices=_config["num_gpus"],
            profiler="simple",
            precision=_config["precision"],
            accelerator=_config["device"],
            strategy="auto",
            deterministic=True,
            # benchmark=True,
            max_epochs=_config["max_epoch"],
            max_steps=max_steps,
            callbacks=callbacks,
            logger=logger,
            limit_train_batches=_config['limit_train_batches'],
            accumulate_grad_batches=accum_iter,
            log_every_n_steps=1,
            val_check_interval=_config["val_check_interval"],
            limit_val_batches=_config['limit_val_batches']
        )
    import numpy as np
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # trainer.validate(model, datamodule=dm)
    # test_dm = dm.dms[0].test_dataset
    # ckpt_path_list = ['/home/cuizaixu_lab/huangweixuan/data/checkpoint/Finetune_edf_cosine_backbone_large_patch200_Lion_finetune_all_timeswin_2018_Finetune/6_fold/version_0/ModelCheckpoint-epoch=37-val_acc=0.8260-val_score=5.3318.ckpt',
    #                   '/home/cuizaixu_lab/huangweixuan/data/checkpoint/Finetune_edf_cosine_backbone_large_patch200_Lion_finetune_all_timeswin_2018_Finetune/6_fold/version_0/ModelCheckpoint-epoch=28-val_acc=0.8290-val_score=5.3443.ckpt']
    # ckpt_path_list = ['/home/cuizaixu_lab/huangweixuan/data/checkpoint/Finetune_edf_cosine_backbone_large_patch200_Lion_finetune_all_timeswin_2018_Finetune/7_fold/version_0/ModelCheckpoint-epoch=39-val_acc=0.8810-val_score=5.5265.ckpt',
    #                   '/home/cuizaixu_lab/huangweixuan/data/checkpoint/Finetune_edf_cosine_backbone_large_patch200_Lion_finetune_all_timeswin_2018_Finetune/7_fold/version_0/ModelCheckpoint-epoch=30-val_acc=0.8840-val_score=5.5266.ckpt']
    # ckpt_path_list = [os.path.join(ckpt_path, 'ModelCheckpoint-epoch=21-val_acc=0.8860-val_score=5.5298.ckpt'),
                      # os.path.join(ckpt_path, 'ModelCheckpoint-epoch=30-val_acc=0.8850-val_score=5.5256.ckpt'),
                      # os.path.join(ckpt_path, 'ModelCheckpoint-epoch=15-val_acc=0.8900-val_score=5.5179.ckpt')
                      # ]
    for ckpt in ckpt_path_list:
        if _config['mode'] == 'pretrain':
            if 'simple_conv' in _config['model_arch']:
                model = LightningUNet(mode='pretrain')
            else:
                model = Model_Pre(_config)
        else:
            if 'simple_conv' in _config['model_arch']:
                model = LightningUNet(mode='finetune', num_classes=_config['num_classes'])
            else:
                model = Model(_config, num_classes=_config['num_classes'])

        dm = MultiDataModule(_config, kfold=k)
        rank_zero_info(f'Using k fold: now is {k}, now_ckpt_path: {ckpt}')
        trainer.test(model, datamodule=dm, ckpt_path=ckpt)
