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

def get_idx2name(_config, k):
    dm = MultiDataModule(_config, kfold=k)
    idx_2_name = None
    dm.setup(stage='test')
    for dst in dm.dms:
        idx_2_name = dst.test_dataset.idx_2_name
    print(f'idx_2_name: {idx_2_name}')
    return idx_2_name

@record
@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config['seed'])
    print(_config)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    if _config['resume_during_training'] is not None:
        start_idx = int(_config['resume_during_training'])
    else:
        start_idx = 0
    if _config['kfold'] is None:
        end_loop = 1
    else:
        end_loop = _config['kfold']
    for k in range(start_idx, end_loop):

        # for k in range(0, _config['kfold']):
        version = _config['kfold_test']

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
        if version is not None:
            rank_zero_info(f'version: {version}')
            if isinstance(version, int):
                ckpt_path = os.path.join(_config['output_dir'], f'{name}/fold_{k}/version_{version}')
            else:
                ckpt_path = os.path.join(_config['output_dir'], f'{name}/fold_{k}/version_{version[k]}')
            ckpt_path = os.path.join(ckpt_path, 'last.ckpt')
            rank_zero_info(f'ckpt_path: {ckpt_path}')

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

        model = Model(_config)
        if _config['kfold'] is not None:
            dm = MultiDataModule(_config, kfold=k)
        else:
            dm = MultiDataModule(_config)
        real_ckpt = None
        if _config['load_path'] is not None and _config['load_path'] != "":
            real_ckpt = _config['load_path']
            rank_zero_info(f"Using k fold: now is {k}, now_ckpt_path: {_config['load_path']}")
        else:
            real_ckpt = ckpt_path
            rank_zero_info(f"Using k fold: now is {k}, now_ckpt_path: {ckpt_path}")
        if _config['persub'] is not None:
            idx_2_name = get_idx2name(_config, k=k)
            model = Model(_config, persub=True, test_sub_names=idx_2_name, _ckpt=real_ckpt,
                          persub_mode=_config['persub'], fold_now=k)
        else:
            model = Model(_config, _ckpt=real_ckpt, fold_now=k)
        trainer.test(model, datamodule=dm, ckpt_path=real_ckpt)
