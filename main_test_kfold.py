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


@record
@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config['seed'])
    print(_config)
    version = None
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    k = _config['kfold_test']
    # if k==0 or k==1: #resume
    #     continue
    rank_zero_info(f'Using k fold: now is {k}')
    exp_name = f'{_config["exp_name"]}'
    # model = Mu_Std(_config)
    if _config['mode'] == 'pretrain':
        model = Model_Pre(_config)
    else:
        model = Model(_config)
    dm = MultiDataModule(_config, kfold=k)
    logger_path = _config["log_dir"]
    rank_zero_info(f'logger_path: {logger_path}')
    os.makedirs(logger_path, exist_ok=True)
    name = f'{exp_name}_{_config["lr_policy"]}_{_config["model_arch"]}_{_config["loss_function"]}'
    if _config['extra_name'] is not None:
        name = f'{_config["extra_name"]}_{_config["lr_policy"]}_{_config["model_arch"]}_{_config["loss_function"]}'
    if _config['fft_only'] is True:
        name += '_fft_only'
    elif _config['time_only'] is True:
        name += '_time_only'
    elif _config['mode'] == 'pretrain':
        name += '_pretrain'
    else:
        name += '_' + _config['mode']
    if _config['all_time'] is not None:
        name += '_all_time_'
    if _config['use_pooling'] is not None:
        name += _config['use_pooling']
    if _config["eval"]:
        name += 'eval'

    logger = pl.loggers.TensorBoardLogger(
        logger_path,
        name=name,
    )
    monitor = 'validation/the_metric' if _config['mode'] == 'pretrain' else "CrossEntropy/validation/max_accuracy_epoch"
    if _config['loss_names']['FpFn'] > 0:
        filename = 'ModelCheckpoint-epoch={epoch:02d}-val_acc={FpFN/validation/F1:.4f}-val_score={' \
                   'validation/the_metric:.4f}'
    elif _config['loss_names']['CrossEntropy'] > 0:
        filename = 'ModelCheckpoint-epoch={epoch:02d}-val_acc={' \
                   'CrossEntropy/validation/max_accuracy_epoch:.4f}-val_score={validation/the_metric:.4f}'
    else:
        filename = None
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'/data/checkpoint/{k}_fold/{name}/version_{logger.version}',
        filename=filename,
        save_top_k=20,
        verbose=True,
        monitor=monitor,
        # monitor="CrossEntropy/validation/max_accuracy_epoch",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False
    )
    summary = ModelSummary(
        max_depth=-1)
    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback, summary]
    accum_iter = _config['accum_iter']
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    if _config['dist_on_itp']:
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
    trainer.test(model, datamodule=dm)
