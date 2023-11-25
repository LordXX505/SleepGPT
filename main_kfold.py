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
from main.modules import Test
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
    version=None
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    for k in range(_config['kfold']):
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
        if _config['all_time']:
            name += '_all_time_'
        if  _config['use_pooling']:
            name += _config['use_pooling']
        if _config["eval"]:
            name += 'eval'

        logger = pl.loggers.TensorBoardLogger(
            logger_path,
            name=name,
        )
        if _config['mode'] == 'pretrain':
            monitor = 'validation/the_metric'
        elif _config['mode'] == 'Spindledetection':
            monitor = 'FpFn/validation/F1'
        else:
            "CrossEntropy/validation/max_accuracy_epoch"

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f'/home/cuizaixu_lab/huangweixuan/data/checkpoint/{k}_fold/{name}/version_{logger.version}',
            filename='ModelCheckpoint-epoch={epoch:02d}-val_acc={CrossEntropy/validation/max_accuracy_epoch:.4f}-val_score={validation/the_metric:.4f}',
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
                strategy=distributed_strategy,
                deterministic=True,
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

        if _config["fft_only"] is True:
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                for key in ["fft_proj", "fft_cls_token", "norm2_fft", "mlp_fft", "gamma_2", "token_type_embeddings",
                            "itc_freq_weak_proj", "itc_freq_strong_proj", "logit_scale", "transformer.norm"]:
                    if key in name:
                        param.requires_grad = True

            for name, param in model.named_parameters():
                rank_zero_info("{}\t{}".format(name, param.requires_grad))
        if _config["all_time"] is True:
            if _config['get_param_method'] == 'layer_decay' or _config['get_param_method'] == 'no_layer_decay':
                for param in model.parameters():
                    param.requires_grad = False
                grad_name = ["fc_norm","transformer.blocks.10", "transformer.blocks.11",
                             "transformer.norm", "pooler", "decoder_transformer_block", "stage_pred", "spindle_pred_proj"]
                if _config['use_pooling'] == 'cls':
                    grad_name.append("cls_token")
                if _config['use_relative_pos_emb']:
                    grad_name.append("relative_position_bias_table")
                for name, param in model.named_parameters():
                    for key in grad_name:
                        if key in name and "pos_encoding.pe" not in name:
                            param.requires_grad = True

            for name, param in model.named_parameters():
                rank_zero_info("{}\t{}\t{}".format(name, param.requires_grad, param.shape))
        if not _config["eval"]:
            trainer.fit(model, datamodule=dm,)
            # trainer.fit(model, datamodule=dm,
            #             ckpt_path='/data/checkpoint/1_fold/Finetune_phy_cosine_backbone_large_patch200_l1_finetune_all_time_swin/version_1/last.ckpt')
        else:
            import numpy as np
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # trainer.validate(model, datamodule=dm)
            # test_dm = dm.dms[0].test_dataset
            trainer.test(model, datamodule=dm)

