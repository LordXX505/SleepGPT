import sys
import time

import torch

from main.config import ex
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
import os
import numpy as np

from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules import Model, Model_Pre
import signal
from lightning.pytorch.tuner import Tuner
from main.modules.get_mu_std import Mu_Std
from torch.distributed.elastic.multiprocessing.errors import record

@record
@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    rank_zero_info(f"seed: {_config['random_seed']}")
    for random_seed in _config['random_seed']:
        pl.seed_everything(random_seed)
        print(_config)
        torch.multiprocessing.set_sharing_strategy('file_system')
        if _config['resume_during_training'] is not None:
            start_idx = int(_config['resume_during_training'])
        else:
            start_idx = 0
        if _config['kfold'] == None:
            end_idx = 1
        else:
            end_idx = _config['kfold']
        for k in range(start_idx, end_idx):
            rank_zero_info(f'Using k fold: now is {k}')
            exp_name = f'{_config["exp_name"]}'
            # model = Mu_Std(_config)
            if _config['mode'] == 'pretrain':
                model = Model_Pre(_config)
            else:
                model = Model(_config, fold_now=k, num_classes=_config['num_classes'])
            if _config['mode'] == 'Spindledetection':
                data_dir_temp = _config['data_dir'][0]
                mass_aug_times = _config['mass_aug_times']
                data_dir_temp_list = data_dir_temp.split('/')
                orig_names = data_dir_temp_list[-2].split('_')
                orig_names[3] = str(mass_aug_times)
                data_dir_temp_list[-2] = '_'.join(orig_names)
                _config['data_dir'] = ['/'.join(data_dir_temp_list)]
                rank_zero_info(f"Modified data dir: {_config['data_dir']}")
            dm = MultiDataModule(_config, kfold=k)
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
            if _config["eval"]:
                name += '_eval'
            if _config['EDF_Mode'] is not None:
                name += '_' + _config['EDF_Mode']
            if _config['subset'] is not None:
                name += '_data_' + str(_config['subset'])
            logger_path = os.path.join(_config["log_dir"], name)
            rank_zero_info(f'logger_path: {logger_path}')
            os.makedirs(logger_path, exist_ok=True)
            logger = pl.loggers.TensorBoardLogger(
                logger_path,
                name=f'fold_{k}',
            )
            if _config['mode'] == 'pretrain':
                monitor = 'validation/the_metric'
            elif _config['mode'] == 'Spindledetection':
                monitor = 'FpFn/validation/F1'
            elif 'downstream' in _config['mode']:
                monitor = "CrossEntropy/test/score"
            elif 'cap' in _config['mode']:
                monitor = "Pathology/validation/max_accuracy_epoch"
            else:
                monitor = "CrossEntropy/validation/max_accuracy_epoch"
            rank_zero_info(f'monitor: {monitor}')
            if _config['loss_names']['Spindle'] > 0 or _config['loss_names']['Apnea'] > 0:
                filename = 'ModelCheckpoint-epoch={epoch:02d}-val_acc={FpFn/validation/F1:.4f}-val_score={' \
                           'validation/the_metric:.4f}'
            elif _config['loss_names']['CrossEntropy'] > 0:
                filename = 'ModelCheckpoint-epoch={epoch:02d}-val_acc={' \
                           'CrossEntropy/validation/max_accuracy_epoch:.4f}' \
                           '-val_macro={CrossEntropy/validation/tf/macro_f1:.4f}' \
                           '-val_score={validation/the_metric:.4f}'
            elif _config['loss_names']['Pathology'] > 0:
                filename = 'ModelCheckpoint-epoch={epoch:02d}-val_acc={' \
                           'Pathology/validation/max_accuracy_epoch:.4f}' \
                           '-val_macro={Pathology/validation/macro_f1:.4f}' \
                           '-val_score={validation/the_metric:.4f}'
            else:
                filename = None
            print(f'filename: {filename}')
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                # dirpath=f'/home/cuizaixu_lab/huangweixuan/data/checkpoint/{name}/{k}_fold/version_{logger.version}',
                dirpath=f"{os.path.join(_config['output_dir'])}/{name}/fold_{k}/version_{logger.version}",
                filename=filename,
                save_top_k=int(_config['save_top_k']),
                verbose=True,
                monitor=monitor,
                # monitor="CrossEntropy/validation/max_accuracy_epoch",
                mode="min" if 'score' in monitor else 'max',
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
                    max_epochs=_config["max_epoch"],
                    max_steps=max_steps,
                    # callbacks=callbacks,
                    logger=logger,
                    accumulate_grad_batches=accum_iter,
                    log_every_n_steps=1,
                    val_check_interval=_config["val_check_interval"],
                    limit_val_batches=_config['limit_val_batches'],
                    gradient_clip_val=_config['gradient_clip_val']
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
                    limit_val_batches=_config['limit_val_batches'],
                    gradient_clip_val=_config['gradient_clip_val']
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
            if _config["all_time"] is True and _config['grad_name'].startswith('partial'):
                if _config['get_param_method'] == 'layer_decay' or _config['get_param_method'] == 'no_layer_decay':
                    for param in model.parameters():
                        param.requires_grad = False

                    grad_name = ["fc_norm", "transformer.norm", "pooler",
                                 "decoder_transformer_block", "stage_pred", "spindle_pred_proj"]
                    if _config['grad_name'].startswith('partial'):
                        for _ in range(int(_config['grad_name'].split('_')[-1]), 12):
                            grad_name.append(f"transformer.blocks.{_}")
                    if _config['use_pooling'] == 'cls':
                        grad_name.append("cls_token")
                    if _config['use_relative_pos_emb']:
                        grad_name.append("relative_position_bias_table")
                    for name, param in model.named_parameters():
                        for key in grad_name:
                            if key in name and "pe" not in name:
                                param.requires_grad = True
                for name, param in model.named_parameters():
                    rank_zero_info("{}\t{}\t{}".format(name, param.requires_grad, param.shape))
            if not _config["eval"]:
                flag = True
                if _config['resume_during_training'] is not None and k == int(_config['resume_during_training']):
                    rank_zero_info(f'k == {k}, resuming checkpoints, loaded from : {_config["resume_ckpt_path"]}')
                    if _config["resume_ckpt_path"] != "":
                        flag = False
                        trainer.fit(model, datamodule=dm,
                                    ckpt_path=_config['resume_ckpt_path'])
                if flag is True:
                    rank_zero_info(f'k == {k}, no resuming checkpoints')
                    trainer.fit(model, datamodule=dm,)
            else:
                if _config['mode'] == 'visualization_mask_ratio_dynamic':
                    for mask_ratio in [0.45, 0.6, 0.75, 0.9]:
                        model.transformer.mask_ratio[0] = mask_ratio
                        _config['mask_ratio'] = [mask_ratio]
                        dm = MultiDataModule(_config, kfold=k)
                        rank_zero_info(f'mask_ratio: {mask_ratio}')
                        trainer.test(model, datamodule=dm)
                else:
                    trainer.test(model, datamodule=dm)

