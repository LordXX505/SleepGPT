import sys
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
from main.modules import Model, Model_Pre, LightningUNet
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from lightning.pytorch.tuner import Tuner
from main.modules.get_mu_std import Mu_Std
from torch.distributed.elastic.multiprocessing.errors import record
import glob
@record
@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config['seed'])
    print(_config)
    torch.multiprocessing.set_sharing_strategy('file_system')
    exp_name = f'{_config["exp_name"]}'
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
    dm = MultiDataModule(_config)
    os.makedirs(_config["log_dir"], exist_ok=True)
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
        if _config['use_pooling'] is not None:
            name += '_all_time_' + _config['use_pooling']
        else:
            name += '_all_time'

    rank_zero_info(f'name: {name}')
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=name,
    )
    monitor = 'validation/the_metric' if _config['mode'] == 'pretrain' else "CrossEntropy/validation/max_accuracy_epoch"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{os.path.join(_config['output_dir'])}/{name}/version_{logger.version}",
        filename='ModelCheckpoint-epoch={epoch:02d}-val_acc={CrossEntropy/validation/max_accuracy_epoch:.4f}-val_score={validation/the_metric:.4f}',
        save_top_k=5,
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
    if _config['kfold_load_path'] is not None:
        version = f"version_{_config['kfold_test']}"

        ckpt_path = os.path.join(_config['kfold_load_path'], f'{name}/{version}')
        rank_zero_info(f'ckpt_path: {ckpt_path}')
        ckpt_path_list = glob.glob(ckpt_path + '/*')
        rank_zero_info(f'using ckpt_path_list: {ckpt_path_list}')
    else:
        ckpt_path_list = [_config['ckpt']]
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
    if _config["all_time"] is True:
        # if _config['get_param_method'] == 'layer_decay':
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     grad_name = ["transformer.blocks.10", "transformer.blocks.11", "fc_norm",
        #                  "transformer.norm", "pooler", "decoder_transformer_block", "stage_pred"]
        #     if _config['use_pooling'] == 'cls':
        #         grad_name.append("cls_token")
        #     if _config['use_relative_pos_emb']:
        #         grad_name.append("relative_position_bias_table")
        #     for name, param in model.named_parameters():
        #         for key in grad_name:
        #             if key in name and "pos_encoding.pe" not in name:
        #                 param.requires_grad = True

        for name, param in model.named_parameters():
            rank_zero_info("{}\t{}\t{}".format(name, param.requires_grad, param.shape))
    # print(trainer.log_dir)
    # # # Create a Tuner
    # tuner = Tuner(trainer)
    # # #
    # # # # Run learning rate finder
    # lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-8, max_lr=1e-2)
    # # #
    # # # # Results can be found in
    # print(lr_finder.results)
    # # #
    # # # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # import matplotlib.pyplot as plt
    # plt.savefig(os.path.join(_config["log_dir"], f'{exp_name}_{_config["lr_policy"]}_{_config["model_arch"]}_{_config["loss_function"]}/lr_find.png')
    #             )
    # # #
    # # # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(new_lr)
    # update hparams of the model
    if not _config["eval"]:
        trainer.fit(model, datamodule=dm,)
        # trainer.fit(model, datamodule=dm,
        #             ckpt_path='/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_2/last.ckpt')
    else:
        import numpy as np
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # trainer.validate(model, datamodule=dm)
        # test_dm = dm.dms[0].test_dataset
        for ckpt in ckpt_path_list:
            rank_zero_info(f'now_ckpt_path: {ckpt}')
            trainer.test(model, datamodule=dm, ckpt_path=ckpt)
        # res = {}
        # for _, index in enumerate(model.res_index):
        #     index = index[:, 0].view(-1)
        #     pred = model.res_feats[_]
        #     label = model.res_label[_]
        #     for j, i in enumerate(index):
        #         name, idx = os.path.split(test_dm.get_name(i.cpu().numpy()))
        #         idx = int(idx.split('.')[0])
        #         if name not in res:
        #             res[name] = {
        #                 'idx': [],
        #                 'pred': [],
        #                 'label': []
        #             }
        #         res[name]['idx'].append(idx)
        #         res[name]['pred'].append(pred[j])
        #         res[name]['label'].append(label[j])
        #
        # for name in res.keys():
        #     arg = np.argsort(res[name]['idx'])
        #     res[name]['idx'] = np.array(res[name]['idx'])[arg]
        #     res[name]['pred'] = torch.stack(res[name]['pred'])[arg]
        #     res[name]['label'] = torch.stack(res[name]['label'])[arg]
        # sort_val_acc = []
        # for name in res.keys():
        #     total_acc = (res[name]['pred'] == res[name]['label']).sum() / res[name]['pred'].numel()
        #     sort_val_acc.append(np.array([total_acc.cpu().numpy(), name]))
        # print(sort_val_acc)
        # np.savez(os.path.join('/data/data/', 'val_acc'), name_acc=sort_val_acc)
        #
        #
