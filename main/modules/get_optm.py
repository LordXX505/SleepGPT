from functools import partial

import torch
from torch.optim import lr_scheduler, AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch.distributed as dist
from typing import Union, Optional
import math
from torch.optim.lr_scheduler import LambdaLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from main.utils import Lion
from collections import Counter
from bisect import bisect_right

from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


def get_scheduler(optimizer, opt, warmup_steps, max_steps, end_lr, decay_power, Lambda):
    print('opt.lr_policy = [{}]'.format(opt['lr_policy']))
    if opt['lr_policy'] == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt['lr_policy'] == 'step':
        if opt.lr_decay_iters is None:
            opt.lr_decay_iters = 25
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt['lr_policy'] == 'step2':
        if opt.lr_decay_iters is None:
            opt.lr_decay_iters = 25
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt['lr_policy'] == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.001, patience=10)
    elif opt['lr_policy'] == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.001, patience=5)
    elif opt['lr_policy'] == 'step_warmstart':
        def lambda_rule(epoch):
            if epoch < warmup_steps:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt['lr_policy'] == 'step_warmstart2':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < warmup_steps:
                lr_l = 0.1
            elif warmup_steps <= epoch < 10000:
                lr_l = 0.002/optimizer.defaults["lr"]
            elif 10000 <= epoch < 30000:
                lr_l = 0.0008/optimizer.defaults["lr"]
            elif 30000 <= epoch < 40000:
                lr_l = 0.0001/optimizer.defaults["lr"]
            elif 40000 <= epoch <= 45000:
                lr_l = 0.00001/optimizer.defaults["lr"]
            elif 45000 <= epoch:
                lr_l = 0.000001/optimizer.defaults["lr"]
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        print("get_polynomial_decay_schedule_with_warmup")
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=int(decay_power),
        )
    return scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0,
                 t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

def adjust_learning_rate(optimizer, min_lr, num_warmup_epochs: int, num_training_steps: int, last_epoch: int = -1):
    """Decay the learning rate with half-cycle cosine after warmup"""

    lr_init = optimizer.defaults["lr"]

    lr_lambda = partial(
        _get_my_cosine_decay_schedule_with_warmup_lr_labmda,
        num_warmup_epochs=num_warmup_epochs,
        num_training_epochs=num_training_steps,
        param_groups=optimizer.param_groups,
        lr_init=lr_init,
        min_lr=min_lr
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_my_cosine_decay_schedule_with_warmup_lr_labmda(epoch, param_groups, num_warmup_epochs, num_training_epochs, min_lr, lr_init):
    if epoch < num_warmup_epochs:
        lr = lr_init * epoch / num_warmup_epochs
    else:
        lr = min_lr + (lr_init - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - num_warmup_epochs) / (num_training_epochs - num_warmup_epochs)))

    for param_group in param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr


def param_groups_lrd(pl_module, lr=1, weight_decay=0.05, no_weight_decay_list=[], layer_decay=1.0):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    if len(no_weight_decay_list) == 0:
        no_weight_decay_list = [
        "cls_token",
        "pos_embed",
        "fft_cls_token",
        "cls_token_pos_embed",
        "channel_embed",
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
        "fc_norm",
        "token_type_embeddings"
    ]

    num_layers = len(pl_module.transformer.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in pl_module.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or any(nn in n for nn in no_weight_decay_list) or n.endswith(".bias"):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "name": group_name,
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": lr,
            }
            param_groups[group_name] = {
                "name": group_name,
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": lr,
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    import json
    rank_zero_info("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def param_groups_no_layer_decay(pl_module, lr=1, weight_decay=0.05, no_weight_decay_list=[]):
    if not no_weight_decay_list:
        no_weight_decay_list = [
            "transformer.cls_token",
            "transformer.pos_embed",
            "transformer.fft_cls_token",
            "transformer.cls_token_pos_embed",
            "transformer.channel_embed",
            "token_type_embeddings",
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
            "norm2_time",
            "norm2_fft",
            "token_type_embeddings"
        ]
    head_names = ["spindle_pred_proj", "stage_pred", "pooler", "decoder_transformer_block"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    num_layers = len(pl_module.transformer.blocks) + 1
    param_group_names = {}
    param_groups = {}
    for n, p in pl_module.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or any(nn in n for nn in no_weight_decay_list) or n.endswith(".bias"):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        this_scale = 1.0
        if any(nn in n for nn in head_names):
            this_scale = lr_mult
        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "name": group_name,
                "lr_scale":1 ,
                "weight_decay": this_decay,
                "params": [],
                "lr": lr*this_scale,
            }
            param_groups[group_name] = {
                "name": group_name,
                "lr_scale": 1,
                "weight_decay": this_decay,
                "params": [],
                "lr": lr*this_scale,
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    import json
    rank_zero_info("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def get_layer_id_for_vit(name: str, num_layers: int):
    if name.startswith("transformer"):
        name = ".".join(name.split(".")[1:])

    if name in {'cls_token', 'pos_embed', 'fft_cls_token',
                'cls_token_pos_embed', 'mask_token', 'channel_embed'}:
        return 0
    elif name.startswith(('patch_embed', 'token_type_embeddings')):
        return 0

    if 'cross_attn' in name or 'xattn' in name \
       or 'spo2_extractor' in name or 'spo2_encoder' in name:
        return num_layers

    # ---- ③ 普通 Blocks / Norm ----
    if name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('norm'):
        return num_layers - 1      # 顶层 LayerNorm
    else:
        return num_layers

def set_schedule(pl_module):
    lr = pl_module.hparams.config["lr"]
    wd = pl_module.hparams.config["weight_decay"]
    Lambda = pl_module.hparams.config['Lambda']
    if pl_module.hparams.config['dist_on_itp']:
        eff_batch_size = pl_module.hparams.config["batch_size"] * \
                         pl_module.hparams.config["accum_iter"] * dist.get_world_size()
    else:
        eff_batch_size = pl_module.hparams.config["batch_size"]
    if lr is None:  # only base_lr is specified
        lr = pl_module.hparams.config["blr"] * eff_batch_size / 256
        rank_zero_info("base lr: %.2e" % (lr * 256 / eff_batch_size))
    rank_zero_info("Initial lr: %.2e Initial wd: %.2e"%(lr, wd))

    lr = lr/Lambda
    wd = wd*Lambda
    rank_zero_info("actual lr: %.2e Lambda%.2e" % (lr, Lambda))
    rank_zero_info("actual wd: %.2e" % wd)
    rank_zero_info("accumulate grad iterations: %d" % pl_module.hparams.config["accum_iter"])
    rank_zero_info("effective batch size: %d" % eff_batch_size)
    no_decay = [
        "transformer.cls_token",
        "transformer.pos_embed",
        "transformer.fft_cls_token",
        "transformer.cls_token_pos_embed",
        "transformer.channel_embed",
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
        "norm2_time",
        "norm2_fft",
    ]
    end_lr = pl_module.hparams.config["end_lr"]/Lambda
    decay_power = pl_module.hparams.config["lr_policy"]
    optim_type = pl_module.hparams.config["optim"]
    if Lambda != 1.0:
        assert optim_type=='Lion', f"optim_type must be Lion when Lambda!=1.0"
    if pl_module.hparams.config['mode'] !='pretrain' and pl_module.hparams.config['get_param_method'] == 'layer_decay':
        optimizer_grouped_parameters = param_groups_lrd(pl_module, lr=lr, weight_decay=wd,
                                                        layer_decay=pl_module.hparams.config["layer_decay"])
    else:
        optimizer_grouped_parameters = param_groups_no_layer_decay(pl_module, lr=lr, weight_decay=wd)


    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    elif optim_type == "Lion":
        optimizer = Lion(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.99))
    rank_zero_info("****Optim type = {}".format(optimizer))
    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        max_steps = (
                len(pl_module.trainer.datamodule.train_dataloader())
                * pl_module.trainer.max_epochs
                // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)
    rank_zero_info("****Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

    if decay_power == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=max_steps,
            lr_min=pl_module.hparams.config['min_lr']/Lambda,
            warmup_lr_init=pl_module.hparams.config['warmup_lr']/Lambda,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True
        )
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=warmup_steps,
        #     num_training_steps=max_steps,
        # )
        sched = {"scheduler": scheduler, "interval": "step"}

    else:
        scheduler = get_scheduler(optimizer, opt={'lr_policy': str(decay_power)},
                                  warmup_steps=warmup_steps, max_steps=max_steps, end_lr=end_lr, decay_power=decay_power, Lambda=Lambda)

        sched = {"scheduler": scheduler, "interval": "step"}

    rank_zero_info('****Scheduler: {sched}'.format(sched=sched))
    return (
        [optimizer],
        [sched],
    )
