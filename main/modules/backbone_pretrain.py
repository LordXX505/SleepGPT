import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from lightning.pytorch.cli import LRSchedulerTypeUnion
from lightning.pytorch.utilities.types import STEP_OUTPUT
from . import get_optm
from main.utils import init_weights, set_metrics
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scipy import interpolate
from timm.models import create_model
from . import heads
from typing import Any, Optional
from . import objectives
from . import multiway_transformer
from lightning import LightningModule
import pynvml


class Model_Pre(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.first_loss_step = False

        self.save_hyperparameters()
        self.relative_position_index = None
        self.relative_position_embed = None
        self.first_log_gpu = False
        self.lr = config['lr']
        # only for test
        # mask_ex = torch.ones((32, 57))
        # mask_ex[:, :50] = 0
        # self.example_input_array = {"batch": {"epochs": torch.Tensor(32, 57, 3000), 'mask': mask_ex}}
        self.mode = config['mode']
        self.patch_size = config['patch_size']
        self.transformer = multiway_transformer.__dict__[config["model_arch"]](
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            config=self.hparams.config,
        )
        self.tfffn_start_layer_index = self.transformer.tfffn_start_layer_index  # 12
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)
        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.time_only = config['time_only']

        self.fft_only = config['fft_only']
        # task layters
        if self.time_only is not True and self.fft_only is not True and self.hparams.config["loss_names"]['itm']>0:
            self.pooler = heads.Pooler(self.num_features, self.num_features)
            self.pooler.apply(init_weights)

        if config['loss_names']['mtm'] > 0:
            self.Masked_docoder = heads.Masked_decoder(self.transformer.embed_dim, self.transformer.patch_size, self.transformer.num_patches,
                                                       self.transformer.max_channels)
            self.Masked_docoder_fft = heads.Masked_decoder2(self.transformer.embed_dim, self.transformer.patch_size, self.transformer.num_patches,
                                                       self.transformer.max_channels)

        if config['loss_names']['itc'] > 0:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            if self.time_only:
                self.itc_weak_time_proj = heads.ITCHead(self.num_features)
                self.itc_weak_time_proj.apply(init_weights)
                self.itc_weak_mask_time_proj = heads.ITCHead(self.num_features)
                self.itc_weak_mask_time_proj.apply(init_weights)
                self.itc_time_strong_proj = heads.ITCHead(self.num_features)
                self.itc_time_strong_proj.apply(init_weights)
                self.logit_mask_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            if self.fft_only:
                self.itc_freq_weak_proj = heads.ITCHead(self.num_features)
                self.itc_freq_weak_proj.apply(init_weights)
                self.itc_freq_strong_proj = heads.ITCHead(self.num_features)
                self.itc_freq_strong_proj.apply(init_weights)
            if self.fft_only is not True and self.time_only is not True:
                self.itc_time_proj = heads.ITCHead(self.num_features)
                self.itc_time_proj.apply(init_weights)
                self.itc_freq_proj = heads.ITCHead(self.num_features)
                self.itc_freq_proj.apply(init_weights)
                self.itc_tf_time_proj = heads.ITCHead(self.num_features)
                self.itc_tf_freq_proj = heads.ITCHead(self.num_features)
                self.logit_tf_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

                self.itc_tf_time_proj.apply(init_weights)
                self.itc_tf_freq_proj.apply(init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.num_features)

        if config['loss_names']['FpFn'] > 0:
            assert config['spindle'] is True
            self.spindle_pred_proj = heads.Spindle_Head(self.num_features, self.transformer.patch_size)

        if config['loss_names']['CrossEntropy'] > 0:
            assert config['stage'] is True
            self.stage_pred_proj = heads.Stage_Head(self.num_features)
        set_metrics(self)
        self.current_tasks = list()
        self.init_weights()
        self.load_pretrained_weight()

        # ===================== Downstream ===================== #
        # ========================On do========================= #
        # spindle detection
        # movement disorder
        # epilepsy detection
        # sleep cognitive impairment
        # sleep rem detection

    def load_pretrained_weight(self):
        print(self.hparams.config["load_path"])
        if self.hparams.config["load_path"] != "":
            config = self.hparams.config
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))

            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                raise NotImplementedError
            if state_dict is None:
                rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt

            for key in state_dict:
                var = state_dict[key]
                rank_zero_info("%s = %s" % (key, str(var.size())))

            rank_zero_info(config["loss_names"])
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

    def init_weights(self):
        self.token_type_embeddings.apply(init_weights)
        if self.hparams.config['loss_names']['mtm'] > 0:
            self.Masked_docoder.apply(init_weights)
        if self.hparams.config['loss_names']['FpFn'] > 0:
            self.spindle_pred_proj.apply(init_weights)
        if self.hparams.config['loss_names']['CrossEntropy'] > 0:
            self.stage_pred_proj.apply(init_weights)
        if self.hparams.config["loss_names"]["itm"] > 0:
            self.itm_score.apply(init_weights)

    def build_relative_position_embed(self, config):
        if not self.transformer.need_relative_position_embed:
            self.relative_position_embed = False
            self.relative_position_index = None
            return
        else:
            raise NotImplementedError("relative_position_embed is not implemented now.")

    def get_attention_mask(self, attention_mask: torch.Tensor = None, attention_mask_fft: torch.Tensor = None):
        num_patches = self.transformer.num_patches
        if self.time_only or self.fft_only:
            attention_mask = attention_mask.repeat_interleave(num_patches, dim=1)
            cls_token = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            return [cls_token, attention_mask]
        attention_mask = attention_mask.repeat_interleave(num_patches, dim=1)
        cls_token = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask_fft = attention_mask_fft.repeat_interleave(num_patches, dim=1)
        cls_token_fft = torch.ones((attention_mask_fft.shape[0], 1), device=attention_mask_fft.device)
        return [cls_token, attention_mask, cls_token_fft, attention_mask_fft]

    def gpu_monitor(self, x, phase='transformer.block', block_log=True):
        if x.is_cuda and self.first_log_gpu is False:
            print("*******beginning {}********".format(phase))
            pynvml.nvmlInit()
            unit = 1024 * 1024 * 1024
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("device: ", x.device)
            print("Memory Total: ", meminfo.total / unit)
            print("Memory Free: ", meminfo.free / unit)
            print("Memory Used: ", meminfo.used / unit)
            if block_log:
                self.first_log_gpu = True

    def infer(self, batch, time_mask=False, stage="train"):
        epochs = batch['epochs']  # time, fft
        attention_mask = batch['mask']
        if self.time_only:
            res = self.transformer.time_embed(epochs, attn_mask=attention_mask[1], mask=time_mask, mask_w=batch['random_mask'][0])
        elif self.fft_only:
            res = self.transformer.fft_embed(epochs, attn_mask=attention_mask[1], mask=time_mask)
        else:
            if time_mask:
                res = self.transformer.embed(epochs, attn_mask=attention_mask[1], mask=time_mask,
                                             mask_w=batch['random_mask'][
                                                 0])  # get embeddings  # ret:{embed:[N,L_t + L_f + 2,D], mask:[N, L_t]}
            else:
                res = self.transformer.embed(epochs, attn_mask=attention_mask[1], mask=time_mask)
        attention_mask = torch.cat(attention_mask, dim=1)  # batch, L_t+L_f+2
        time_max_len = res['x_len']  # 1+num_patches*max_channels
        # print('time_max_len', time_max_len)
        x = res['x']  # time, fftfor
        # print('res x ', torch.isnan(x).sum())
        # assert attention_mask.shape[1] == x.shape[1]
        x_embeds, fft_embeds = (
            x[:, :time_max_len] + self.token_type_embeddings(torch.zeros((x.shape[0], time_max_len), dtype=torch.long,
                                                                         device=x.device)),
            x[:, time_max_len:] + self.token_type_embeddings(
                torch.ones((x.shape[0], x.shape[1] - time_max_len), dtype=torch.long, device=x.device))
        )
        x_embeds_nan = torch.isnan(x_embeds).sum()
        fft_embeds_nan = torch.isnan(fft_embeds).sum()
        assert x_embeds_nan == 0 and fft_embeds_nan == 0, f"the time embeds nan is {x_embeds_nan} and fft is {fft_embeds_nan}"
        if self.time_only:
            co_embeds = x_embeds
            attention_mask = attention_mask[:, :co_embeds.shape[1]]
        elif self.fft_only:
            co_embeds = fft_embeds
            attention_mask = attention_mask[:, :co_embeds.shape[1]]
        else:
            co_embeds = torch.cat((x_embeds, fft_embeds), dim=1)
        # print('x_embeds, fft_embeds', torch.isnan(x_embeds).sum(), torch.isnan(fft_embeds).sum())
        x = co_embeds
        for i, blk in enumerate(self.transformer.blocks):
            # print('-------------------layer {}-------------------'.format(i))
            if self.time_only:
                x = blk(x, mask=attention_mask, modality_type='time', relative_position_bias=None)
            elif self.fft_only:
                x = blk(x, mask=attention_mask, modality_type='fft', relative_position_bias=None)
            else:
                x = blk(x, mask=attention_mask, modality_type='tf', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        if self.training:
            self.gpu_monitor(x, block_log=False)
        x = self.transformer.norm(x)
        time_feats, fft_feats = (
            x[:, :time_max_len],
            x[:, time_max_len:]
        )
        if time_mask is not True:
            cls_feats = self.pooler(x[:, 0])
            cls_feats_fft = None
        else:
            if self.time_only:
                cls_feats = self.Masked_docoder(time_feats)  # b, L*t, patch_size
                cls_feats_fft = None
            else:
                cls_feats = self.Masked_docoder(time_feats)  # b, L*t, patch_size
                cls_feats_fft = self.Masked_docoder_fft(fft_feats)
        ret = {
            "time_feats": time_feats,
            "fft_feats": fft_feats,
            "cls_feats": cls_feats,
            "cls_feats_fft": cls_feats_fft,
            "time_max_len": time_max_len,
            "batch": batch,  # epochs, mask, Stage_label, Spindle_label
            'time_mask_patch': res['time_mask_patch'],  # mask to calculate the loss
            'fft_mask_patch': res['fft_mask_patch'],
            'stage': stage,
        }
        # print("cls_feats:", torch.isnan(cls_feats).sum(), cls_feats.shape)
        return ret

    def infer_time_only(self, batch):
        weak_aug, weak_attention_mask = batch['epochs'][0], batch['mask'][0]
        strong_aug, strong_attention_mask = batch['epochs'][1], batch['mask'][1]
        weak_aug_res = self.transformer.time_embed(weak_aug, attn_mask=weak_attention_mask[1], mask=False,
                                                   )
        strong_aug_res = self.transformer.time_embed(strong_aug, attn_mask=strong_attention_mask[1], mask=False)
        mask_x_res = self.transformer.time_embed(weak_aug, attn_mask=weak_attention_mask[1], mask=True,
                                                 mask_w=batch['random_mask'][0])

        assert weak_aug_res['x_len'] == strong_aug_res['x_len']
        time_max_len = weak_aug_res['x_len']
        weak_x = weak_aug_res['x']
        strong_x = strong_aug_res['x']
        weak_attention_mask = torch.cat(weak_attention_mask, dim=1)
        strong_attention_mask = torch.cat(strong_attention_mask, dim=1)
        weak_x_embeds, strong_x_embeds = (
            weak_x[:, :time_max_len] + self.token_type_embeddings(
                torch.zeros((weak_x.shape[0], time_max_len), dtype=torch.long,
                            device=weak_x.device)),
            strong_x[:, :time_max_len] + self.token_type_embeddings(
                torch.zeros((strong_x.shape[0], time_max_len), dtype=torch.long,
                            device=strong_x.device)),
        )
        x = weak_x_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=weak_attention_mask, modality_type='time', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        weak_x_embeds = x
        x = strong_x_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=strong_attention_mask, modality_type='time', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        strong_x_embeds = x
        if self.training:
            self.gpu_monitor(x, phase='weak_x_embeds and strong_x_embeds', block_log=False)
        time_weak_hiddens, time_strong_hiddens = weak_x_embeds, strong_x_embeds
        time_weak_hiddens = self.transformer.norm(time_weak_hiddens)
        time_strong_hiddens = self.transformer.norm(time_strong_hiddens)
        time_weak_feats, time_strong_feats = (
            time_weak_hiddens, time_strong_hiddens
        )
        cls_weak_feats, cls_strong_feats = self.itc_weak_time_proj(time_weak_feats[:, 0]), self.itc_time_strong_proj(
            time_strong_feats[:, 0])

        cls_weak_feats = cls_weak_feats / cls_weak_feats.norm(dim=-1, keepdim=True)
        cls_strong_feats = cls_strong_feats / cls_strong_feats.norm(dim=-1, keepdim=True)

        mask_x = mask_x_res['x']
        mask_x_embeds = mask_x[:, :time_max_len] + self.token_type_embeddings(
            torch.zeros((mask_x.shape[0], time_max_len), dtype=torch.long,
                        device=mask_x.device))

        x = mask_x_embeds
        for i, blk in enumerate(self.transformer.blocks):
            # print('-------------------layer {}-------------------'.format(i))
            x = blk(x, mask=weak_attention_mask, modality_type='time', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        x = self.transformer.norm(x)
        if self.training:
            self.gpu_monitor(x, phase="mask_x_embeds", block_log=False)

        time_feats = x
        cls_mask_feats = self.Masked_docoder(time_feats)

        cls_weak_mask_feats = self.itc_weak_mask_time_proj(time_feats[:, 0])
        cls_weak_mask_feats = cls_weak_mask_feats / cls_weak_mask_feats.norm(dim=-1, keepdim=True)

        ret = {
            "cls_weak_feats": cls_weak_feats,
            "cls_strong_feats": cls_strong_feats,
            "cls_weak_mask_feats": cls_weak_mask_feats,
            "batch": batch,  # epochs, ids_keep, mask, Stage_label, Spindle_label,
            "mask_feats": cls_mask_feats,
            'time_mask_patch': mask_x_res['time_mask_patch'],  # mask to calculate the loss
        }
        return ret

    def infer_fft_only(self, batch):
        weak_aug, weak_attention_mask = batch['epochs'][0], batch['mask'][0]
        strong_aug, strong_attention_mask = batch['epochs'][1], batch['mask'][1]
        weak_aug_res = self.transformer.fft_embed(weak_aug, mask=False)
        strong_aug_res = self.transformer.fft_embed(strong_aug, mask=False)
        assert weak_aug_res['x_len'] == strong_aug_res['x_len']
        fft_max_len = weak_aug_res['x_len']
        weak_x = weak_aug_res['x']
        strong_x = strong_aug_res['x']
        weak_attention_mask = torch.cat(weak_attention_mask, dim=1)
        strong_attention_mask = torch.cat(strong_attention_mask, dim=1)
        weak_x_embeds, strong_x_embeds = (
            weak_x[:, :fft_max_len] + self.token_type_embeddings(
                torch.ones((weak_x.shape[0], fft_max_len), dtype=torch.long,
                           device=weak_x.device)),
            strong_x[:, :fft_max_len] + self.token_type_embeddings(
                torch.ones((strong_x.shape[0], fft_max_len), dtype=torch.long,
                           device=strong_x.device)),
        )
        x = weak_x_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=weak_attention_mask, modality_type='fft', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        weak_x_embeds = x
        x = strong_x_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=strong_attention_mask, modality_type='fft', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        strong_x_embeds = x
        if self.training:
            self.gpu_monitor(x, block_log=False)

        fft_weak_hiddens, fft_strong_hiddens = weak_x_embeds, strong_x_embeds
        fft_weak_hiddens = self.transformer.norm(fft_weak_hiddens)
        fft_strong_hiddens = self.transformer.norm(fft_strong_hiddens)
        fft_weak_feats, fft_strong_feats = (
            fft_weak_hiddens, fft_strong_hiddens
        )
        cls_weak_feats, cls_strong_feats = self.itc_freq_weak_proj(fft_weak_feats[:, 0]), self.itc_freq_strong_proj(
            fft_strong_feats[:, 0])

        cls_weak_feats = cls_weak_feats / cls_weak_feats.norm(dim=-1, keepdim=True)
        cls_strong_feats = cls_strong_feats / cls_strong_feats.norm(dim=-1, keepdim=True)

        ret = {
            "cls_weak_feats": cls_weak_feats,
            "cls_strong_feats": cls_strong_feats,
            "batch": batch,  # epochs, ids_keep, mask, Stage_label, Spindle_label,
        }
        return ret

    def infer_time(self, batch, mask=False):
        epochs = batch['epochs']  # time
        attention_mask = batch['mask']

        res = self.transformer.time_embed(epochs, attn_mask=attention_mask[1],
                                          mask=mask)  # get embeddings  # ret:{embed:[N,L_t + 1,D], mask:None# }
        time_max_len = res['x_len']  # num_patches*max_channels
        x = res['x']  # time
        attention_mask = torch.cat((attention_mask[0], attention_mask[3]), dim=1)  # batch, L_t+1
        x_embeds, fft_embeds = (
            x[:, :time_max_len] + self.token_type_embeddings(torch.zeros((x.shape[0], time_max_len), dtype=torch.long,
                                                                         device=x.device)),
            None
        )
        co_embeds = x_embeds
        x = co_embeds
        all_hidden_states = []

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=attention_mask, modality_type='time', relative_position_bias=None)
            all_hidden_states.append(x)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer time transformer.blocks layer{i} is out of break"
        tfffn_hiddens = all_hidden_states[self.tfffn_start_layer_index - 1]
        for tfffn_index in range(self.tfffn_start_layer_index, self.num_layers):
            tfffn_hiddens = self.transformer.blocks[tfffn_index](tfffn_hiddens, mask=attention_mask, modality_type="tf",
                                                                 relative_position_bias=None)
            tfffn_hiddens_nan = torch.isnan(tfffn_hiddens).sum()
            assert tfffn_hiddens_nan == 0, f"infer_time tfffn_hiddens_nan transformer.blocks layer{tfffn_index} is out of break"
        if self.training:
            self.gpu_monitor(x, block_log=False)

        time_hiddens = all_hidden_states[-1]
        time_hiddens = self.transformer.norm(time_hiddens)

        time_feats, fft_feats = (
            time_hiddens,
            None
        )
        cls_feats = self.itc_time_proj(time_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        tfffn_hiddens = self.transformer.norm(tfffn_hiddens)
        cls_tfffn_feats = self.itc_tf_time_proj(tfffn_hiddens[:, 0])
        cls_tfffn_feats = cls_tfffn_feats / cls_tfffn_feats.norm(dim=-1, keepdim=True)

        ret = {

            "time_feats": time_feats,
            "fft_feats": fft_feats,
            "cls_feats": cls_feats,
            "cls_tfffn_feats": cls_tfffn_feats,
            "batch": batch,  # epochs, ids_keep, mask, Stage_label, Spindle_label
        }
        return ret

    def infer_fft(self, batch, mask=False):
        epochs = batch['epochs'] # fft
        # print('infer_fft_epochs', epochs)
        res = self.transformer.fft_embed(epochs, mask=mask)  # get embeddings  # ret:{embed:[N,L_f + 1,D], mask:None# }
        # print(res)
        fft_max_len = res['x_len']  # num_patches*max_channels
        x = res['x']  # fft
        attention_mask = batch['mask']
        attention_mask = torch.cat((attention_mask[2], attention_mask[3]), dim=1)  # batch, L_f+1
        x_embeds, fft_embeds = (
            None,
            x[:, :fft_max_len] + self.token_type_embeddings(torch.ones((x.shape[0], fft_max_len), dtype=torch.long,
                                                                       device=x.device), ),
        )
        co_embeds = fft_embeds
        # print('infer_fft_epochs_embeds', co_embeds)
        x = co_embeds
        all_hidden_states = []

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=attention_mask, modality_type='fft', relative_position_bias=None)
            all_hidden_states.append(x)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer_fft transformer.blocks layer{i} is out of break"
        tfffn_hiddens = all_hidden_states[self.tfffn_start_layer_index - 1]
        for tfffn_index in range(self.tfffn_start_layer_index, self.num_layers):
            tfffn_hiddens = self.transformer.blocks[tfffn_index](tfffn_hiddens, mask=attention_mask, modality_type="tf",
                                                                 relative_position_bias=None)
            tfffn_hiddens_nan = torch.isnan(tfffn_hiddens).sum()
            assert tfffn_hiddens_nan == 0, f"infer_fft tfffn_hiddens_nan transformer.blocks layer{tfffn_index} is out of break"
        if self.training:
            self.gpu_monitor(x, block_log=False)

        fft_hiddens = all_hidden_states[-1]
        fft_hiddens = self.transformer.norm(fft_hiddens)

        time_feats, fft_feats = (
            None,
            fft_hiddens
        )

        cls_feats = self.itc_freq_proj(fft_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        tfffn_hiddens = self.transformer.norm(tfffn_hiddens)
        cls_tfffn_feats = self.itc_tf_freq_proj(tfffn_hiddens[:, 0])
        cls_tfffn_feats = cls_tfffn_feats / cls_tfffn_feats.norm(dim=-1, keepdim=True)

        ret = {

            "time_feats": time_feats,
            "fft_feats": fft_feats,
            "cls_feats": cls_feats,
            "cls_tfffn_feats": cls_tfffn_feats,
            "batch": batch,  # epochs, ids_keep, mask, Stage_label, Spindle_label
        }
        return ret

    def patchify_2D(self, labels):
        """
        Args:
            labels: (N, channels, time, FFT)
        Returns:
            res: (N, patches_time, patch)
        """
        patch_size = (2, 100)
        patches = (labels.shape[2] // patch_size[0], labels.shape[3]//patch_size[1])
        x = labels.reshape(shape=(labels.shape[0], labels.shape[1], patches[0], patch_size[0], patch_size[1]))  # N, c, patches, patch_size
        x = x.reshape(labels.shape[0], labels.shape[1]*patches[0], -1)
        return x

    def unpatchify_2D(self, x):
        """
        x: (N, patches_time, patch)
        """
        patch_size = (2, 100)

        p = self.transformer.patch_size
        num_patch = self.transformer.num_patches
        x = x.reshape(x.shape[0], self.transformer.max_channels,  num_patch, patch_size[0], patch_size[1])
        time = x.reshape(x.shape[0], -1, num_patch * patch_size[0], patch_size[1])
        return time

    def patchify(self, labels):
        """
        Args:
            labels: (N, channels, fs*duration)
        Returns:
            res: (N, patches_time, patch)
        """
        patch_size = self.patch_size
        assert labels.shape[2] % patch_size == 0
        x = labels.reshape(labels.shape[0], labels.shape[1], -1, patch_size)
        x = x.reshape(x.shape[0], -1 ,patch_size)
        return x

    def unpatchify(self, x):
        """
        x: (N, patches_time, patch)
        """
        p = self.transformer.patch_size
        num_patch = self.transformer.num_patches
        x = x.reshape(x.shape[0], self.transformer.max_channels, -1, p)
        time = x.reshape(x.shape[0], -1, num_patch * p)
        return time

    def forward_masked_loss_channel(self, predict, labels, time_mask_patch):
        """
        Args:
            predict:  [N, L_t, patch_size:200]
            labels:   [N, L_t, fs*duration]
            time_mask_patch:

        Returns:

        """
        patch_label = self.patchify(labels)
        assert predict.shape == patch_label.shape
        # assert predict.shape[1] == self.transformer.num_patches * self.hparams.config['random_choose_channels']
        # compare_idx = torch.gather(input=predict, dim=1, index=(torch.where(time_mask_patch == 1)[0]).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 200))[0]
        # compare_idx_x = compare_idx.unsqueeze(0)
        # compare_idx_y = compare_idx.unsqueeze(1)
        # print(torch.abs(compare_idx_x-compare_idx_y))
        if self.hparams.config['loss_function'] == 'l1':
            l1loss = nn.L1Loss(reduction='none')
            loss = l1loss(predict, patch_label)
        elif self.hparams.config['loss_function'] == 'l2':
            loss = (predict - patch_label) ** 2
        else:
            loss = (predict - patch_label) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * time_mask_patch).reshape(predict.shape[0], self.hparams.config['random_choose_channels'], -1).sum(dim=-1)  # mean loss on removed patches
        loss = loss/time_mask_patch.reshape(predict.shape[0], self.hparams.config['random_choose_channels'], -1).sum(dim=-1)
        return loss

    def forward_masked_loss(self, predict, labels, time_mask_patch):
        """
        Args:
            predict:  [N, L_t, patch_size:200]
            labels:   [N, L_t, fs*duration]
            time_mask_patch:

        Returns:

        """
        patch_label = self.patchify(labels)
        if not self.first_log_gpu:
            rank_zero_info(f"predict shape: {predict.shape}, patch_label shape: {patch_label.shape}")
            rank_zero_info(f"time_mask_patch: {time_mask_patch}")
        assert predict.shape == patch_label.shape
        # compare_idx = torch.gather(input=predict, dim=1, index=(torch.where(time_mask_patch == 1)[0]).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 200))[0]
        # compare_idx_x = compare_idx.unsqueeze(0)
        # compare_idx_y = compare_idx.unsqueeze(1)
        # print(torch.abs(compare_idx_x-compare_idx_y))
        if self.hparams.config['loss_function'] == 'l1':
            l1loss = nn.L1Loss(reduction='none')
            loss = l1loss(predict, patch_label)
        elif self.hparams.config['loss_function'] == 'l2':
            loss = (predict - patch_label) ** 2
        else:
            loss = (predict - patch_label) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * time_mask_patch).sum() / time_mask_patch.sum()  # mean loss on removed patches
        return loss

    def forward_masked_loss_2D(self, predict, labels, time_mask_patch):
        """
        Args:
            predict:  [N, L_t, patch_size:200]
            labels:   [N, L_t, fs*duration]
            time_mask_patch:

        Returns:

        """
        patch_label = self.patchify_2D(labels)  # N, 15*C, 2*100
        assert predict.shape == patch_label.shape
        assert predict.shape[1] == self.transformer.num_patches*self.transformer.max_channels
        if not self.first_log_gpu:
            rank_zero_info(f"predict shape: {predict.shape}, patch_label shape: {patch_label.shape}")
            rank_zero_info(f"time_mask_patch: {time_mask_patch}")
        # compare_idx = torch.gather(input=predict, dim=1, index=(torch.where(time_mask_patch == 1)[0]).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 200))[0]
        # compare_idx_x = compare_idx.unsqueeze(0)
        # compare_idx_y = compare_idx.unsqueeze(1)
        # print(torch.abs(compare_idx_x-compare_idx_y))
        if self.hparams.config['loss_function'] == 'l1':
            l1loss = nn.L1Loss(reduction='none')
            loss = l1loss(predict, patch_label)
        elif self.hparams.config['loss_function'] == 'l2':
            loss = (predict - patch_label) ** 2
        else:
            loss = (predict - patch_label) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * time_mask_patch).sum() / time_mask_patch.sum()  # mean loss on removed patches
        return loss
    def prepare_forward(self):
        pass

    def forward(self, batch, stage) -> Any:
        ret = dict()
        if 1:
            # get the FFT
            with torch.no_grad():
                # pynvml.nvmlInit()
                # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # unit = 1024 * 1024 * 1024
                # print("-----------FFT-----------")
                # print("Memory Total: ", meminfo.total/unit)
                # print("Memory Free: ", meminfo.free/unit)
                # print("Memory Used: ", meminfo.used/unit)
                # if stage=='validation':
                #     print(batch['epochs'].shape[0])
                if len(batch['epochs']) == 1:
                    if self.time_only:
                        attention_mask = self.get_attention_mask(attention_mask=batch['mask'][0],)  # List[[b, 1], [b, num_patch*c], [b, 1], [b, num_patch*c]]
                        batch['mask'] = attention_mask
                        batch['epochs'] = (batch['epochs'][0])
                    elif self.fft_only:
                        epochs_fft, attn_mask_fft = self.transformer.get_fft(batch['epochs'][0], batch['mask'][0], aug=True)
                        attention_mask = self.get_attention_mask(attn_mask_fft)  # List[[b, 1], [b, num_patch*c], [b, 1], [b, num_patch*c]]
                        batch['mask'] = attention_mask
                        batch['epochs'] = (epochs_fft)
                    else:
                        epochs_fft, attn_mask_fft = self.transformer.get_fft(batch['epochs'][0], batch['mask'][0])
                        batch['epochs'] = (batch['epochs'][0], epochs_fft)
                        if not self.first_log_gpu:
                            print(batch['mask'][0].shape)
                        attention_mask = self.get_attention_mask(batch['mask'][0],
                                                                 attn_mask_fft)  # List[[b, 1], [b, num_patch*c], [b, 1], [b, num_patch*c]]

                        batch['mask'] = attention_mask
                elif batch['epochs'].shape[0] == 2:
                    if "mtm" in self.current_tasks:
                        assert self.time_only is True
                        weak_attention_mask = self.get_attention_mask(attention_mask=batch['mask'][0])
                        strong_attention_mask = self.get_attention_mask(attention_mask=batch['mask'][1])
                        batch['mask'] = [weak_attention_mask, strong_attention_mask]
                    else:
                        assert self.fft_only is True
                        epochs_weak_fft, attn_weak_mask_fft = self.transformer.get_fft(batch['epochs'][0], batch['mask'][0])
                        epochs_strong_fft, attn_strong_mask_fft = self.transformer.get_fft(batch['epochs'][1], batch['mask'][1], aug=True)
                        batch['epochs'] = (epochs_weak_fft, epochs_strong_fft)

                        weak_attention_mask = self.get_attention_mask(attention_mask=attn_weak_mask_fft)
                        strong_attention_mask = self.get_attention_mask(attention_mask=attn_strong_mask_fft)
                        batch['mask'] = [weak_attention_mask, strong_attention_mask]

                else:
                    raise NotImplementedError("batch['epochs'].shape[0] should be equal to or less than 2")
                # print(attention_mask)
                # print('forward epochs_fft is nan:', torch.isnan(epochs_fft).sum())
                # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # print("-----------FFT_post-----------")
                # print("Memory Total: ", meminfo.total/unit)
                # print("Memory Free: ", meminfo.free/unit)
                # print("Memory Used: ", meminfo.used/unit)

                # batch['epochs'] = torch.ones(48, 10+8, 3000, device=self.device)
                # batch['mask'] = self.get_attention_mask(torch.ones(48, 10, device=self.device),
                #                                         torch.ones(48, 8, device=self.device))
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch, time_mask=True, stage=stage))
            return ret

        if self.time_only or self.fft_only:
            ret.update(objectives.compute_time_fft_only(self, batch, stage=stage))
            if self.training:
                self.gpu_monitor(batch['epochs'][0], phase="time_only or fft_only all gpu consuming", block_log=True)

            return ret

        if "mtm" in self.current_tasks:
            # print('mtm')
            ret.update(objectives.compute_mtm(self, batch, stage))
        if "itc" in self.current_tasks:
            # print('itc')
            ret.update(objectives.compute_itc(self, batch, stage=stage))
        if "itm" in self.current_tasks:
            # print('itm')
            ret.update(
                objectives.compute_itm_hardneg(self, batch, ret['itc_f2t_logits'], ret['itc_t2f_logits'], stage=stage))
        if self.training:
            self.gpu_monitor(batch['epochs'][0], phase="all task consume", block_log=True)
        # if "FpFn" in self.current_tasks:
        #     ret.update(objectives.compute_FpFn(self, batch))
        #
        # if "CrossEntropy" in self.current_tasks:
        #     ret.update(objectives.compute_timeonly_mlm(self, batch))
        return ret

    def training_step(self, batch) -> STEP_OUTPUT:
        self.set_task()
        output = self(batch, stage="train")
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        if not self.first_loss_step:
            res = [v for k, v in output.items() if "loss" in k]
            rank_zero_info(f"total loss: f{len(res)}")
            self.first_loss_step = True
        return total_loss

    def on_train_epoch_end(self) -> None:
        # print('on_train_batch_end')
        # if self.global_step % 10 == 0:
        #     for name, parms in self.named_parameters():
        #         self.log(f'{name}/grad_value', torch.mean(parms.grad))
        self.epoch_end(stage="train")

    def validation_step(self, batch, batch_idx):
        self.set_task()
        output = self(batch, stage="validation")

    def on_validation_epoch_end(self) -> None:
        self.epoch_end(stage="validation")

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        # for params in self.optimizers().param_groups:
        #     print(params['lr'], params['weight_decay'])
        # if self.hparams.config["lr_policy"] in ['cosine', 'polynomial_decay'] or isinstance(self.hparams.config["lr_policy"], int):
        #     scheduler.step(self.global_step)  # type: ignore[call-arg]
        # else:
        if self.hparams.config['lr_policy'] == 'cosine':
            scheduler.step_update(self.global_step)
        else:
            scheduler.step()
        # for params in self.optimizers().param_groups:
        #     print(params['lr'], params['weight_decay'])

    def configure_optimizers(self):
        return get_optm.set_schedule(self)

    def set_task(self):
        return self._set_task()

    def _set_task(self):
        self.current_tasks = [
            k for k, v in self.hparams.config["loss_names"].items() if v >= 1
        ]

    def epoch_end(self, stage):
        phase = stage
        the_metric = 5

        for loss_name, v in self.hparams.config["loss_names"].items():
            if v < 1:
                continue
            value = 0
            if loss_name == 'mtm':
                value = getattr(self, f"{phase}_{loss_name}_loss").compute()
                self.log(f"{loss_name}/{phase}/score", value, prog_bar=True, on_epoch=True)
                value = - value
                getattr(self, f"{phase}_{loss_name}_loss").reset()
                value2 = getattr(self, f"{phase}_{loss_name}_loss2").compute()
                self.log(f"{loss_name}/{phase}/score", value, prog_bar=True, on_epoch=True)
                value = value - value2
                getattr(self, f"{phase}_{loss_name}_loss2").reset()
            elif loss_name == 'itc':
                if self.time_only or self.fft_only:
                    value_w2s = getattr(self, f"{phase}_{loss_name}_w2s_accuracy").compute()
                    self.log(f"{loss_name}/{phase}/w2s_accuracy_epoch", value_w2s)
                    getattr(self, f"{phase}_{loss_name}_w2s_accuracy").reset()

                    value_s2w = getattr(self, f"{phase}_{loss_name}_s2w_accuracy").compute()
                    self.log(f"{loss_name}/{phase}/s2w_accuracy_epoch", value_s2w)
                    getattr(self, f"{phase}_{loss_name}_s2w_accuracy").reset()
                    self.log(
                        f"{loss_name}/{phase}/loss_epoch",
                        getattr(self, f"{phase}_{loss_name}_loss").compute(),
                    )
                    getattr(self, f"{phase}_{loss_name}_loss").reset()
                    if self.time_only:
                        value_w2s_mask = getattr(self, f"{phase}_{loss_name}_w2s_mask_accuracy").compute()
                        self.log(f"{loss_name}/{phase}/w2s_mask_accuracy_epoch", value_w2s_mask)
                        getattr(self, f"{phase}_{loss_name}_w2s_mask_accuracy").reset()

                        value_s2w_mask = getattr(self, f"{phase}_{loss_name}_s2w_mask_accuracy").compute()
                        self.log(f"{loss_name}/{phase}/s2w_mask_accuracy_epoch", value_s2w_mask)
                        getattr(self, f"{phase}_{loss_name}_s2w_mask_accuracy").reset()

                        value = value_w2s + value_s2w + value_w2s_mask + value_s2w_mask
                    else:
                        value = value_w2s +value_s2w
                else:
                    value_f2t = getattr(self, f"{phase}_{loss_name}_f2t_accuracy").compute()
                    self.log(f"{loss_name}/{phase}/f2t_accuracy_epoch", value_f2t)
                    getattr(self, f"{phase}_{loss_name}_f2t_accuracy").reset()

                    value_t2f = getattr(self, f"{phase}_{loss_name}_t2f_accuracy").compute()
                    self.log(f"{loss_name}/{phase}/t2f_accuracy_epoch", value_t2f)
                    getattr(self, f"{phase}_{loss_name}_t2f_accuracy").reset()

                    self.log(
                        f"{loss_name}/{phase}/loss_epoch",
                        getattr(self, f"{phase}_{loss_name}_loss").compute(),
                    )
                    getattr(self, f"{phase}_{loss_name}_loss").reset()

                    value_tf_f2t = getattr(self, f"{phase}_{loss_name}_tf_f2t_accuracy").compute()
                    self.log(f"{loss_name}/{phase}/tf_f2t_accuracy_epoch", value_tf_f2t)
                    getattr(self, f"{phase}_{loss_name}_tf_f2t_accuracy").reset()

                    value_vl_t2f = getattr(self, f"{phase}_{loss_name}_tf_t2f_accuracy").compute()
                    self.log(f"{loss_name}/{phase}/tf_t2f_accuracy_epoch", value_vl_t2f)
                    getattr(self, f"{phase}_{loss_name}_tf_t2f_accuracy").reset()

                    value = value_f2t + value_t2f
            elif loss_name == 'itm':
                value = getattr(self, f"{phase}_{loss_name}_accuracy").compute()
                self.log(f"{loss_name}/{phase}/accuracy_epoch", value)
                getattr(self, f"{phase}_{loss_name}_accuracy").reset()
                self.log(
                    f"{loss_name}/{phase}/loss_epoch",
                    getattr(self, f"{phase}_{loss_name}_loss").compute(),
                )
                getattr(self, f"{phase}_{loss_name}_loss").reset()
            the_metric += value
        rank_zero_info(f"{phase}/{the_metric}")
        self.log(f"{phase}/the_metric", the_metric, prog_bar=True, on_epoch=True)
