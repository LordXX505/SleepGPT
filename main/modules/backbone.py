import copy
import os

import h5py
from lightning.pytorch.utilities import grad_norm
import sys
from functools import partial
from einops import rearrange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from lightning.pytorch.cli import LRSchedulerTypeUnion
from lightning.pytorch.utilities.types import STEP_OUTPUT
from . import get_optm
from main.Visualization import plot_conf, Visual_cal_all
from main.utils import init_weights, set_metrics
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scipy import interpolate
from timm.models import create_model
from . import heads
from typing import Any, Optional
from . import objectives
from . import multiway_transformer
from lightning import LightningModule
from . import mixup
import pynvml
from . import vit
from . import Swin_transformer
from main.utils import cal_F1_score, cal_Precision, cal_Recall
import torch.distributed as dist
from main.modules.long_net import LongNetTransformer
from . import WearableAdapter
class Model(LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.time_fft_relative_position_index = None
        self.fft_relative_position_index = None
        self.time_relative_position_index = None
        self.save_hyperparameters()
        self.num_relative_distance = None
        self.relative_position_bias_table = None
        self.all_num_relative_distance = None
        self.relative_position_index = None
        self.relative_position_embed = None
        self.smoothing = config['smoothing']
        self.first_log_gpu = False
        self.lr = config['lr']
        self.use_all_label = self.hparams.config['use_all_label']
        self.res_index = []
        self.res_label = []
        self.res_feats = []
        self.res_name = {}
        self.visual = config['visual']
        self.loss_mode = None
        if 'fold_now' in kwargs:
            self.fold_now = kwargs['fold_now']
        if self.visual is True:
            self.min_loss = 1000
            self.min_idx = -1
        # only for test
        # mask_ex = torch.ones((32, 57))
        # mask_ex[:, :50] = 0
        # self.example_input_array = {"batch": {"epochs": torch.Tensor(32, 57, 3000), 'mask': mask_ex}}
        if 'persub' in kwargs:
            self.persub = kwargs['persub']
            self.test_names = kwargs['test_sub_names']
            if self.visual is True:
                print(f'test_names: {self.test_names}')
                self.all_embeddings = torch.zeros(len(self.test_names), 5, 4, 1536)
                self.name_2_idx = {}
                idx = 0
                for name in self.test_names.values():
                    base_name = os.path.basename(name)
                    self.name_2_idx[base_name] = idx
                    idx += 1
            rank_zero_info(f'test_names: {self.test_names}')
            self._ckpt = kwargs['_ckpt']  # main_test_kfold_persub.py Model(_ckpt=ckpt)
            if 'persub_mode' in kwargs:
                self.persub_mode = kwargs['persub_mode']
        else:
            self.persub = False

        self.prob = config['sp_prob']
        self.IOU_th = config['IOU_th']
        self.mode = config['mode']
        self.patch_size = config['patch_size']
        self.use_pooling = config['use_pooling']
        self.use_relative_pos_emb = config['use_relative_pos_emb']
        self.fft_num_relative_distance = (3000 // config['patch_size']) * config['random_choose_channels']
        self.time_num_relative_distance = (3000 // config['patch_size']) * config['random_choose_channels']
        self.build_relative_position_embed(config)
        self.poly = config['poly']
        self.num_encoder_layers = config['num_encoder_layers']
        if self.use_pooling:
            dpr = [x.item() for x in
                   torch.linspace(0.00, config["drop_path_rate"], 12 + self.num_encoder_layers)]
        else:
            dpr = config["drop_path_rate"]
        rank_zero_info(f"dpr_backbone: {dpr}")
        self.use_triton = config['use_triton']
        if config['spo2_ods_settings']['inj'] is True:
            self.spo2_ods_settings = config['spo2_ods_settings']
        else:
            self.spo2_ods_settings = {}
        self.transformer = multiway_transformer.__dict__[config["model_arch"]](
            patch_size=self.patch_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            config=self.hparams.config,
            use_mean_pooling=(self.use_pooling == 'mean'),
            use_relative_pos_emb=self.use_relative_pos_emb,
            all_num_relative_distance=self.all_num_relative_distance,
            use_triton=self.use_triton,
            spo2_ods_settings=self.spo2_ods_settings
        )
        self.num_features = self.transformer.num_features
        # Build SpO₂ encoder if provided
        if self.spo2_ods_settings.get('inj', False):
            self.d_spo2 = self.spo2_ods_settings.get('d_spo2', self.num_features)
            xattn_layers = self.spo2_ods_settings.get('xattn_layers', [])
            self.spo2_encoder = WearableAdapter.MultiScaleSpO2Extractor(d_model=self.d_spo2)
            output_dim = self.num_features*2
            if self.spo2_ods_settings.get('concat', False):
                self.pre_proj = nn.Linear(self.num_features*2, self.d_spo2)
                self.pre_proj.apply(init_weights)
                output_dim = self.d_spo2*2
            if self.spo2_ods_settings.get('use_seq', False):
                input_dim = self.d_spo2*2 if self.spo2_ods_settings.get('concat', False) else self.num_features
                self.spo2_seq_encoder = WearableAdapter.OdsTemporalEncoder(input_dim=input_dim, hidden_dim=input_dim//2,
                                                                           num_layers=1, model_type=self.spo2_ods_settings.get('model_type', 'lstm'))
                output_dim = self.spo2_seq_encoder.output_dim
                self.spo2_seq_encoder.apply(init_weights)
            self.ods_head = nn.Sequential(
                nn.Linear(output_dim, output_dim//2), nn.ReLU(),
                nn.Linear(output_dim//2, 1)
            )
            self.spo2_encoder.apply(init_weights)
            self.ods_head.apply(init_weights)
            if self.spo2_ods_settings.get('hybrid_loss', False):
                ods_pos_w = self.spo2_ods_settings.get('ods_pos_w', 10)
                rank_zero_info(f'ods_pos_w: {ods_pos_w}')
                self.hybrid_loss = partial(WearableAdapter.hybrid_loss, ods_pos_w=ods_pos_w)
            else:
                self.hybrid_loss = None

        self.tfffn_start_layer_index = self.transformer.tfffn_start_layer_index  # 12
        self.num_layers = len(self.transformer.blocks)
        self.window_size = config['Swin_window_size']
        self.all_time = config['all_time']
        self.decoder_features = config['decoder_features']
        self.time_size = config['time_size']
        self.patch_time = config['patch_time']
        self.decoder_depth = config['decoder_depth']
        self.decoder_heads = config['decoder_heads']
        self.time_mask_last = False
        self.longnet_pool = config['longnet_pool']
        if self.patch_time != 30:
            self.time_mask_last = True
            print(f'backbone patch_time: {self.patch_time, self.time_mask_last}')
        if len(config['visual_setting']) != 0:
            if 'mask_same' in config['visual_setting'].keys():
                self.mask_same = config['visual_setting']['mask_same']
            else:
                self.mask_same = False
            if 'mode' in config['visual_setting'].keys():
                self.visual_mode = config['visual_setting']['mode']
                self.save_extra_name = config['visual_setting']['save_extra_name']
            if self.persub is False:
                self.save_cnt = 0
            else:
                self.save_cnt = {}
            rank_zero_info(f'visual_setting: {self.mask_same, self.mode}')

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.mixup_fn = None
        self.use_g_mid = config['use_g_mid']
        self.local_pooling = config['local_pooling']
        if self.local_pooling:
            self.stage_pred_local_proj = nn.Linear(self.num_features * 2, 5)
            self.stage_pred_local_proj.apply(init_weights)
            self.local_norm = nn.LayerNorm(self.num_features * 2)
        mixup_active = config['mixup'] > 0
        self.num_classes = kwargs['num_classes']
        if mixup_active:
            rank_zero_info('Using mixup')
            self.mixup_fn = mixup.Mixup(
                mixup_alpha=config['mixup'],
                prob=1, switch_prob=0.5, mode='batch',
                label_smoothing=0.1, num_classes=kwargs['num_classes'])

        self.time_only = config['time_only']
        self.fft_only = config['fft_only']
        self.multi_y = config['multi_y']
        self.return_alpha = config['return_alpha']
        self.use_local_f = config['local_pooling']
        assert self.time_only is False
        assert self.fft_only is False
        if config['loss_names']['Pathology'] > 0:
            if self.visual is not True:
                if self.use_pooling == 'longnet':
                    self.pooler = heads.Attn(self.num_features * 2, self.num_features * 2, reshape=False,
                                             channels=self.transformer.num_patches, double=False,
                                             channel_wise=False, return_alpha=self.return_alpha)
                    self.decoder_transformer_block = LongNetTransformer(dim = self.decoder_features, depth=self.decoder_depth, dim_head=self.decoder_heads,
                                                                  num_patches=self.time_size * self.transformer.num_patches, dilated_ratios=config['longnet_dr'], segment_lengths=config['longnet_sl'],
                                                                global_pool=self.longnet_pool
                                                                  )
                    self.head = heads.LongnetClassificationHead(self.decoder_features, num_classes=7, selected_layers=config['decoder_selected_layers'])
                else:
                    self.pooler = heads.Attn(self.num_features * 2, self.num_features * 2, reshape=False,
                                             channels=self.transformer.num_patches, double=False,
                                             channel_wise=False, return_alpha=self.return_alpha)
                    self.head = heads.LongnetClassificationHead(self.decoder_features, num_classes=7, selected_layers=config['decoder_selected_layers'])
                self.pooler.apply(init_weights)
                self.decoder_transformer_block.apply(init_weights)
                self.head.apply(init_weights)

        if config['loss_names']['CrossEntropy'] > 0:
            # task layers
            if self.use_pooling is not None:
                if self.all_time:
                    if self.use_pooling == 'linear':
                        self.pooler = heads.Attn(self.num_features * 2, self.decoder_features * 2, reshape=False,
                                                 channels=self.transformer.num_patches, double=False,
                                                 channel_wise=False, )
                        self.pooler.apply(init_weights)
                    elif 'Trans' in self.use_pooling:
                        self.time_norm = nn.LayerNorm(self.decoder_features)
                        self.fft_norm = nn.LayerNorm(self.decoder_features)
                        if 'cls' not in self.use_pooling:
                            self.pooler_fft = heads.Attn(self.num_features, self.decoder_features, reshape=False,
                                                         channels=self.transformer.max_channels, double=False,
                                                         channel_wise=True)
                            self.pooler_time = heads.Conv_embed(self.num_features, self.decoder_features,
                                                                kernel_size=self.transformer.num_patches,
                                                                reshape=True, )
                        else:
                            self.pooler_fft = nn.Linear(self.num_features, self.decoder_features)
                            self.pooler_time = nn.Linear(self.num_features, self.decoder_features)
                        self.pooler_fft.apply(init_weights)
                        self.pooler_time.apply(init_weights)

                        self.decoder_transformer_block = \
                            vit.Transformer(dim=self.decoder_features, out_features=self.decoder_features,
                                            nheads=config['decoder_heads'],
                                            feedforward_dim=self.decoder_features * 4, dropout=0.,
                                            num_encoder_layers=self.num_encoder_layers, pool=config['pool'],
                                            attn_drop_rate=0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            use_all_label=self.use_all_label,
                                            use_global_fft=config['use_global_fft'], drop_rate=0,
                                            num_patches=self.time_size, multi_y=self.multi_y,
                                            use_relative_pos_emb=self.use_relative_pos_emb,
                                            use_multiway=config['use_multiway'],
                                            drop_path_rate=dpr[12:])
                        self.decoder_transformer_block.apply(init_weights)
                    elif self.use_pooling == 'swin':
                        self.pooler = heads.Attn(self.num_features * 2, self.decoder_features, reshape=False,
                                                 channels=self.transformer.num_patches, double=False,
                                                 channel_wise=False, return_alpha=self.return_alpha)
                        self.pooler.apply(init_weights)
                        self.subject_tp = {}
                        self.subject_num = {}
                        self.decoder_transformer_block = Swin_transformer.GlobalSwin(
                            time_size=self.time_size, num_classes=kwargs['num_classes'],
                            embed_dim=self.decoder_features,
                            window_size=self.window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=config["drop_path_rate"],
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                            use_checkpoint=False, fused_window_process=False,
                            patches_resolution=self.time_size * self.transformer.num_patches,
                        )

                    elif self.use_pooling == 'longnet':
                        self.pooler = heads.Attn(self.num_features * 2, self.num_features * 2, reshape=False,
                                                 channels=self.transformer.num_patches, double=False,
                                                 channel_wise=False, return_alpha=self.return_alpha)
                        self.decoder_transformer_block = LongNetTransformer(dim=self.decoder_features,
                                                                            depth=self.decoder_depth,
                                                                            dim_head=self.decoder_heads,
                                                                            num_patches=self.time_size * self.transformer.num_patches,
                                                                            dilated_ratios=config['longnet_dr'],
                                                                            segment_lengths=config['longnet_sl'],
                                                                            sleep_stage=True,
                                                                            global_pool=True
                                                                            )
                        self.head = heads.LongnetClassificationHead(self.decoder_features, num_classes=kwargs['num_classes'],
                                                                    selected_layers=config['decoder_selected_layers'])
                        self.pooler.apply(init_weights)
                        self.decoder_transformer_block.apply(init_weights)
                        self.head.apply(init_weights)
                    else:
                        self.decoder_transformer_block = \
                            vit.DecoderTransformer(hidden_size=self.transformer.embed_dim,
                                                   decoder_depth=config['decoder_depth'],
                                                   enc_dim=self.decoder_features,
                                                   num_heads=24,
                                                   dpr=config["drop_path_rate"],
                                                   multi=False
                                                   )
                        self.decoder_transformer_block.apply(init_weights)
            else:
                self.pooler = heads.Attn(self.num_features * 2, self.decoder_features * 2, reshape=False,
                                         channels=self.transformer.num_patches, double=False,
                                         channel_wise=False, )
                self.pooler.apply(init_weights)
        self.weight = None
        if config['loss_names']['Spindle'] > 0:
            self.store_real_data = torch.zeros((20, 3000, 8, 1000), device=self.device)
            self.store_whole_eeg = torch.zeros((20, 3000, 1000), device=self.device)
            self.store_whole_label = torch.zeros((20, 3000, 1000), device=self.device)
            self.max_index = torch.zeros(20, device=self.device)
            self.weight = config['CE_Weight']
            self.spindle_pred_proj = heads.Spindle_Head(self.num_features, self.transformer.patch_size,
                                                        Use_FPN=config['Use_FPN'],
                                                        decoder_depth=config['Even_decoder_depth'],
                                                        enc_dim=config['Event_enc_dim'],
                                                        dpr=config["drop_path_rate"], num_queries=config['num_queries'],
                                                        seq_len=self.patch_time * 100, FPN_resnet=config['FPN_resnet'])
        if config['loss_names']['Apnea'] > 0:
            self.store_real_data = torch.zeros((20, 3000, 8, 1000), device=self.device)
            self.store_whole_eeg = torch.zeros((20, 3000, 1000), device=self.device)
            self.store_whole_label = torch.zeros((20, 3000, 1000), device=self.device)
            self.max_index = torch.zeros(20, device=self.device)
            self.weight = config['CE_Weight']
            self.apnea_pred_proj = heads.Apnea_Head(self.num_features, self.transformer.patch_size,
                                                    Use_FPN=config['Use_FPN'],
                                                    decoder_depth=config['Event_decoder_depth'],
                                                    enc_dim=config['Event_enc_dim'],
                                                    dpr=config["drop_path_rate"],
                                                    seq_len=self.patch_time * 100, )

        if config['loss_names']['CrossEntropy'] > 0 and self.use_pooling != 'swin' and self.use_pooling != 'longnet':
            # self.stage_pred_proj = heads.Stage_Head(self.num_features)
            if self.all_time:
                if self.use_local_f:
                    setattr(self, f'stage_pred_local_proj', nn.Sequential(
                        nn.Linear(self.num_features * 2, 5),
                    ))
                for name in self.multi_y:
                    if name == 'tf':
                        if self.use_pooling == 'attn':
                            # decoder_features = self.decoder_features*3
                            decoder_features = self.decoder_features * 3 * self.transformer.max_channels
                        else:
                            decoder_features = self.decoder_features * 2 * self.transformer.max_channels
                    else:
                        decoder_features = self.decoder_features
                    setattr(self, f'stage_pred_{name}_proj', nn.Sequential(
                        nn.Linear(decoder_features, 5),
                    ))
                    getattr(self, f'stage_pred_{name}_proj').apply(init_weights)
            else:
                self.stage_pred_proj = nn.Sequential(
                    nn.Linear(self.decoder_features * 2, 5),
                )
        if config['loss_names']['mtm'] > 0:
            self.Masked_docoder = heads.Masked_decoder(self.transformer.embed_dim, self.transformer.patch_size,
                                                       self.transformer.num_patches,
                                                       self.transformer.max_channels)
            self.Masked_docoder_fft = heads.Masked_decoder2(self.transformer.embed_dim, self.transformer.patch_size,
                                                            self.transformer.num_patches,
                                                            self.transformer.max_channels)
        set_metrics(self, **kwargs)
        self.current_tasks = list()
        self.init_weights()
        self.load_pretrained_weight()

    def load_pretrained_weight(self):
        print(self.hparams.config["load_path"])
        if self.hparams.config["load_path"] != "":

            config = self.hparams.config
            if isinstance(self.hparams.config["load_path"], list):
                ckpt = torch.load(self.hparams.config["load_path"][self.kfold], map_location="cpu")
            else:
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
        if self.hparams.config['loss_names']['Spindle'] > 0:
            self.spindle_pred_proj.apply(init_weights)

        if self.hparams.config['loss_names']['Apnea'] > 0:
            self.apnea_pred_proj.apply(init_weights)

    def build_relative_position_embed(self, config, modality=2):
        if not self.use_relative_pos_emb:
            self.relative_position_embed = False
            self.relative_position_index = None
            return
        rank_zero_info('*********Using relative_position_embed*********')
        channels = config['random_choose_channels']
        patch_size = 3000 // config['patch_size']
        rpe_num_patches = channels * modality
        self.num_relative_distance = (patch_size * 2 - 1) * channels * modality + rpe_num_patches * (
                rpe_num_patches - 1) + 6

        position_ids = torch.arange(patch_size)
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        min_distance = int(1 - patch_size)
        # rank_zero_info("min_distance: {}".format(min_distance))
        rel_pos_mat = rel_pos_mat - min_distance
        relative_position_index = \
            torch.zeros(size=(patch_size,) * 2, dtype=position_ids.dtype)
        relative_position_index[0:, 0:] = rel_pos_mat
        rpe_len = 0
        res_matrix = []
        for i in range(rpe_num_patches):
            temp_relative_position_index = relative_position_index.clone()
            temp_relative_position_index = temp_relative_position_index + rpe_len
            rpe_len += (patch_size * 2 - 1)
            row_index = []
            sum = 0
            for j in range(rpe_num_patches):
                if j != i:
                    row_index.append(torch.ones((patch_size, patch_size)) * (rpe_len + sum))
                    sum += 1
                else:
                    row_index.append(temp_relative_position_index)
            rpe_len += rpe_num_patches - 1
            row_index_res = torch.cat(row_index, dim=1)
            res_matrix.append(row_index_res)
        time_fft_relative_position_index = torch.cat(res_matrix, dim=0)
        self.time_fft_relative_position_index = torch.zeros(
            (2 + rpe_num_patches * patch_size, 2 + rpe_num_patches * patch_size))
        self.time_fft_relative_position_index[0, 0] = self.num_relative_distance - 1
        self.time_fft_relative_position_index[0, 1:] = self.num_relative_distance - 2
        self.time_fft_relative_position_index[1:, 0] = self.num_relative_distance - 3
        self.time_fft_relative_position_index[1:1 + patch_size * channels, 1:1 + patch_size * channels] \
            = time_fft_relative_position_index[:patch_size * channels, :patch_size * channels]
        self.time_fft_relative_position_index[1:1 + patch_size * channels, 2 + patch_size * channels:] \
            = time_fft_relative_position_index[:patch_size * channels, patch_size * channels:]
        self.time_fft_relative_position_index[2 + patch_size * channels:, 1:1 + patch_size * channels] \
            = time_fft_relative_position_index[patch_size * channels:, :patch_size * channels]
        self.time_fft_relative_position_index[2 + patch_size * channels:, 2 + patch_size * channels:] \
            = time_fft_relative_position_index[patch_size * channels:, patch_size * channels:]
        self.time_fft_relative_position_index[1 + patch_size * channels, :] = self.num_relative_distance - 5
        self.time_fft_relative_position_index[:, 1 + patch_size * channels] = self.num_relative_distance - 6
        self.time_fft_relative_position_index[
            1 + patch_size * channels, 1 + patch_size * channels] = self.num_relative_distance - 4

        assert (torch.max(
            self.time_fft_relative_position_index) == self.num_relative_distance - 1), f"{torch.max(self.relative_position_index)}, {self.num_relative_distance}"
        # self.num_relative_distance = 2*(self.time_num_relative_distance+1)
        # self.all_num_relative_distance = self.num_relative_distance + 2*(self.fft_num_relative_distance+1) + 2
        #
        # time_position_ids = torch.arange(self.time_num_relative_distance)
        # time_rel_pos_mat = time_position_ids.unsqueeze(-2) - time_position_ids.unsqueeze(-1)
        # min_distance = int(1-self.time_num_relative_distance)
        # # rank_zero_info("min_distance: {}".format(min_distance))
        # time_rel_pos_mat = time_rel_pos_mat - min_distance
        # time_relative_position_index = \
        #     torch.zeros(size=(self.time_num_relative_distance+1,) * 2, dtype=time_position_ids.dtype)
        # time_relative_position_index[1:, 1:] = time_rel_pos_mat
        # time_relative_position_index[0, 0:] = self.num_relative_distance - 3
        # time_relative_position_index[0:, 0] = self.num_relative_distance - 2
        # time_relative_position_index[0, 0] = self.num_relative_distance - 1
        # self.time_relative_position_index = time_relative_position_index
        # fft_position_ids = torch.arange(self.fft_num_relative_distance)
        # fft_rel_pos_mat = fft_position_ids.unsqueeze(-2) - fft_position_ids.unsqueeze(-1)
        # min_distance = int(1 - self.fft_num_relative_distance)
        # # rank_zero_info("min_distance: {}".format(min_distance))
        # fft_rel_pos_mat = fft_rel_pos_mat - min_distance
        # fft_rel_pos_mat += self.num_relative_distance + 2
        # fft_relative_position_index = \
        #     torch.zeros(size=(self.fft_num_relative_distance + 1,) * 2, dtype=fft_rel_pos_mat.dtype)
        # fft_relative_position_index[1:, 1:] = fft_rel_pos_mat
        # fft_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        # fft_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        # fft_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        # self.fft_relative_position_index = fft_relative_position_index
        # time2fft_relative_position_index = torch.ones(self.time_num_relative_distance+1, self.fft_num_relative_distance+1) * (
        #     self.num_relative_distance)
        # fft2time_relative_position_index = torch.ones(self.time_num_relative_distance+1, self.fft_num_relative_distance+1) * (
        #     self.num_relative_distance + 1)
        # time_row_relative_position_index = torch.cat((time_relative_position_index, time2fft_relative_position_index),
        #                                              1)  # 196, 393=197+196
        # fft_row_relative_position_index = torch.cat((fft2time_relative_position_index, fft_relative_position_index),
        #                                              1)  # 197, 393
        # time_fft_relative_position_index = torch.cat(
        #     (time_row_relative_position_index, fft_row_relative_position_index), 0)  # 393, 393
        # self.time_fft_relative_position_index = time_fft_relative_position_index
        self.all_num_relative_distance = self.num_relative_distance
        torch.set_printoptions(threshold=np.inf)
        rank_zero_info('Local RPE')

        rank_zero_info(self.time_fft_relative_position_index)

    def get_attention_mask(self, attention_mask: torch.Tensor = None, attention_mask_fft: torch.Tensor = None,
                           manual=False):
        if manual is True:
            return attention_mask
        num_patches = self.transformer.num_patches
        c = attention_mask.shape[1]
        if self.transformer.actual_channels is not None:
            attention_mask = torch.ones([attention_mask.shape[0], len(self.transformer.actual_channels)],
                                        device=attention_mask.device)
            attention_mask_fft = torch.ones([attention_mask.shape[0], len(self.transformer.actual_channels)],
                                            device=attention_mask.device)
            c = len(self.transformer.actual_channels)
        if self.time_only or self.fft_only:
            attention_mask = attention_mask.repeat_interleave(num_patches, dim=1)
            cls_token = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            return [cls_token, attention_mask]
        attention_mask = attention_mask.repeat_interleave(num_patches, dim=1)
        cls_token = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask_fft = attention_mask_fft.repeat_interleave(num_patches, dim=1)
        cls_token_fft = torch.ones((attention_mask_fft.shape[0], 1), device=attention_mask_fft.device)
        # if self.transformer.actual_channels is not None:
        #     assert 0 not in cls_token and 0 not in attention_mask
        if self.time_mask_last is True:
            mask_last_matrix = torch.zeros(15, device=attention_mask.device)
            idx = self.patch_time // 2
            mask_last_matrix[:idx] = 1
            mask_last_matrix = mask_last_matrix.unsqueeze(0).repeat(1, c)
            attention_mask = attention_mask * mask_last_matrix
            attention_mask_fft = attention_mask_fft * mask_last_matrix
        # if not self.first_log_gpu:
        #     rank_zero_info(f"attention_mask: {attention_mask}, attention_mask_fft: {attention_mask_fft}")
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

    def infer(self, batch, time_mask=False, stage="train", mask_same=False):
        cls_feats_fft = None
        epochs = batch['epochs']  # time, fft
        attention_mask = batch['mask']
        if time_mask:
            mask_w = batch['random_mask'][0]
        else:
            mask_w = None
        if mask_same is True:
            res = self.transformer.embed(epochs, attn_mask=attention_mask[1],
                                         mask=time_mask, mask_w=mask_w, mask_w_fft=mask_w)
        else:
            res = self.transformer.embed(epochs, attn_mask=attention_mask[1],
                                         mask=time_mask, mask_w=mask_w)
        # res = self.transformer.embed(epochs, attn_mask=attention_mask[1],
        #                              mask=False)  # get embeddings  # ret:{embed:[N,L_t + L_f + 2,D], mask:[N, L_t]}
        attention_mask = torch.cat(attention_mask, dim=-1)  # batch, 1 + L_t + 1 + L_f
        if not self.first_log_gpu:
            rank_zero_info(f"attention_mask.shape : {attention_mask.shape}")
        time_max_len = res['x_len']  # 1+num_patches*max_channels
        if self.transformer.actual_channels is not None:
            assert time_max_len == 1 + self.transformer.num_patches * len(self.transformer.actual_channels)
        # print('time_max_len', time_max_len)
        x = res['x']  # time, fft
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
        assert x_embeds_nan == 0 and fft_embeds_nan == 0, f"the time embeds nan is {x_embeds_nan} and fft is {fft_embeds_nan}, index:{batch['index']}"
        co_embeds = torch.cat((x_embeds, fft_embeds), dim=1)
        # print('x_embeds, fft_embeds', torch.isnan(x_embeds).sum(), torch.isnan(fft_embeds).sum())
        x = co_embeds
        all_hiddens = []
        spo2_tok = None
        h_ods_stack = []
        if self.spo2_ods_settings.get('inj', False):
            if self.spo2_encoder is not None and "spo2" in batch:
                spo2_tok = self.spo2_encoder(batch["spo2"])  # (B,T_eeg,d_spo2)
                assert torch.isnan(spo2_tok).sum() == 0, "SpO₂ encoder produced NaN"
        for i, blk in enumerate(self.transformer.blocks):
            if getattr(blk, "use_xattn", False) and spo2_tok is not None:
                x = blk(x,
                        mask=attention_mask,
                        modality_type='tf',
                        relative_position_bias=None,
                        relative_position_index=self.time_fft_relative_position_index,
                        spo2_tok=spo2_tok)  # ★ 传入
                # 保存第一次注入后的特征作为 ODS 输入
                h_ods_stack.append(x)
            else:
                x = blk(x,
                        mask=attention_mask,
                        modality_type='tf',
                        relative_position_bias=None,
                        relative_position_index=self.time_fft_relative_position_index)
            all_hiddens.append(x)
            assert not torch.isnan(x).any(), f"NaN in x before attn, layer {i}"
            assert not torch.isinf(x).any(), f"Inf in x before attn, layer {i}"
            if spo2_tok is not None:
                assert not torch.isnan(spo2_tok).any(), f"NaN in spo2_tok at layer {i}"
                assert not torch.isinf(spo2_tok).any(), f"Inf in spo2_tok at layer {i}"
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer transformer.blocks layer{i} is out of break"
        if len(h_ods_stack) == 0:
            h_ods = x
        elif len(h_ods_stack) == 1:
            h_ods = h_ods_stack[0]
        else:
            h_ods = torch.stack(h_ods_stack, dim=0).mean(0)  # (B, T_eeg, d)

        if self.training:
            self.gpu_monitor(x, block_log=False)
        x = self.transformer.norm(x)

        local_feats = None
        cls_feats = {}
        if self.spo2_ods_settings.get('inj', False):
            assert 'Ods_label' in batch.keys()
            time_ods_feats, fft_ods_feats = (
                h_ods[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                h_ods[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
            )
            time_h_ods = time_ods_feats[:, 1:]
            fft_h_ods = fft_ods_feats[:, 1:]
            feats_ods = torch.mean(torch.cat([time_h_ods, fft_h_ods], dim=-1), dim=1)
            if self.spo2_ods_settings.get('concat', False):
                spo2_tok_mean = spo2_tok.mean(1) # B,D
                spo2_tok_proj = self.pre_proj(feats_ods)
                concat_feats_ods = torch.cat([spo2_tok_proj, spo2_tok_mean], dim=-1)
            else:
                concat_feats_ods = feats_ods
            if self.spo2_ods_settings.get('use_seq', False):
                B, D = concat_feats_ods.shape
                concat_feats_ods_reshape = concat_feats_ods.reshape(-1, self.time_size, D)
                out = rearrange(self.spo2_seq_encoder(concat_feats_ods_reshape)[0], 'B T D -> (B T) D', T=self.time_size)
                spo2_cls_feats = self.ods_head(out)
            else:
                spo2_cls_feats = self.ods_head(concat_feats_ods)
            cls_feats.update({'spo2_cls_feats': spo2_cls_feats})

        if self.current_tasks[0] == 'CrossEntropy':
            time_feats, fft_feats = (
                x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
            )
            if self.local_pooling:
                x_time = time_feats[:, 1:].mean(1)
                x_fft = fft_feats[:, 1:].mean(1)
                local_feats = self.local_norm(torch.cat([x_time, x_fft], dim=-1))
            if self.use_pooling == 'linear':
                C = len(
                    self.transformer.actual_channels) if self.transformer.actual_channels is not None else self.transformer.max_channels
                x_time = rearrange(time_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_fft = rearrange(fft_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_pool = torch.cat([x_time, x_fft], dim=-1)
                attention_mask_t = rearrange(attention_mask[:, 1:time_max_len], 'B (C P) -> (B P) C', C=C)
                cls_feats = {'tf': self.pooler(x_pool, attn_mask=attention_mask_t)}
            elif 'Trans' in self.use_pooling:
                if 'cls' not in self.use_pooling:
                    x_time = time_feats[:, 1:]
                    # print("x_time:shape", x_time.shape)
                    x_fft = fft_feats[:, 1:].reshape(-1, self.transformer.max_channels, self.transformer.num_patches,
                                                     self.num_features)
                else:
                    x_time = time_feats[:, :1]
                    x_fft = fft_feats[:, :1]
                x_time = self.time_norm(
                    self.pooler_time(x_time).reshape(-1, self.time_size, self.decoder_features))
                # rank_zero_info(f"x_time shape:{x_time.shape}")
                # rank_zero_info(f"x_fft shape:{x_fft.shape}")

                x_fft = self.fft_norm(
                    self.pooler_fft(x_fft, time_split=-1).reshape(-1, self.time_size, self.decoder_features))
                if self.all_time:
                    # cls_feats = torch.cat([token[:, -1, :], self.decoder_transformer_block(token, batch['epochs'][:, :self.transformer.max_channels])], dim=-1)
                    cls_feats = self.decoder_transformer_block(x_time, x_fft, batch=batch['epochs'][0], use_tf=True,
                                                               use_g_mid=self.use_g_mid and not self.training
                                                               , epoch_mask=batch['epoch_mask'], training=self.training)
                else:
                    cls_feats = self.fc_norm(x)
            elif self.use_pooling == 'time_fft':
                tfffn_hiddens = all_hiddens[self.tfffn_start_layer_index - 1]
                for tfffn_index in range(self.tfffn_start_layer_index, self.num_layers):
                    tfffn_hiddens = self.transformer.blocks[tfffn_index](tfffn_hiddens, mask=attention_mask,
                                                                         modality_type="tf",
                                                                         relative_position_bias=None, not_use_tf=True, )
                    tfffn_hiddens_nan = torch.isnan(tfffn_hiddens).sum()
                    assert tfffn_hiddens_nan == 0, f"infer_fft tfffn_hiddens_nan transformer.blocks layer{tfffn_index} is out of break"
                self.transformer.norm(tfffn_hiddens)
                time_feats, fft_feats = (
                    tfffn_hiddens[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                    tfffn_hiddens[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
                )
                x_time_embed = time_feats[:, 0]
                x_fft_embed = fft_feats[:, 0]

                # b, c, d = x_fft.shape
                # x_fft = x_fft.reshape(b * self.transformer.max_channels, -1, d)
                x_time_embed = self.pooler_time(x_time_embed)
                x_fft_embed = self.pooler_fft(x_fft_embed)

                token = torch.stack([x_time_embed, x_fft_embed], dim=1)
                # time_cls = time_feats[:, 0]
                # fft_cls = fft_feats[:, 0]
                # alpha = self.tf_w(torch.stack([time_cls, fft_cls], dim=1), time_split=1)
                # x_time = x_time*alpha[0]
                # x_fft = x_fft*alpha[1]
                # x_all = torch.cat([x_time, x_fft], dim=1)
                # token = self.pooler(x_all, time_split=time_max_len-1)

                if self.all_time:
                    # cls_feats = torch.cat([token[:, -1, :], self.decoder_transformer_block(token, batch['epochs'][:, :self.transformer.max_channels])], dim=-1)
                    cls_feats = self.decoder_transformer_block(
                        token[:, 0].reshape(-1, self.time_size, self.decoder_features),
                        token[:, 1].reshape(-1, self.time_size, self.decoder_features),
                        use_tf=True,
                        use_g_mid=False,
                        epoch_mask=batch['epoch_mask'], training=self.training)
            elif self.use_pooling == 'longnet':
                C = len(
                    self.transformer.actual_channels) if self.transformer.actual_channels is not None else self.transformer.max_channels
                x_time = rearrange(time_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_fft = rearrange(fft_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_pool = torch.cat([x_time, x_fft], dim=-1)
                attention_mask_t = rearrange(attention_mask[:, 1:time_max_len], 'B (C P) -> (B P) C', C=C)
                token = self.pooler(x_pool, attn_mask=attention_mask_t)
                if self.return_alpha is True:
                    rank_zero_info(f'token: {token[0]}')
                    return {'token': token}
                outs = self.decoder_transformer_block(rearrange(token, '(B T) P D ->B (T P) D', T=self.time_size))
                logits = self.head(outs)
                cls_feats = {'tf': logits, 'features': outs}
            else:
                C = len(
                    self.transformer.actual_channels) if self.transformer.actual_channels is not None else self.transformer.max_channels
                x_time = rearrange(time_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_fft = rearrange(fft_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_pool = torch.cat([x_time, x_fft], dim=-1)
                attention_mask_t = rearrange(attention_mask[:, 1:time_max_len], 'B (C P) -> (B P) C', C=C)
                token = self.pooler(x_pool, attn_mask=attention_mask_t)
                if self.return_alpha is True:
                    rank_zero_info(f'token: {token[0]}')
                    return {'token': token}
                cls_feats.update(self.decoder_transformer_block(rearrange(token, '(B T) P D ->B (T P) D', T=self.time_size)))
        elif self.current_tasks[0] == 'Spindle':
            time_feats, fft_feats = (
                x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
            )
            max_len = self.patch_time // 2
            time_c3 = time_feats[:, 1:1 + max_len]
            fft_c3 = fft_feats[:, 1:1 + max_len]
            # cls = torch.cat([time_c3, fft_c3], dim=-1)
            # cls = torch.transpose(torch.cat([time_c3, fft_c3], dim=-1), dim0=1, dim1=2)
            cls_feats = self.spindle_pred_proj((time_c3, fft_c3))
        elif self.current_tasks[0] == 'Apnea':
            time_feats, fft_feats = (
                x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
            )
            max_len = self.patch_time // 2
            time_c3 = time_feats[:, 1:1 + max_len]
            fft_c3 = fft_feats[:, 1:1 + max_len]
            # cls = torch.cat([time_c3, fft_c3], dim=-1)
            # cls = torch.transpose(torch.cat([time_c3, fft_c3], dim=-1), dim0=1, dim1=2)
            cls_feats = self.apnea_pred_proj((time_c3, fft_c3))
        elif self.current_tasks[0] == 'Pathology':
            if  self.visual is True:
                time_feats, fft_feats = (
                    x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                    x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
                )
                cls_feats = time_feats[:, 1:]
                cls_feats_fft = fft_feats[:, 1:]
            else:
                time_feats, fft_feats = (
                    x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                    x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
                )
                C = len(
                    self.transformer.actual_channels) if self.transformer.actual_channels is not None else self.transformer.max_channels
                x_time = rearrange(time_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_fft = rearrange(fft_feats[:, 1:], 'B (C P) D -> (B P) C D', C=C)
                x_pool = torch.cat([x_time, x_fft], dim=-1)
                attention_mask_t = rearrange(attention_mask[:, 1:time_max_len], 'B (C P) -> (B P) C', C=C)
                token = self.pooler(x_pool, attn_mask=attention_mask_t)
                outs = self.decoder_transformer_block(rearrange(token, '(B T) P D ->B (T P) D', T=self.time_size))
                logits = self.head(outs)
                cls_feats = logits
        else:
            # print(time_max_len)
            time_feats, fft_feats = (
                x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
            )
            if hasattr(self, Masked_docoder) and hasattr(self, Masked_docoder_fft):
                cls_feats = self.Masked_docoder(time_feats)  # b, L*t, patch_size
                cls_feats_fft = self.Masked_docoder_fft(fft_feats)
            else:
                cls_feats = {'features': time_feats}
                cls_feats_fft = {'features': fft_feats}

            # if self.mask_same:
            #     print(f"res['time_mask_patch']: {res['time_mask_patch']}, res['fft_mask_patch']: {res['fft_mask_patch']}")
            #     assert res['time_mask_patch'] == res['fft_mask_patch'], f"res['time_mask_patch']: {res['time_mask_patch']}" \
            #                                                             f"res['fft_mask_patch']: {res['fft_mask_patch']}"
        ret = {
            "local_feats": local_feats,
            "cls_feats": cls_feats,
            "cls_feats_fft": cls_feats_fft,
            "time_max_len": time_max_len,
            "batch": batch,  # epochs, mask, Stage_label, Spindle_label
            'time_mask_patch': res['time_mask_patch'],  # mask to calculate the loss
            'fft_mask_patch': res['fft_mask_patch'],
            'stage': stage,
            'x': x,
        }

        # print("cls_feats:", torch.isnan(cls_feats).sum(), cls_feats.shape)
        return ret

    def infer_time(self, batch, time_mask=False, stage="train"):
        epochs = batch['epochs']  # time
        attention_mask = batch['mask']
        if time_mask:
            mask_w = batch['random_mask'][0]
        else:
            mask_w = None
        res = self.transformer.time_embed(epochs, attn_mask=attention_mask[1],
                                          mask=time_mask,
                                          mask_w=mask_w)  # get embeddings  # ret:{embed:[N,L_t + 1,D], mask:None# }
        time_max_len = res['x_len']  # 1 + num_patches*max_channels
        x = res['x']  # time
        attention_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)  # batch, L_t+1
        x_embeds, fft_embeds = (
            x[:, :time_max_len] + self.token_type_embeddings(torch.zeros((x.shape[0], time_max_len), dtype=torch.long,
                                                                         device=x.device)),
            None
        )
        co_embeds = x_embeds
        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=attention_mask, modality_type='time', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer time transformer.blocks layer{i} is out of break"
        if self.training:
            self.gpu_monitor(x, block_log=False)
        x = self.transformer.norm(x)
        if self.current_tasks[0] == 'CrossEntropy':
            time_feats, fft_feats = (
                x[:, :time_max_len] * attention_mask[:, :time_max_len].unsqueeze(-1),
                x[:, time_max_len:] * attention_mask[:, time_max_len:].unsqueeze(-1)
            )
            if self.use_pooling == 'attn':
                x_time = time_feats[:, 0]
                # b, c, d = x_fft.shape
                # x_fft = x_fft.reshape(b * self.transformer.max_channels, -1, d)
                x_time_embed = self.pooler_time(x_time)
                if self.all_time:
                    cls_feats = self.decoder_transformer_block(
                        x_time=x_time_embed.reshape(-1, self.time_size, self.decoder_features),
                        x_fft=None,
                        use_tf=False,
                        use_g_mid=False,
                        epoch_mask=batch['epoch_mask'], training=self.training)
                else:
                    raise NotImplementedError('self.use_pooling == "attn" and self.all_time is not True')

        ret = {
            "cls_feats": cls_feats,
            "time_max_len": time_max_len,
            "batch": batch,  # epochs, mask, Stage_label, Spindle_label
            'stage': stage,
            'x': x,
        }
        # print("cls_feats:", torch.isnan(cls_feats).sum(), cls_feats.shape)
        return ret

    def infer_fft(self, batch, time_mask=False, stage="train"):
        epochs = batch['epochs']  # time
        attention_mask = batch['mask']

        res = self.transformer.fft_embed(epochs,
                                         mask=False)  # get embeddings  # ret:{embed:[N,L_t + 1,D], mask:None# }
        fft_max_len = res['x_len']  # 1 + num_patches*max_channels
        x = res['x']  # time
        attention_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)  # batch, L_t+1
        x_embeds, fft_embeds = (
            None,
            x[:, :fft_max_len] + self.token_type_embeddings(torch.ones((x.shape[0], fft_max_len), dtype=torch.long,
                                                                       device=x.device)),
        )
        co_embeds = fft_embeds
        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=attention_mask, modality_type='fft', relative_position_bias=None)
            x_nan = torch.isnan(x).sum()
            assert x_nan == 0, f"infer time transformer.blocks layer{i} is out of break"
        if self.training:
            self.gpu_monitor(x, block_log=False)
        x = self.transformer.norm(x)
        if self.current_tasks[0] == 'CrossEntropy':
            time_feats, fft_feats = (
                None,
                x[:, :fft_max_len] * attention_mask[:, :fft_max_len].unsqueeze(-1)
            )
            if self.use_pooling == 'attn':
                x_fft = fft_feats[:, 0]
                # b, c, d = x_fft.shape
                # x_fft = x_fft.reshape(b * self.transformer.max_channels, -1, d)
                x_fft_embed = self.pooler_fft(x_fft)
                if self.all_time:
                    # cls_feats = torch.cat([token[:, -1, :], self.decoder_transformer_block(token, batch['epochs'][:, :self.transformer.max_channels])], dim=-1)
                    cls_feats = self.decoder_transformer_block(x_time=None,
                                                               x_fft=x_fft_embed.reshape(-1, self.time_size,
                                                                                         self.decoder_features),

                                                               use_tf=False,
                                                               use_g_mid=False,
                                                               epoch_mask=batch['epoch_mask'], training=self.training)
                else:
                    raise NotImplementedError('self.use_pooling == "attn" and self.all_time is not True')

        ret = {
            "cls_feats": cls_feats,
            "time_max_len": fft_max_len,
            "batch": batch,  # epochs, mask, Stage_label, Spindle_label
            'stage': stage,
            'x': x,
        }
        # print("cls_feats:", torch.isnan(cls_feats).sum(), cls_feats.shape)
        return ret

    def patchify_2D(self, labels):
        """
        Args:
            labels: (N, channels, time, FFT)
        Returns:
            res: (N, patches_time, patch)
        """
        patch_size = (2, 100)
        patches = (labels.shape[2] // patch_size[0], labels.shape[3] // patch_size[1])
        x = labels.reshape(shape=(
            labels.shape[0], labels.shape[1], patches[0], patch_size[0], patch_size[1]))  # N, c, patches, patch_size
        x = x.reshape(labels.shape[0], labels.shape[1] * patches[0], -1)
        return x

    def unpatchify_2D(self, x):
        """
        x: (N, patches_time, patch)
        """
        patch_size = (2, 100)

        p = self.transformer.patch_size
        num_patch = self.transformer.num_patches
        x = x.reshape(x.shape[0], self.transformer.max_channels, num_patch, patch_size[0], patch_size[1])
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
        x = x.reshape(x.shape[0], -1, patch_size)
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
        if self.hparams.config['loss_function'] == 'l1':
            l1loss = nn.L1Loss(reduction='none')
            loss = l1loss(predict, patch_label)
        elif self.hparams.config['loss_function'] == 'l2':
            loss = (predict - patch_label) ** 2
        else:
            loss = (predict - patch_label) ** 2
        global_variance = torch.var(labels, dim=-1)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * time_mask_patch).reshape(predict.shape[0], self.hparams.config['random_choose_channels'],
                                                -1).sum(dim=-1)  # mean loss on removed patches
        loss = loss / time_mask_patch.reshape(predict.shape[0], self.hparams.config['random_choose_channels'], -1).sum(
            dim=-1)
        assert loss.shape == global_variance.shape, f'loss shape: {loss.shape}, var shape: {global_variance.shape}'
        nmse = torch.where(global_variance != 0, loss / global_variance, 0)
        return nmse

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
        assert predict.shape[1] == self.transformer.num_patches * self.transformer.max_channels
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
        assert time_mask_patch.sum() != 0
        loss = (loss * time_mask_patch).sum() / time_mask_patch.sum()  # mean loss on removed patches
        return loss

    def forward_masked_loss_2D_channel(self, predict, labels, time_mask_patch):
        """
        Args:
            predict:  [N, L_t, patch_size:200]
            labels:   [N, L_t, fs*duration]
            time_mask_patch:

        Returns:

        """
        patch_label = self.patchify_2D(labels)  # N, 15*C, 2*100
        if isinstance(self.transformer.mask_ratio, list) and np.all(np.array(self.transformer.mask_ratio) == 0.0):
            return torch.tensor(torch.nan)
        assert predict.shape == patch_label.shape
        assert predict.shape[1] == self.transformer.num_patches * self.transformer.max_channels
        if not self.first_log_gpu:
            rank_zero_info(f"predict shape: {predict.shape}, patch_label shape: {patch_label.shape}")
            rank_zero_info(f"time_mask_patch: {time_mask_patch}")
        if self.hparams.config['loss_function'] == 'l1':
            l1loss = nn.L1Loss(reduction='none')
            loss = l1loss(predict, patch_label)
        elif self.hparams.config['loss_function'] == 'l2':
            loss = (predict - patch_label) ** 2
        else:
            loss = (predict - patch_label) ** 2
        global_variance = torch.var(labels, dim=(-2, -1))
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        assert time_mask_patch.sum() != 0
        loss = (loss * time_mask_patch).reshape(predict.shape[0], self.hparams.config['random_choose_channels'],
                                                -1).sum(dim=-1)  # mean loss on removed patches
        loss = loss / time_mask_patch.reshape(predict.shape[0], self.hparams.config['random_choose_channels'], -1).sum(
            dim=-1)
        nmse = torch.where(global_variance != 0, loss / global_variance, 0)
        return nmse

    def prepare_forward(self):
        pass

    def forward(self, batch, stage, manual=False, aug_fft=False, loss_mode=None) -> Any:
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
                assert len(batch['epochs']) == 1
                if 'Stage_label' in batch.keys():
                    batch['Stage_label'] = torch.stack(batch['Stage_label'], dim=0).squeeze(-1)
                if 'Ods_label' in batch.keys():
                    batch['Ods_label'] = torch.stack(batch['Ods_label'], dim=0).squeeze(1).squeeze(-1).reshape(-1)
                    batch['ods_valid'] = torch.stack(batch['ods_valid'], dim=0).squeeze(1).squeeze(-1).reshape(-1)
                if 'Spindle_label' in batch.keys():
                    batch['Spindle_label'] = torch.stack(batch['Spindle_label'], dim=0).squeeze(1).squeeze(-1)
                if 'Apnea_label' in batch.keys():
                    batch['Apnea_label'] = torch.stack(batch['Apnea_label'], dim=0).squeeze(1).squeeze(-1)
                if 'Pathology_label' in batch.keys():
                    batch['Pathology_label'] = torch.stack(batch['Pathology_label'], dim=0).squeeze(1).squeeze(-1)
                if self.training and self.mixup_fn is not None:
                    batch['epochs'][0], batch['Stage_label'] = self.mixup_fn(batch['epochs'][0], batch['Stage_label'])
                epochs_fft, attn_mask_fft = self.transformer.get_fft(batch['epochs'][0], batch['mask'][0], aug=aug_fft)
                batch['epochs'] = (batch['epochs'][0], epochs_fft)
                attention_mask = self.get_attention_mask(batch['mask'][0],
                                                         attn_mask_fft,
                                                         manual=manual)  # List[[b, 1], [b, num_patch*c], [b, 1], [b, num_patch*c]]
                batch['mask'] = attention_mask
                # for i in attention_mask:
                # rank_zero_info(f"{i}.shape: {i.shape}")
                if self.hparams.config['use_pooling'] == 'time_fft' or self.hparams.config['use_pooling'] == 'attn' \
                        or self.hparams.config['use_pooling'] == 'swin' or self.hparams.config['use_pooling'] == 'longnet':
                    # mid = int(self.time_size // 2) + 1
                    if not self.first_log_gpu:
                        rank_zero_info(f'stage mid : {-1}')
                        if 'Stage_label' in batch.keys():
                            rank_zero_info(f'stage shape : {batch["Stage_label"].shape}')
                        if 'Spindle_label' in batch.keys():
                            rank_zero_info(f'spindle shape : {batch["Spindle_label"].shape}')
                        if 'Apnea_label' in batch.keys():
                            rank_zero_info(f'apnea shape : {batch["Apnea_label"].shape}')
                        if 'Pathology_label' in batch.keys():
                            rank_zero_info(f'apnea shape : {batch["Pathology_label"].shape}')
                        rank_zero_info(f'epochs 0  shape : {batch["epochs"][0].shape}')
                        rank_zero_info(f'epochs 1  shape : {batch["epochs"][1].shape}')

                    if self.use_all_label == 'all':
                        if self.use_g_mid and not self.training:
                            batch['Stage_label'] = batch['Stage_label'][:, int(self.time_size // 2 + 1)]
                        else:
                            if self.mixup_fn is None or not self.training:
                                batch['Stage_label'] = batch['Stage_label'].reshape(-1)
                    else:
                        batch['Stage_label'] = batch['Stage_label'][:, -1]
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch, time_mask=False, stage=stage))
            return ret

        assert len(self.current_tasks) == 1 and self.current_tasks[0] in ['CrossEntropy', 'Spindle', 'mtm', 'Apnea','Pathology']
        if self.current_tasks[0] == 'mtm':
            ret.update(self.infer(batch, stage=stage, time_mask=True, mask_same=self.mask_same))
        if self.current_tasks[0] == 'CrossEntropy':
            ret.update(objectives.compute_ce(self, batch, stage=stage, persub=self.persub, loss_mode=loss_mode))
            if self.training:
                self.gpu_monitor(batch['epochs'][0], phase='compute_ce last gpu', block_log=True)
        if self.current_tasks[0] == 'Pathology':
            ret.update(objectives.compute_po(self, batch, stage=stage, persub=self.persub))
        if self.current_tasks[0] == 'Spindle':
            if stage == 'train':
                if self.trainer.max_steps is None or self.trainer.max_steps == -1:
                    max_steps = (
                            len(self.trainer.datamodule.train_dataloader())
                            * self.trainer.max_epochs
                            // self.trainer.accumulate_grad_batches
                    )
                else:
                    max_steps = self.trainer.max_steps
                # rank_zero_info(f'weight: {self.weight}')
                ret.update(objectives.compute_fpfn(self, prob=self.prob, IOU_th=self.IOU_th, batch=batch, stage=stage,
                                                   use_fpfn=self.hparams.config['use_fpfn'],
                                                   store=False, weight=self.weight))
            else:
                rank_zero_info('==========Using compute_entire_fpfn==========')
                rank_zero_info(f'forward weight: {self.weight}')
                ret.update(
                    objectives.compute_entire_fpfn(self, prob=self.prob, IOU_th=self.IOU_th, batch=batch, stage=stage,
                                                   use_fpfn=self.hparams.config['use_fpfn'],
                                                   store=False, compute=True, aggregate=True, weight=self.weight))
            if self.current_tasks[0] == 'Apnea':
                if stage == 'train':
                    ret.update(
                        objectives.compute_fpfn(self, prob=self.prob, IOU_th=self.IOU_th, batch=batch, stage=stage,
                                                use_fpfn=self.hparams.config['use_fpfn'],
                                                store=False, weight=self.weight))
                else:
                    rank_zero_info('==========Using compute_entire_fpfn==========')
                    rank_zero_info(f'forward weight: {self.weight}')
                    ret.update(
                        objectives.compute_entire_fpfn(self, prob=self.prob, IOU_th=self.IOU_th, batch=batch,
                                                       stage=stage,
                                                       use_fpfn=self.hparams.config['use_fpfn'],
                                                       store=False, compute=True, aggregate=True, weight=self.weight))


            if self.training:
                self.gpu_monitor(batch['epochs'][0], phase='compute_fpfn last gpu', block_log=True)

        return ret

    def _freeze_crossattn(self):
        for blk_id in self.hparams.config["spo2_ods_settings"]["xattn_layers"]:
            blk = self.transformer.blocks[blk_id]
            for p in blk.xattn.parameters():
                p.requires_grad = False
            for p in blk.gate_fc.parameters():
                p.requires_grad = False
            blk.gamma_spo.requires_grad = False
            blk.gamma_spo.data.zero_()

    def _unfreeze_crossattn(self):
        for blk_id in self.hparams.config["spo2_ods_settings"]["xattn_layers"]:
            blk = self.transformer.blocks[blk_id]
            for p in blk.xattn.parameters():
                p.requires_grad = True
            for p in blk.gate_fc.parameters():
                p.requires_grad = True
            blk.gamma_spo.requires_grad = True

    def _freeze_stage(self):
        # 冻结 EEG 主干与分类器
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.decoder_transformer_block.parameters():
            p.requires_grad = False
        for p in self.pooler.parameters():
            p.requires_grad = False

    def _unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def training_step(self, batch) -> STEP_OUTPUT:
        assert self.training is True
        self.set_task()
        if False and self.spo2_ods_settings.get('inj', False):
            epoch = self.current_epoch
            stage1_epoch = self.hparams.config.get("stage1_epoch", 5)
            stage2_epoch = self.hparams.config.get("stage2_epoch", 10)

            if epoch < stage1_epoch:
                self.loss_mode = "stage_only"
                self._freeze_crossattn()
            elif epoch < stage2_epoch:
                self.loss_mode = "ods_only"
                self._unfreeze_crossattn()
                if self.hparams.config.get("freeze_stage", False):
                    self._freeze_stage()
            else:
                self.loss_mode = "hybrid"
                self._unfreeze_all()
            output = self(batch, stage="train", loss_mode=self.loss_mode)
        else:
            output = self(batch, stage="train", loss_mode=self.loss_mode)
        # rank_zero_info(f'output: {output.items()}')
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    # def on_after_backward(self) -> None:
    #     rank_zero_info("on_after_backward enter")
    #     for n, p in self.named_parameters():
    #         if p.grad is None and p.requires_grad:
    #             rank_zero_info(n)
    #     rank_zero_info("on_after_backward exit")
    def on_train_epoch_end(self) -> None:
        # print('on_train_batch_end')
        # if self.global_step % 10 == 0:
        #     for name, parms in self.named_parameters():
        #         self.log(f'{name}/grad_value', torch.mean(parms.grad))
        self.epoch_end(stage="train")
#
#     ret = {
#         "local_feats": local_feats,
#         "cls_feats": cls_feats,
#         "cls_feats_fft": cls_feats_fft,
#         "time_max_len": time_max_len,
#         "batch": batch,  # epochs, mask, Stage_label, Spindle_label
#         'time_mask_patch': res['time_mask_patch'],  # mask to calculate the loss
#         'fft_mask_patch': res['fft_mask_patch'],
#         'stage': stage,
#         'x': x,
#     }
    def test_step(self, batch, batch_idx):
        if self.training:
            raise Exception(f"self.training is not False in test")
        self.set_task()
        if self.visual is True and self.training is not True:
            if self.visual_mode == 'UMAP':
                output = self(batch, stage="test")
                if "features" in output:
                    cls_feats_feature = output['features']
                else:
                    cls_feats_feature = torch.cat([output['cls_feats'], output['cls_feats_fft']], dim=-1)
                predicted = output['feats']
                true_lable = output['label']
                if predicted is None:
                    predicted = [0]
                name = batch['name']
                if 'save_info' in output:
                    save_info = output['save_info']
                else:
                    save_info = None
                if hasattr(self, 'fold_now'):
                    kfold = self.fold_now
                else:
                    kfold = self.hparams.config["kfold"] if self.hparams.config['kfold'] is not None else '0'
                if self.persub:
                    for idx in range(len(name)):
                        if save_info is not None:
                            # si = save_info[idx].item()
                            try:
                                self.all_embeddings[self.name_2_idx[name[idx]]][true_lable[idx]] += cls_feats_feature[idx].reshape(4, 15, -1).mean(dim=1).to('cpu')
                            except:
                                print(f'name idx: {name[idx]}, self.name_2_idx: {self.name_2_idx}')
                                raise RuntimeError
                            # self.save({'cls_feats_feature': cls_feats_feature[idx],
                            #            'predicted': predicted,
                            #            'true_lable': true_lable[idx]}, path=f'{self.visual_mode}/{self.save_extra_name}/{si}/{name[idx]}/{kfold}',
                            #           sub_name=name[idx], si=si)
                        else:
                            try:
                                self.all_embeddings[self.name_2_idx[name[idx]]][true_lable[idx]] += cls_feats_feature[idx].reshape(4, 15, -1).mean(dim=1).to('cpu')
                            # self.save({'cls_feats_feature': cls_feats_feature[idx],
                            #            'predicted': predicted,
                            #            'true_lable': true_lable[idx]},
                            #           path=f'{self.visual_mode}/{self.save_extra_name}/{name[idx]}/{kfold}',
                            #           sub_name=name[idx])
                            except:
                                print(f'name idx: {name[idx]}, self.name_2_idx: {self.name_2_idx}')
                                raise RuntimeError

                else:
                    self.save({'cls_feats_feature': cls_feats_feature,
                               'predicted': predicted,
                               'true_lable': true_lable},
                              path=f'{self.visual_mode}/{kfold}/{self.save_extra_name}')
            else:
                min_idx = Visual_cal_all.visual(batch, self, self.persub)
                self.min_idx = batch['name'][min_idx]
                rank_zero_info(f"min_idx: {min_idx}, batch_idx: {batch['name'][min_idx]}")
        else:
            output = self(batch, stage="test")
            if self.return_alpha is True:
                for i, stage in enumerate(batch['Stage_label']):
                    getattr(self, f"{stage}_mapping")(output['token'][i].view(-1))
            elif self.current_tasks[0] == 'CrossEntropy':
                self.res_index.append(output['index'].reshape(-1, self.time_size))
                self.res_label.append(output['label'].reshape(-1, self.time_size))
                self.res_feats.append(output['feats'].reshape(-1, self.time_size))

        rank_zero_info("test_step end")

        # rank_zero_info(f'total_loss: {total_loss}')

    def on_test_epoch_end(self) -> None:
        rank_zero_info("on_test_epoch_end")
        if self.current_tasks[0] == 'Spindle' or self.current_tasks[0] == 'Apnea':
            output = objectives.compute_entire_fpfn(self, prob=self.prob, IOU_th=self.IOU_th, batch=None, stage='test',
                                                    use_fpfn=self.hparams.config['use_fpfn'],
                                                    store=True, compute=False, aggregate=True, path=self.logger.version)
            rank_zero_info(f'test output: {output}')

        self.epoch_end(stage="test")

    def validation_step(self, batch, batch_idx):
        self.set_task()
        if self.training:
            raise Exception(f"self.training is not False in validation")
        output = self(batch, stage="validation")
        if self.current_tasks[0] == 'CrossEntropy':
            self.res_index.append(output['index'].reshape(-1, self.time_size))
            self.res_label.append(output['label'].reshape(-1, self.time_size))
            self.res_feats.append(output['feats'].reshape(-1, self.time_size))

    def on_validation_epoch_end(self) -> None:
        if self.current_tasks[0] == 'Spindle' or self.current_tasks[0] == 'Apnea':
            output = objectives.compute_entire_fpfn(self, prob=self.prob, IOU_th=self.IOU_th, batch=None,
                                                    stage='validation',
                                                    use_fpfn=self.hparams.config['use_fpfn'], path=self.logger.version,
                                                    store=True, compute=False, aggregate=True, weight=self.weight)
            rank_zero_info(f'validation output: {output}')
        self.epoch_end(stage="validation")
    def save_rem_features(self, save_dir, subject_id, features_dict):
        with h5py.File(os.path.join(save_dir, f"{subject_id}.h5"), 'a') as f:
            for epoch_id, feat_tensor in features_dict.items():
                f.create_dataset(f'epoch_{epoch_id}', data=feat_tensor.cpu().numpy(), compression='gzip')

    def predict_step(self, batch, batch_idx):
        self.set_task()
        if self.training:
            raise Exception(f"self.training is not False in validation")
        output = self(batch, stage="predict")
        feats_sub_dict = defaultdict(dict)
        name = batch['name']
        index = batch['index']
        if self.visual is True:
            if self.visual_mode == 'rem_feat':
                cls_feats = output['cls_feats']['features']
                B, C, D = cls_feats.shape
                cls_feats_fft = output['cls_feats_fft']['features']
                features = torch.cat([cls_feats, cls_feats_fft], dim=-1)
                items = [(str(names[i]), int(index[i]), features[i].detach().cpu()) for i in range(B)]
                for idx, n in enumerate(name):
                    feats_sub_dict[n][index[idx]] = features

                if hasattr(self, '_ckpt'):
                    path = os.path.join(path, f'{self._ckpt.split("/")[-1]}')
                else:
                    path = 'shhs_feat'
                os.makedirs(
                    os.path.join(f'./result',
                                 f'{path}'),
                    exist_ok=True)
                save_dir =  os.path.join(f'./result',
                                 f'{path}')
                for n in feats_sub_dict.keys():
                    self.save_rem_features(save_dir=save_dir, subject_id=n, features_dict=feats_sub_dict[n])
    def on_predict_epoch_end(self) -> None:
        self.epoch_end(stage="predict")

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        # for params in self.optimizers().param_groups:
        #     print(params['lr'], params['weight_decay'])
        # if self.hparams.config["lr_policy"] in ['cosine', 'polynomial_decay'] or isinstance(self.hparams.config["lr_policy"], int):
        #     scheduler.step(self.global_step)  # type: ignore[call-arg]
        # else:
        if 'finetune' in self.hparams.config['mode'].lower() or self.hparams.config['mode'] == 'Spindledetection' or 'Apneadetection' in self.hparams.config['mode']:
            mode = True
        else:
            mode = False
        if mode and self.hparams.config['lr_policy'] == 'cosine':
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
            k for k, v in self.hparams.config["loss_names"].items() if
            v >= 1
        ]

    def save(self, data_dict, path, sub_name=None, si=0, agg=True):
        if hasattr(self, '_ckpt'):
            path = os.path.join(path, f'{self._ckpt.split("/")[-1]}')
        os.makedirs(
            os.path.join(f'/home/cuizaixu_lab/huangweixuan/DATA/result',
                        f'{path}'),
            exist_ok=True)
        if agg is True:
            if self.persub is False:
                save_cnt = self.save_cnt
            else:
                if si in self.save_cnt:
                    if sub_name in self.save_cnt[si]:
                        save_cnt = self.save_cnt[si][sub_name]
                    else:
                        self.save_cnt[si] = {sub_name: 0}
                        save_cnt = 0
                else:
                    self.save_cnt[si] = {sub_name: 0}
                    save_cnt = 0
            torch.save(data_dict,
                       os.path.join(f'/home/cuizaixu_lab/huangweixuan/DATA/result',
                                    f'{path}', f'res_{save_cnt}.ckpt'))
            if self.persub is False:
                self.save_cnt += 1
            else:
                self.save_cnt[si][sub_name] += 1
        else:
            torch.save(data_dict,
                       os.path.join(f'/home/cuizaixu_lab/huangweixuan/DATA/result',
                                    f'{path}', f'res.ckpt'))

    def epoch_end(self, stage):
        phase = stage
        the_metric = 5
        if hasattr(self, 'fold_now'):
            kfold = self.fold_now
        else:
            kfold = self.hparams.config["kfold"] if self.hparams.config['kfold'] is not None else '0'
        if self.visual is True:
            if self.visual_mode != 'UMAP':
                value = getattr(self, f"{phase}_loss").compute()
                value2 = getattr(self, f"{phase}_loss2").compute()
                if hasattr(self, 'fold_now'):
                    kfold = self.fold_now
                else:
                    kfold = self.hparams.config["kfold"] if self.hparams.config['kfold'] is not None else '0'
                mode = self.visual_mode
                getattr(self, f"{phase}_loss").reset()
                getattr(self, f"{phase}_loss2").reset()
                for c in range(self.transformer.choose_channels.shape[0]):
                    self.log(f"{phase}_c{c}_loss", value[c], prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log(f"{phase}_c{c}_loss2", value2[c], prog_bar=True, on_epoch=True, sync_dist=True)
                persubject_loss = {}
                self.save(path=f'{mode}/{kfold}/overall_{self.transformer.mask_ratio}',
                          dict={'loss': value, 'loss2': value2})
                if self.persub:
                    for tn in self.test_names.values():
                        _tn = tn.split('/')[-1]
                        sub_loss = getattr(self, f"test_{_tn}_loss").compute()
                        sub_loss2 = getattr(self, f"test_{_tn}_loss2").compute()
                        persubject_loss[_tn] = {'loss1': sub_loss, 'loss2': sub_loss2}
                    self.save(path=f'{mode}/{kfold}', dict=persubject_loss)
            else:
                for idx, name in enumerate(self.test_names.values()):
                    base_name = os.path.basename(name)
                    self.save({'cls_feats_feature': self.all_embeddings[idx],
                               'predicted': [0],
                               'true_lable': [0]},
                              path=f'{self.visual_mode}/{self.save_extra_name}/{base_name}/{kfold}',
                              sub_name=base_name, agg=False)
        else:
            for loss_name, v in self.hparams.config["loss_names"].items():
                if v < 1:
                    continue
                metric = 0
                if loss_name == 'CrossEntropy':
                    if self.return_alpha is True:
                        res_dict = {}
                        for stage in [0, 1, 2, 3, 4]:
                            value = getattr(self, f'{stage}_mapping').compute()
                            res_dict[stage] = value
                            for i in range(value.shape[0]):
                                self.log(f"{loss_name}/{stage}/value_{i}", value[i], prog_bar=True, on_epoch=True,
                                         sync_dist=True)
                        self.save(res_dict, path=f'attn/{kfold}')
                        return
                    value = getattr(self, f"{phase}_{loss_name}_loss").compute()
                    self.log(f"{loss_name}/{phase}/score", value, prog_bar=True, on_epoch=True, sync_dist=True)
                    getattr(self, f"{phase}_{loss_name}_loss").reset()
                    max_acc = 0.0
                    max_macro = 0.0
                    ods_acc = 0.0
                    multi_y = copy.deepcopy(self.multi_y)
                    if self.spo2_ods_settings.get('inj', False):
                        ods_acc = getattr(self, f"{phase}_{loss_name}_accuracy_spo2_cls_feats").compute()
                        self.log(f"{loss_name}/{phase}/ods_acc", ods_acc,  on_epoch=True, sync_dist_group=True, prog_bar=True)
                        getattr(self, f"{phase}_{loss_name}_accuracy_spo2_cls_feats").reset()
                        confmat = getattr(self, f"{phase}_{loss_name}_conf_spo2_cls_feats").compute()
                        if stage == 'validation':
                            rank_zero_info(f"conf_spo2_cls_feats: {confmat}")
                        getattr(self, f"{phase}_{loss_name}_conf_spo2_cls_feats").reset()
                    if self.local_pooling:
                        multi_y.append('local')
                    for name in multi_y:
                        value_acc = float(
                            format(getattr(self, f"{phase}_{loss_name}_accuracy_{name}").compute(), '.3f'))
                        max_acc = max(max_acc, value_acc)
                        self.log(f"{loss_name}/{phase}/{name}/accuracy_epoch", value_acc, prog_bar=True, on_epoch=True,
                                 sync_dist=True)
                        getattr(self, f"{phase}_{loss_name}_accuracy_{name}").reset()
                        value = value_acc - value

                        confmat = getattr(self, f"{phase}_{loss_name}_conf_{name}").compute()
                        if stage == 'validation':
                            rank_zero_info(f"{name}: {confmat}")
                        precision, recall, kappa, sensitivity, specificity = objectives.confusion(confmat)
                        macro_f1 = torch.mean(2 * precision * recall / (precision + recall), dim=-1)
                        max_macro = max(max_macro, macro_f1)
                        self.log(f"{loss_name}/{phase}/{name}/macro_f1", macro_f1, prog_bar=True, on_epoch=True,
                                 sync_dist=True)
                        self.log(f"{loss_name}/{phase}/{name}/kappa_score", kappa, prog_bar=True, on_epoch=True,
                                 sync_dist=True)
                        for i in range(len(precision)):
                            self.log(f"ce/{phase}/ce_{i}_precision_{name}", precision[i])
                            self.log(f"ce/{phase}/ce_{i}_recall_{name}", recall[i])
                        if not self.training and stage == 'test':
                            confmat = confmat.cpu()
                            figure = plot_conf.plot_confusion_matrix(confmat, classes=5, normalize=False,
                                                                     title='Normalized confusion matrix')
                            tensorboard = self.logger.experiment
                            tensorboard.add_figure(f'Validation Normalized confusion matrix {self.global_step}', figure)
                            persubject_acc = {}
                            if self.persub:
                                for tn in self.test_names.values():
                                    _tn = tn.split('/')[-1]
                                    sub_conf = getattr(self, f"test_{_tn}_conf").compute()
                                    persubject_acc[_tn] = sub_conf
                                try:
                                    os.makedirs(
                                        f'/home/cuizaixu_lab/huangweixuan/DATA/temp_log/{self.hparams.config["kfold"]}',
                                        exist_ok=True)
                                    rank_zero_info(f'save persub path = /home/cuizaixu_lab/huangweixuan/DATA/temp_log/{self.hparams.config["kfold"]}')
                                    torch.save(persubject_acc, f'/home/cuizaixu_lab/'
                                                               f'huangweixuan/DATA/temp_log/{self.hparams.config["kfold"]}/{self._ckpt.split("/")[-1]}')
                                except:
                                    os.makedirs(
                                        f'/lustre/home/2201210064/Sleep/temp_log/{self.hparams.config["kfold"]}',
                                        exist_ok=True)
                                    rank_zero_info(
                                        f'save persub path = /lustre/home/2201210064/Sleep/temp_log/{self.hparams.config["kfold"]}')
                                    torch.save(persubject_acc, f'/lustre/home/'
                                                               f'2201210064/Sleep/temp_log/{self.hparams.config["kfold"]}/{self._ckpt.split("/")[-1]}')
                            # plt.close('all')

                        getattr(self, f"{phase}_{loss_name}_conf_{name}").reset()
                    metric = max_acc + max_macro + ods_acc
                    self.log(f"{loss_name}/{phase}/max_accuracy_epoch", max_acc, prog_bar=True, on_epoch=True,
                             sync_dist=True, )
                elif loss_name == 'Pathology':
                    value = getattr(self, f"{phase}_{loss_name}_loss").compute()
                    self.log(f"{loss_name}/{phase}/score", value, prog_bar=True, on_epoch=True, sync_dist=True)
                    getattr(self, f"{phase}_{loss_name}_loss").reset()
                    value_acc = float(
                        format(getattr(self, f"{phase}_{loss_name}_accuracy").compute(), '.3f'))
                    self.log(f"{loss_name}/{phase}/max_accuracy_epoch", value_acc, prog_bar=True, on_epoch=True,
                             sync_dist=True)
                    getattr(self, f"{phase}_{loss_name}_accuracy").reset()
                    confmat = getattr(self, f"{phase}_{loss_name}_conf").compute()
                    rank_zero_info(f"confmat: {confmat}")
                    precision, recall, kappa, sensitivity, specificity = objectives.confusion(confmat)
                    macro_f1 = torch.mean(2 * precision * recall / (precision + recall), dim=-1)
                    self.log(f"{loss_name}/{phase}/macro_f1", macro_f1, prog_bar=True, on_epoch=True,
                             sync_dist=True)
                    self.log(f"{loss_name}/{phase}/kappa_score", kappa, prog_bar=True, on_epoch=True,
                             sync_dist=True)
                    self.log(f"{loss_name}/{phase}/precision", precision, prog_bar=True, on_epoch=True,
                             sync_dist=True)
                    self.log(f"{loss_name}/{phase}/recall", recall, prog_bar=True, on_epoch=True,
                             sync_dist=True)
                    getattr(self, f"{phase}_{loss_name}_conf").reset()
                    metric = value_acc + macro_f1

                elif loss_name == 'Spindle':
                    value = getattr(self, f"{phase}_{loss_name}_loss").compute()
                    self.log(f"{loss_name}/{phase}/score", value, prog_bar=True, on_epoch=True, sync_dist=True)
                    getattr(self, f"{phase}_{loss_name}_loss").reset()
                    TP = getattr(self, f"{phase}_FpFn_TP").compute()
                    FN = getattr(self, f"{phase}_FpFn_FN").compute()
                    FP = getattr(self, f"{phase}_FpFn_FP").compute()
                    if stage == 'train':
                        Recall = cal_Recall(TP, FN)
                        Precision = cal_Precision(TP, FP)
                        F1 = cal_F1_score(Precision, Recall)
                    else:
                        Recall = getattr(self, f"{phase}_FpFn_Recall").compute()
                        # print(Recall)
                        Precision = getattr(self, f"{phase}_FpFn_Precision").compute()
                        F1 = getattr(self, f"{phase}_FpFn_F1").compute()
                    rank_zero_info(f'end: stage: {stage}, TP:{TP}, FN: {FN}, FP: {FP}, value: {value}, '
                                   f'recall: {Recall}, Precision:{Precision}, F1: {F1}')
                    self.log(f"{loss_name}/{phase}/Recall", Recall, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log(f"{loss_name}/{phase}/Precision", Precision, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log(f"{loss_name}/{phase}/F1", F1, prog_bar=True, on_epoch=True, sync_dist=True)
                    getattr(self, f"{phase}_FpFn_TP").reset()
                    getattr(self, f"{phase}_FpFn_FN").reset()
                    getattr(self, f"{phase}_FpFn_FP").reset()
                    if stage != 'train':
                        getattr(self, f"{phase}_FpFn_Recall").reset()
                        getattr(self, f"{phase}_FpFn_Precision").reset()
                        getattr(self, f"{phase}_FpFn_F1").reset()
                    self.store_real_data = torch.zeros((20, 3000, 8, 1000), device=self.device)
                    self.store_whole_eeg = torch.zeros((20, 3000, 1000), device=self.device)
                    self.store_whole_label = torch.zeros((20, 3000, 1000), device=self.device)
                    self.max_index = torch.zeros(20, device=self.device)
                    metric = value
                elif loss_name == 'Apnea':
                    value = getattr(self, f"{phase}_{loss_name}_loss").compute()
                    self.log(f"{loss_name}/{phase}/score", value, prog_bar=True, on_epoch=True, sync_dist=True)
                    getattr(self, f"{phase}_{loss_name}_loss").reset()
                    TP = getattr(self, f"{phase}_FpFn_TP").compute()
                    FN = getattr(self, f"{phase}_FpFn_FN").compute()
                    FP = getattr(self, f"{phase}_FpFn_FP").compute()
                    if stage == 'train':
                        Recall = cal_Recall(TP, FN)
                        Precision = cal_Precision(TP, FP)
                        F1 = cal_F1_score(Precision, Recall)
                    else:
                        Recall = getattr(self, f"{phase}_FpFn_Recall").compute()
                        # print(Recall)
                        Precision = getattr(self, f"{phase}_FpFn_Precision").compute()
                        F1 = getattr(self, f"{phase}_FpFn_F1").compute()
                    rank_zero_info(f'end: stage: {stage}, TP:{TP}, FN: {FN}, FP: {FP}, value: {value}, '
                                   f'recall: {Recall}, Precision:{Precision}, F1: {F1}')
                    self.log(f"{loss_name}/{phase}/Recall", Recall, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log(f"{loss_name}/{phase}/Precision", Precision, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log(f"{loss_name}/{phase}/F1", F1, prog_bar=True, on_epoch=True, sync_dist=True)
                    getattr(self, f"{phase}_FpFn_TP").reset()
                    getattr(self, f"{phase}_FpFn_FN").reset()
                    getattr(self, f"{phase}_FpFn_FP").reset()
                    if stage != 'train':
                        getattr(self, f"{phase}_FpFn_Recall").reset()
                        getattr(self, f"{phase}_FpFn_Precision").reset()
                        getattr(self, f"{phase}_FpFn_F1").reset()
                    self.store_real_data = torch.zeros((20, 3000, 8, 1000), device=self.device)
                    self.store_whole_eeg = torch.zeros((20, 3000, 1000), device=self.device)
                    self.store_whole_label = torch.zeros((20, 3000, 1000), device=self.device)
                    self.max_index = torch.zeros(20, device=self.device)
                    metric = value
                the_metric += metric
            self.log(f"{phase}/the_metric", the_metric, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.transformer, norm_type=2)
        self.log_dict(norms)
        if hasattr(self, "spindle_pred_proj"):
            norms2 = grad_norm(self.spindle_pred_proj, norm_type=2)
            self.log_dict(norms2)
        if hasattr(self, "apnea_pred_proj"):
            norms2 = grad_norm(self.apnea_pred_proj, norm_type=2)
            self.log_dict(norms2)
        if hasattr(self, "decoder_transformer_block"):
            norms2 = grad_norm(self.decoder_transformer_block, norm_type=2)
            self.log_dict(norms2)
