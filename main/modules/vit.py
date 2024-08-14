# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on BEIT code bases (https://github.com/microsoft/unilm/tree/master/beit)
# Written by Yutong Lin, Zhenda Xie
# --------------------------------------------------------
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import math
from functools import partial
from . import multiway_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from main.transforms import FFT_Transform
from . import heads
from main.utils import init_weights
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # comment out this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., all_num_relative_distance=0, use_relative_pos_emb=False, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_relative_pos_emb:
            assert all_num_relative_distance != -1
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(all_num_relative_distance, num_heads))

    def get_rel_pos_bias(self, relative_position_index):  # 196, 196
        relative_position_bias = F.embedding(
            relative_position_index.long().to(self.relative_position_bias_table.device),
            self.relative_position_bias_table)  # out = [196, 196, 144], tabele=[1126, 144], co=237,237,1444
        all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, x, y
        return all_relative_position_bias

    def forward(self, x, relative_position_index=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if relative_position_index is not None:
            attn = attn + self.get_rel_pos_bias(relative_position_index).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, use_relative_pos_emb=False, all_num_relative_distance=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_relative_pos_emb=use_relative_pos_emb, all_num_relative_distance=all_num_relative_distance)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, relative_position_index=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), relative_position_index=relative_position_index))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), relative_position_index=relative_position_index))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, out_features, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.max_len = 20

        print('[INFO] Maximum length of pos_enc: {}'.format(self.max_len))

        pe = torch.zeros(self.max_len, out_features)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_features, 2).float() * (-math.log(10000.0) / out_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        # if self.pe.shape != x.shape:
        #     pe = self.pe.unsqueeze(0)
        # else:
        #     pe = self.pe
        x = x + self.pe[:, :x.size(1)]
        return x


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Transformer(nn.Module):

    def __init__(self, dim, out_features, nheads, feedforward_dim, num_patches,
                 num_encoder_layers, dropout=0.0, pool='mean', use_global_fft=True,
                 drop_path_rate=None, qkv_bias=True, qk_scale=None, use_all_label=None,
                 drop_rate=0.0, attn_drop_rate=0.0, norm_layer=None, init_values=0.1,
                 use_relative_pos_emb=False, use_multiway=False, multi_y=None):

        super(Transformer, self).__init__()

        if drop_path_rate is None:

            drop_path_rate = [0.0] * num_encoder_layers
        rank_zero_info(f'Transformer dpr:{drop_path_rate}')
        self.time_fft_relative_position_index = None
        self.fft_relative_position_index = None
        self.time_relative_position_index = None
        self.all_num_relative_distance = 0
        self.num_relative_distance = None
        self.relative_position_index = None
        self.relative_position_embed = None
        self.channels = 4
        self.use_global_fft = use_global_fft
        if self.use_global_fft:
            # self.fft_conv_1d = nn.Conv1d(in_channels=self.channels, out_channels=self.channels*dim, kernel_size=1200
            #                              , groups=self.channels)
            self.fft_conv_norm = nn.LayerNorm(1200, eps=1e-6)
            for i in range(self.channels):
                setattr(self, f"fft_conv_{i}", nn.Linear(1200, dim))
                # setattr(self, f"fft_norm_{i}", nn.LayerNorm(dim))
        self.use_all_label = use_all_label
        # self.use_relative_pos_emb = use_relative_pos_emb
        self.use_relative_pos_emb = use_relative_pos_emb

        self.model_dim = dim
        # self.feedforward_dim = feedforward_dim
        self.time_size = num_patches
        self.out_features = out_features
        self.build_relative_position_embed()
        self.pos_encoding = PositionalEncoding(self.out_features)
        self.dropout = nn.Dropout(p=0.25)
        self.multi_y = multi_y
        if multi_y is None:
            self.multi_y = ['tf']

        # transformer_layer = nn.TransformerEncoderLayer(
        #     d_model=self.model_dim,
        #     nhead=nheads,
        #     dim_feedforward=self.feedforward_dim,
        #     dropout=dropout
        # )
        # self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_encoder_layers)
        self.print = False
        self.use_g_mid_print = False
        self.channel_embed = nn.Parameter(torch.randn(1, self.channels, dim) * .02)

        self.pool = pool
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.pos_fft_drop = nn.Dropout(p=dropout)
        self.rel_pos_bias = None
        self.mask = torch.tensor([0, 1, 2, 3])
        self.use_multiway = use_multiway
        rank_zero_info(f'Global Transformer Using Type{use_multiway}')
        if use_multiway == 'one_stream':
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim, num_heads=nheads, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer,
                    init_values=init_values, all_num_relative_distance=self.all_num_relative_distance,
                    use_relative_pos_emb=self.use_relative_pos_emb)
                for i in range(num_encoder_layers)])
        elif use_multiway == 'multiway':
            self.tfffn_start_layer_index = num_encoder_layers // 2
            max_time_len = (self.time_size*self.channels-1) if self.use_global_fft is False else (self.time_size*2*self.channels - 1)
            self.blocks = nn.ModuleList(
                [
                    multiway_transformer.Block(
                        dim=dim,
                        num_heads=nheads,
                        mlp_ratio=4,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rate[i],
                        norm_layer=norm_layer,
                        with_tfffn=(i >= self.tfffn_start_layer_index),
                        layer_scale_init_values=init_values,
                        max_time_len=max_time_len,
                        time_only=False,
                        fft_only=False,
                        itc=1 if len(self.multi_y) > 1 else 0,
                        itm=1 if len(self.multi_y) > 1 else 0,
                        use_relative_pos_emb=self.use_relative_pos_emb,
                        all_num_relative_distance=self.all_num_relative_distance,
                        num_patches=max_time_len+1
                    )
                    for i in range(num_encoder_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim, num_heads=nheads, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer,
                    init_values=init_values, all_num_relative_distance=self.all_num_relative_distance,
                    use_relative_pos_emb=self.use_relative_pos_emb)
                for i in range(num_encoder_layers)])
            self.fft_blocks = nn.ModuleList([
                Block(
                    dim=dim, num_heads=nheads, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer,
                    init_values=init_values, all_num_relative_distance=self.all_num_relative_distance,
                    use_relative_pos_emb=self.use_relative_pos_emb)
                for i in range(num_encoder_layers)])
            self.time_proj = nn.Linear(dim, dim)
            self.fft_proj = nn.Linear(dim, dim)

        self.norm = nn.Identity() if pool != 'attn' else norm_layer(dim)
        # self.time_norm = self.norm if use_multiway == 'multiway' else norm_layer(dim)
        # self.fft_norm = self.norm if use_multiway == 'multiway' else norm_layer(dim)

        if self.use_all_label == 'all':
            # self.pooler = heads.Attn(hidden_size=self.model_dim, out_size=self.model_dim*3, double=False,channels=self.time_size,
            #                          reshape=False)
            pass
            # self.pooler = heads.Attn(hidden_size=self.model_dim, out_size=self.model_dim, channels=self.time_size,
            #                          reshape=True)
            # if len(self.multi_y) > 1:
            #     for name in self.multi_y:
            #         if name == 'tf':
            #             continue
            #         setattr(self, f'pooler_{name}', heads.Attn(hidden_size=self.model_dim, out_size=self.model_dim,
            #                                                    channels=self.time_size, reshape=False, double=False))
            # self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            # self.w_at = nn.Linear(self.model_dim, 1, bias=False)
        elif pool == 'attn':
            # self.predict_token = nn.Parameter(torch.zeros(1, 1, self.out_features))
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)
        # self.use_global_fft = use_global_fft
        if self.use_global_fft:
            self.token_type_embed = nn.Embedding(3, self.model_dim)
        else:
            self.token_type_embed = nn.Embedding(2, self.model_dim)

        self.fft_transform = FFT_Transform()

    def build_relative_position_embed(self, modality=2):
        if not self.use_relative_pos_emb:
            self.relative_position_embed = False
            self.relative_position_index = None
            return
        rank_zero_info('*********Using relative_position_embed*********')
        channels = self.channels
        patch_size = self.time_size
        if self.use_global_fft:
            modality = 3
        rpe_num_patches = channels * modality
        self.all_num_relative_distance = (patch_size * 2 - 1) * channels * modality + rpe_num_patches * (
                rpe_num_patches - 1)

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
        self.time_fft_relative_position_index = torch.cat(res_matrix, dim=0)
        assert (torch.max(
            self.time_fft_relative_position_index) == self.all_num_relative_distance - 1), f"{torch.max(self.time_fft_relative_position_index)}, {self.all_num_relative_distance}"
        # self.num_relative_distance = 2*(self.time_num_relative_distance+1)
        import numpy as np
        torch.set_printoptions(threshold=np.inf)
        rank_zero_info('Global RPE')
        rank_zero_info(self.time_fft_relative_position_index)
        rank_zero_info(f"time_fft_relative_position_index positive :{np.where(self.time_fft_relative_position_index<0)}")

    def forward(self, x_time, x_fft, batch, use_tf=False, use_g_mid=False, epoch_mask=None, training=True):

        use_time = (x_time is not None)
        use_fft = (x_fft is not None)
        use_tf = use_tf
        if use_tf is True:
            assert (x_time is not None)
            assert (x_fft is not None)
        # print("usetf: ", use_tf)
        if use_time:
            x_time = self.pos_drop(self.pos_encoding(x_time))
            # x_time = x_time.reshape(x_time.shape[0], -1, self.model_dim)
            # x_time = rearrange(x_time, "B T C D -> B (C T) D")
            x_time = x_time + self.token_type_embed(torch.zeros((x_time.shape[0], x_time.shape[1]), dtype=torch.long,
                                                      device=x_time.device))
        if use_fft:
            x_fft = self.pos_drop(self.pos_encoding(x_fft))
            # x_fft = rearrange(x_fft, "B T C D -> B (C T) D")
            # x_fft = x_fft.reshape(x_fft.shape[0], -1, self.model_dim)
            x_fft = x_fft + self.token_type_embed(torch.ones((x_fft.shape[0], x_fft.shape[1]), dtype=torch.long,
                                                                device=x_fft.device))
        # rank_zero_info(f"x_time shape:{x_time.shape}")
        # rank_zero_info(f"x_fft shape:{x_fft.shape}")
        if use_tf is True:
            x = torch.cat([x_time, x_fft], dim=1)
        elif use_time:
            x = x_time
        else:
            x = x_fft
        b, c, d = x.shape

        if self.use_global_fft:
            x_fft_fft = self.fft_conv_norm(torch.log(1 + torch.fft.fft(batch, dim=-1, norm='ortho').abs())[:, self.mask.to(x.device), :1200])
            if training:
                x_fft_fft = self.fft_transform(x_fft_fft)
            states = []
            for i in range(self.channels):
                states_temp = getattr(self, f"fft_conv_{i}")(x_fft_fft[:, i])
                states.append(states_temp)
            x_fft_embed = torch.stack(states, dim=1)
            x_fft_embed = rearrange(x_fft_embed, '(B T) C D -> B T C D', T=self.time_size)
            # x_fft_embed = self.fft_conv_norm(rearrange(self.fft_conv_1d(x_fft_fft), '(B T) C P -> B T C D', T=self.time_size))

            # return x_fft
            # x_fft_embed = self.fft_conv(x_fft_embed)
            # x_fft_embed = self.se(x_fft_embed)
            x_fft_embed = self.pos_encoding(x_fft_embed).reshape(x_fft_embed.shape[0], -1, self.model_dim)
            x_fft_embed = rearrange(x_fft_embed, "B (T C) D -> B (C T) D", T=self.time_size)
            x_fft_embed = x_fft_embed + self.token_type_embed(torch.ones((x_fft_embed.shape[0], x_fft_embed.shape[1]), dtype=torch.long,
                                                                    device=x.device)*2)
            x = torch.cat([x, x_fft_embed], dim=1)

        repeat_nums = 1
        if use_tf is True:
            repeat_nums = 2
        if self.use_global_fft:
            repeat_nums += 1

        # epoch_mask = epoch_mask.repeat_interleave(self.channels, dim=-1)
        # epoch_mask = epoch_mask.repeat(1, repeat_nums)
        # if 0 in before:
        #     import numpy as np
        #     torch.set_printoptions(threshold=np.inf)
        #     rank_zero_info (f"epoch_mask before:{before}")
        #     rank_zero_info (f"epoch_mask after{epoch_mask}, eq: {before==epoch_mask}")

        if hasattr(self, "channel_embed"):
            channel_embed = self.channel_embed.repeat_interleave(self.time_size, dim=1)
            x = x + channel_embed.repeat(1, repeat_nums, 1)

            # x = x + self.channel_embed.repeat(1, repeat_nums*self.time_size, 1)
        if not self.print:
            # rank_zero_info(f'x.shape: {b},{c},{d}----mid={mid}')
            rank_zero_info(f'transformer x.shape: {x.shape[0]},{x.shape[1]},{x.shape[2]}')
            rank_zero_info(f"epoch_mask: {epoch_mask.shape}, use_multiway: {self.use_multiway}")
            rank_zero_info(f"epoch_mask: {self.channel_embed.shape}, repeat:{repeat_nums*self.time_size}")

            self.print = True
        if use_tf is True:
            modality_type = 'tf'
        elif use_time:
            modality_type = 'time'
        else:
            modality_type = 'fft'
        if self.use_multiway == 'multiway':
            if len(self.multi_y) == 1:
                for blk in self.blocks:
                    x = blk(x, modality_type='tf',
                            relative_position_index=self.time_fft_relative_position_index)
            else:
                for blk in self.blocks:
                    x = blk(x, modality_type=modality_type,
                            relative_position_index=None)
        elif self.use_multiway=='one_stream':
            if len(self.multi_y) == 1:
                for blk in self.blocks:
                    x = blk(x, relative_position_index=self.time_fft_relative_position_index)
            else:
                raise NotImplementedError
        else:
            x_time = x[:, :self.time_size * self.channels]
            for blk in self.blocks:
                x_time = blk(x_time, relative_position_index=self.time_fft_relative_position_index)
            x_time = self.time_norm(x_time)
            x_fft = x[:, self.time_size * self.channels:]
            for i, blk in enumerate(self.fft_blocks):
                x_fft = blk(x_fft,
                            relative_position_index=self.time_fft_relative_position_index)
            x = torch.cat([self.time_proj(x_time), self.fft_proj(x_fft)], dim=1)

        x = self.norm(x)
        x = self.dropout(x)
        if self.use_all_label == 'all' and use_tf:
            assert self.pool is None
            time_res = x[:, :self.time_size]
            fft_res = x[:, self.time_size:]
            x = torch.cat([time_res, fft_res], dim=-1)
            # time_res = rearrange(time_res, "B (C T) D -> (B T) C D", T=self.time_size)
            if self.use_global_fft:
                fft_res = x[:, (self.time_size*self.channels):2*(self.time_size*self.channels)]
                # fft_res = rearrange(fft_res, "B (C T) D -> (B T) C D", T=self.time_size)

                fft_res = rearrange(fft_res, "B (C T) D -> B T (C D)", T=self.time_size)

                fft_global = x[:, 2*(self.time_size*self.channels):]
                # fft_global = rearrange(fft_global, "B (C T) D -> (B T) C D", T=self.time_size)

                fft_global = rearrange(fft_global, "B (C T) D -> B T (C D)", T=self.time_size)
                x = torch.cat([time_res, fft_res, fft_global], dim=-1)
                # x = torch.cat([time_res, fft_res, fft_global], dim=1)
                # x = self.pooler(x)
                x = x.reshape(-1, self.channels*3*d)
                # x = x.reshape(-1, 3*d)

            else:
                fft_res = x[:, self.time_size:]
                x = torch.cat([time_res, fft_res], dim=-1)
                x = x.reshape(-1, 2 * d)
            # # res = self.pooler(torch.cat([time_res, fft_res], dim=1), time_split=time_res.shape[1])
            # mid = int(self.time_size // 2 + 1)
        elif use_tf is True:
            if self.pool == 'mean':
                x = self.fc_norm(x.mean(dim=1))
            elif self.pool == 'last':
                x = self.fc_norm(x[:, int(x.size(0) // 2) + 1])
            elif self.pool == 'attn':
                a_states = torch.tanh(self.w_ha(x))
                alpha = torch.softmax(self.w_at(a_states), dim=1).view(x.size(0), 1, x.size(1))
                x = self.fc_norm(torch.bmm(alpha, a_states).view(x.size(0), -1))
            # elif self.pool == 'all':
            #     x = x.reshape(-1, d)

        ret = {}
        if use_time and not use_tf:
            # x_time = x_time.reshape(b * self.time_size, -1, d)
            # x_time = getattr(self, 'pooler_time')(x_time)
            x = x.reshape(-1, d)
        if use_fft and not use_tf:
            x = self.norm(x)
            # x_fft = x_fft.reshape(b * self.time_size, -1, d)
            # x_fft = getattr(self, 'pooler_fft')(x_fft)
            x = x.reshape(-1, d)
        ret.update({modality_type: x})

        return ret

class EventDecoderTransformer(nn.Module):
    def __init__(self, hidden_size, patch_size, weight=None, decoder_depth=6, enc_dim=512, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, dpr=None, norm_layer=nn.LayerNorm, num_layers=1,
                 seq_len=10, multi=True, n_classes=200, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.first = nn.Sequential(nn.Linear(hidden_size, enc_dim),
                                   nn.LayerNorm(enc_dim))
        if multi:
            self.token_type_embeddings = nn.Embedding(2, enc_dim)
        self.blocks = nn.ModuleList()
        if dpr is not None:
            drop_path_rate = [x.item() for x in (torch.linspace(0.00, dpr, decoder_depth))]
        else:
            drop_path_rate = None
        for i in range(decoder_depth):
            self.blocks.append(Block(dim=enc_dim,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=drop_path_rate[i] if dpr is not None else 0.0,
                                     norm_layer=norm_layer,
                                     ))
        self.last = nn.Linear(enc_dim * 2, patch_size)
        # self.last_act = nn.GELU()
        # self.predic_all = nn.Linear(2000, 2000)
        self.pe = PositionalEncoding(out_features=enc_dim)

    def forward_multimodal(self, x):
        time_c3, fft_c3 = x
        B = time_c3.shape[0]
        L = time_c3.shape[1]
        x = torch.cat([time_c3, fft_c3], dim=1)
        x = self.first(x)
        time_c3, fft_c3 = x[:, :L], x[:, L:]
        time_c3, fft_c3 = (
            self.pe(time_c3),
            self.pe(fft_c3)
        )
        x_embeds, fft_embeds = (
            time_c3 + self.token_type_embeddings(
                torch.zeros((B, time_c3.shape[1]), dtype=torch.long,
                            device=time_c3.device)),
            fft_c3 + self.token_type_embeddings(
                torch.ones((B, fft_c3.shape[1]), dtype=torch.long, device=fft_c3.device))
        )
        inputs = torch.cat([x_embeds, fft_embeds], dim=1)
        for block in self.blocks:
            inputs = block(inputs)
        x_embeds, fft_embeds = (
            inputs[:, :L],
            inputs[:, L:]
        )
        x = torch.cat([x_embeds, fft_embeds], dim=-1)
        # x = F.sigmoid(self.predic_all(self.last_act(self.last(x).reshape(B, -1))))
        x = F.sigmoid(self.last(x).reshape(B, -1))
        return x

    def forward_single(self, x):
        time_c3 = x
        B = time_c3.shape[0]
        L = time_c3.shape[1]
        x = self.first(x)
        time_c3 = x[:, :L]
        time_c3 = (
            self.pe(time_c3),
        )
        x_embeds = (
            time_c3)
        inputs = x_embeds
        for block in self.blocks:
            inputs = block(inputs)
        x_embeds = (
            inputs[:, :L],
        )
        x = x_embeds
        # x = F.sigmoid(self.predic_all(self.last_act(self.last(x).reshape(B, -1))))
        x = self.last(x).reshape(B, -1)
        return x

    def forward(self, x, multi=True):
        if multi is True:
            res = self.forward_multimodal(x)
        else:
            res = self.forward_single(x)
        return res