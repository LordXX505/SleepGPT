""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

"""
import os

import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from main.transforms import FFT_Transform
import pynvml


# from visualizer import get_local

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
            use_cb=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.use_cb = use_cb
        # if self.use_cb:
        #     self.lam = nn.Parameter(torch.ones(1, num_patches) * 0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.use_cb:
            x = 0.5 * x + 0.5 * x.mean(dim=1, keepdim=True)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            seq_len=15,
            use_relative_pos_emb=True,
            all_num_relative_distance=-1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_relative_pos_emb = use_relative_pos_emb
        self.num_relative_distance = seq_len * 2 + 2
        self.use_relative_pos_emb = use_relative_pos_emb
        if use_relative_pos_emb:
            assert all_num_relative_distance != -1
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(all_num_relative_distance, num_heads))

    def get_rel_pos_bias(self, relative_position_index):  # 196, 196
        # rank_zero_info(relative_position_index)
        # rank_zero_info(self.relative_position_bias_table)
        relative_position_bias = F.embedding(
            relative_position_index.long().to(self.relative_position_bias_table.device),
            self.relative_position_bias_table)  # out = [196, 196, 144], tabele=[1126, 144], co=237,237,1444
        all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, x, y
        return all_relative_position_bias

    # @get_local('attn')
    def forward(self, x, mask=None, relative_position_bias=None, relative_position_index=None):
        B, N, C = x.shape
        # print(x.shape)
        # pynvml.nvmlInit()
        # unit = 1024*1024*1024
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # print("*******attn********")
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # print('x.shape,mask.shape, self.qkv.weight')
        # print(x.shape, mask.shape, self.qkv.weight.shape)
        # print('x is none:', torch.isnan(x).sum())
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # print('qkv shape', qkv.shape)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # print('qkv is none:', torch.isnan(qkv).sum())
        # print('q, k, v shape', qkv.shape)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # print("*******attn____ q, k, v ********")
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape, mask.shape)
        # print('attn: q: k:', torch.isnan(attn).sum(), torch.isnan(q).sum(), torch.isnan(k).sum())
        # print(attn.shape)
        # print(torch.isnan(attn))
        # print(relative_position_index)
        if self.use_relative_pos_emb:
            attn = attn + self.get_rel_pos_bias(relative_position_index).unsqueeze(0)
        elif relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            # print('bool mask:')
            # for i in mask:
            #     print((i == 0).sum())
            assert mask.shape[-1] == attn.shape[-1], f"mask: {mask.shape}, attn: {attn.shape}"
            mask = mask.bool()
            # print(mask)
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        # print("*******attn____(attn.masked_fill********")
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        attn = attn.softmax(dim=-1).type_as(x)
        # print('softmaxattn:', attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("*******attn____last********")
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        return x
1

class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            with_tfffn=False,
            layer_scale_init_values=0.1,
            max_time_len=40,
            time_only=False,
            fft_only=False,
            itc=False,
            itm=False,
            use_relative_pos_emb=True,
            all_num_relative_distance=-1,
            use_cb=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            seq_len=max_time_len,
            use_relative_pos_emb=use_relative_pos_emb,
            all_num_relative_distance=all_num_relative_distance,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_tf = None

        if with_tfffn:
            self.mlp_tf = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                use_cb=use_cb
            )
            self.norm2_tf = norm_layer(dim)
            if itc or itm:
                print('While Block using with_tfffn, mlp_time and mlp_fft also use.Check it carefully')
                self.norm2_time = norm_layer(dim)
                self.mlp_time = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    use_cb=use_cb

                )
                self.norm2_fft = norm_layer(dim)
                self.mlp_fft = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    use_cb=use_cb

                )
        else:
            if not fft_only:
                self.norm2_time = norm_layer(dim)
                self.mlp_time = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    use_cb=use_cb

                )
            if not time_only:
                self.norm2_fft = norm_layer(dim)
                self.mlp_fft = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    use_cb=use_cb
                )
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0

        self.max_time_len = max_time_len
        rank_zero_info(f'max_time_len: {self.max_time_len}')

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None, relative_position_index=None, not_use_tf=False):
        # unit = 1024*1024*1024
        # print("*******blk********")
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias,
                                     relative_position_index=relative_position_index))
        # print('blk x', torch.isnan(x).sum())
        # print("*******blk_forward********")
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        if modality_type == "time":
            x = x + self.drop_path(self.gamma_2 * self.mlp_time(self.norm2_time(x)))
        elif modality_type == "fft":
            x = x + self.drop_path(self.gamma_2 * self.mlp_fft(self.norm2_fft(x)))
        else:
            if self.mlp_tf is None or not_use_tf:
                # print('self.max_time_len', self.max_time_len)
                x_time = x[:, :1 + self.max_time_len]
                x_fft = x[:, 1 + self.max_time_len:]
                x_time = x_time + self.drop_path(self.gamma_2 * self.mlp_time(self.norm2_time(x_time)))
                # print('blk2 x_time', torch.isnan(x_time).sum())

                x_fft = x_fft + self.drop_path(self.gamma_2 * self.mlp_fft(self.norm2_fft(x_fft)))
                # print('blk2 x_fft', torch.isnan(x_fft).sum())
                x = torch.cat([x_time, x_fft], dim=1)
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp_tf(self.norm2_tf(x)))
        # print('blk2 x', torch.isnan(x).sum())
        # print("*******blk_last********")
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("device: ", x.device)
        # print("Memory Total: ", meminfo.total/unit)
        # print("Memory Free: ", meminfo.free/unit)
        # print("Memory Used: ", meminfo.used/unit)
        return x


class PatchEmbed(nn.Module):

    def __init__(
            self,
            epoch_duration=30,
            fs=100,
            patch_size=200,
            in_chans=57,
            embed_dim=768,
            no_patch_embed_bias=False,
            max_channels=57,
            choose_channels=None,
            fft_only=False,
            time_only=False,
            use_stft=False,
    ):
        super().__init__()
        time_samps = epoch_duration * fs
        num_patches = time_samps // patch_size
        self.epoch_size = time_samps
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.choose_channels = choose_channels
        self.max_channels = max_channels
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if not fft_only:
            self.proj = nn.Conv1d(
                in_chans,
                in_chans*embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False if no_patch_embed_bias else True,
                groups=in_chans
            )
            w = self.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if not time_only and not use_stft:
            # self.fft_proj = torch.nn.Conv2d(in_channels=in_chans, out_channels=embed_dim,
            #                                 kernel_size=(2, 100), stride=(2, 100))
            # w_fft = self.fft_proj.weight.data
            # torch.nn.init.xavier_uniform_(w_fft.view([w_fft.shape[0], -1]))
            self.fft_proj = nn.Conv2d(
                in_channels=in_chans,
                out_channels=in_chans*embed_dim,
                kernel_size=(2, 100),
                stride=(2, 100),
                bias=False if no_patch_embed_bias else True,
                groups=in_chans
            )
            w_fft = self.fft_proj.weight.data
            torch.nn.init.xavier_uniform_(w_fft.view([w_fft.shape[0], -1]))

    def forward(self, x=None, time=True, fft=True):
        B, C, S = x[0].shape
        assert S == self.epoch_size, f"Input image size ({S}) doesn't match model {self.epoch_size}."
        if time and fft:
            assert len(x) == 2
            assert C == self.max_channels
            time_proj = rearrange(self.proj(x[0]), 'B (C D) P -> B (C P) D', C=self.in_chans)
            fft_proj =  rearrange(self.fft_proj(x[1]).squeeze(-1), 'B (C D) P -> B (C P) D', C=self.in_chans)
        # print('self.fft_proj.weight', torch.isnan(self.fft_proj.weight).sum())
        # os.makedirs('./result/', exist_ok=True)
        # torch.save(self.state_dict(), './result/fft_proj.pt')

        # print('max(self.fft_proj.weight), min(self.fft_proj.weight)', torch.max(self.fft_proj.weight), torch.min(self.fft_proj.weight))
        # print('max(x[:, self.max_channels:]), min(x[:, self.max_channels:])', torch.max(x[:, self.max_channels:]), torch.min(x[:, self.max_channels:]))
        # print('torch.isnan(time_proj).sum(), torch.isnan(fft_proj).sum()', torch.isnan(time_proj).sum(),
        #       torch.isnan(fft_proj).sum())
        # print('self.fft_proj.weight.data', self.fft_proj.weight.data)
            res = [time_proj, fft_proj]
            res = torch.concatenate(res, dim=1)
            return res
        elif time:
            assert x is not None

            assert C == self.max_channels
            time_proj = rearrange(self.proj(x[0]), 'B (C D) P -> B (C P) D', C=self.in_chans)

            return time_proj
        else:
            len_ = len(x)
            assert len_ >= 0

            assert C == self.choose_channels.shape[0]
            fft_proj =  rearrange(self.fft_proj(x[len_-1]), 'B (C D) P -> B (C P) D', C=self.in_chans)

            return fft_proj


class MultiWayTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            patch_size=200,
            embed_dim=256,
            depth=10,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None,
            need_relative_position_embed=False,
            use_abs_pos_emb=True,
            layer_scale_init_values=0.1,
            tfffn_start_layer_index=10,
            use_mean_pooling=False,
            config=None,
            use_relative_pos_emb=False,
            all_num_relative_distance=-1
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            tfffn_start_layer_index (int): tf-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        super().__init__()
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        rank_zero_info("drop path rate: {}".format(drop_path_rate))
        # self.choose_channels = torch.tensor([4, 15, 16, 18, 22, 36, 38, 52])
        # self.choose_channels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        # self.choose_channels = torch.tensor([0, 1, 2, 3])  # c=4
        self.choose_channels = torch.tensor(torch.arange(config['random_choose_channels']))
        if config['actual_channels'] == 'shhs':
            self.actual_channels = torch.tensor(torch.arange(4))
        elif config['actual_channels'] == 'physio':
            self.actual_channels = torch.tensor([0, 1 ,2 ,3 ,4, 6])
        else:
            self.actual_channels = None
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_relative_pos_emb = use_relative_pos_emb
        self.need_relative_position_embed = need_relative_position_embed
        self.mask_ratio = config['mask_ratio']
        self.epoch_duration = config['epoch_duration']
        self.fs = config['fs']
        self.epoch_duration = config['epoch_duration']
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.max_channels = config['random_choose_channels']
        self.patch_embed = PatchEmbed(
            epoch_duration=self.epoch_duration,
            fs=self.fs,
            patch_size=patch_size,
            in_chans=self.max_channels,
            embed_dim=embed_dim,
            choose_channels=self.choose_channels,
            max_channels=self.max_channels,
            fft_only=config['fft_only'],
            time_only=config['time_only']
        )
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.tfffn_start_layer_index = tfffn_start_layer_index
        if config["time_only"] is True or config['fft_only'] is True:
            self.tfffn_start_layer_index = depth
            rank_zero_info(
                "Set tfffn_start_layer_index={} for {}-only pretraining".format(self.tfffn_start_layer_index,
                                                                                "time" if config[
                                                                                    "time_only"] else "fft"))
        if config['fft_only'] is not True:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)

        if config['time_only'] is not True:
            self.fft_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.fft_cls_token, std=0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * .02) if self.use_abs_pos_emb else None
        # self.fft_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * .02) if self.use_abs_pos_emb else None

        self.cls_token_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.use_abs_pos_emb else None
        # self.fft_cls_token_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.use_abs_pos_emb else None

        self.pos_drop = nn.Dropout(p=drop_rate)
        if config['loss_names']['mtm'] > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.mask_token, std=.02)

        self.channel_embed = nn.Parameter(torch.randn(1, self.choose_channels.shape[0], embed_dim) * .02)
        # self.fft_channel_embed = nn.Parameter(torch.randn(1, self.choose_channels.shape[0], embed_dim) * .02)
        self.hop_length = 100
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        rank_zero_info(f"dpr: {dpr}")
        max_time_len = len(self.actual_channels) if self.actual_channels is not None else self.max_channels
        self.max_time_len = self.num_patches*max_time_len
        rank_zero_info(f'choose max_time_len: {max_time_len}, self.max_time_len: {self.max_time_len}')

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_tfffn=(i >= self.tfffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_time_len=self.max_time_len,  # 57 * 15
                    time_only=config['time_only'],
                    fft_only=config['fft_only'],
                    itc=config['loss_names']['itc'] > 0,
                    itm=config['loss_names']['itm'] > 0,
                    use_cb=config['use_cb'],
                    use_relative_pos_emb=self.use_relative_pos_emb,
                    all_num_relative_distance=all_num_relative_distance
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        # if config['time_only']:
        #     self.decoder_norm = norm_layer(embed_dim)
        # if config['fft_only']:
        #     self.fft_norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token_pos_embed, std=.02)
        # trunc_normal_(self.channel_embed, std=.02)
        # print('init_weights', w_fft, torch.isnan(w_fft).sum())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def random_masking2(self, x, mask_ratio, attn_mask, mask_w):
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask_w.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        return x, mask_w

    def random_masking(self, x, mask_ratio, attn_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask_tokens = self.mask_token.repeat(x.shape[0], L - len_keep, 1)

        ids_keep = ids_shuffle[:, :len_keep]
        # print("ids_keep: ", ids_keep)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        mask = mask * attn_mask
        return x_masked, mask  # [N,L,D], [N, L]

    def time_embed(self, _x, attn_mask, mask=False, mask_w=None):
        ret = {}
        x = self.patch_embed(_x, time=True, fft=False)
        b, features, num_patches = x.shape[0], self.num_features, self.num_patches
        # x = x.permute(0, 2, 1)
        # x = torch.gather(x, dim=1,
        #                  index=self.choose_channels.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(b, 1, num_patches,
        #                                                                                             features).to(
        #                      x.device))
        # print(self.choose_channels.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(b, 1, num_patches,
        #                                                                                             features).shape)
        # print(x.shape)
        x = x.contiguous().view(b, -1, features)
        B, L, _ = x.shape
        x_time = x
        # masking: length -> length * mask_ratio
        if mask:
            assert mask_w is not None
            x_time, time_mask_patch = self.random_masking2(x_time, self.mask_ratio, attn_mask,
                                                           mask_w)  # [N,L_t,D], [N, L_t]
        else:
            time_mask_patch = None
        ret.update({'time_mask_patch': time_mask_patch})

        time_cls_tokens = self.cls_token.expand(B, -1, -1)
        x_time = torch.cat((time_cls_tokens, x_time), dim=1)
        ret.update({'x_len': x_time.shape[1]})
        if self.pos_embed is not None:
            x_time[:, 1:, ] = x_time[:, 1:, ] + self.pos_embed.repeat(1, self.max_channels, 1)
            x_time[:, 0, :] = x_time[:, 0, :] + self.cls_token_pos_embed
        x_time[:, 1:, ] = x_time[:, 1:, ] + self.channel_embed.repeat_interleave(self.num_patches, dim=1)
        x_time = self.pos_drop(x_time)
        ret.update({'x': x_time})
        return ret

    def fft_embed(self, _x, mask=False):
        ret = {}
        x = self.patch_embed(_x, time=False, fft=True)
        # b, features, num_patches = x.shape[0], self.num_features, self.num_patches
        # x = x.permute(0, 2, 1)
        B, L, _ = x.shape
        x_fft = x
        # masking: length -> length * mask_ratio
        if mask:
            raise NotImplementedError
            # x_fft, fft_mask_patch = self.random_masking(x_fft, self.mask_ratio)  # [N,L_t,D], [N, L_t]
        else:
            fft_mask_patch = None
        ret.update({'time_mask_patch': fft_mask_patch})

        x_fft_cls_tokens = self.fft_cls_token.expand(B, -1, -1)
        x_fft = torch.cat((x_fft_cls_tokens, x_fft), dim=1)
        ret.update({'x_len': x_fft.shape[1]})
        if self.pos_embed is not None:
            x_fft[:, 1:, ] = x_fft[:, 1:, ] + self.pos_embed.repeat(1, self.max_channels, 1)
            x_fft[:, 0, :] = x_fft[:, 0, :] + self.cls_token_pos_embed
        x_fft[:, 1:, ] = x_fft[:, 1:, ] + self.channel_embed.repeat_interleave(self.num_patches, dim=1)

        x_fft = self.pos_drop(x_fft)
        ret.update({'x': x_fft})

        return ret

    def embed(self, _x, attn_mask, mask, mask_w=None):
        ret = {}
        x = self.patch_embed(_x)
        # print('self.patch_embed(_x): ', torch.isnan(x).sum())
        # b, features, c, num_patches = x.shape[0], self.num_features, self.max_channels + self.choose_channels.shape[
        #     0], self.num_patches
        # x = x.permute(0, 2, 1)
        B, L, _ = x.shape
        start_idx = self.num_patches*self.max_channels
        x_time = x[:, :start_idx]
        x_fft = x[:, start_idx:]
        assert x_time.shape == x_fft.shape
        # masking: length -> length * mask_ratio
        if mask:
            assert mask_w is not None
            x_time, time_mask_patch = self.random_masking2(x_time, self.mask_ratio, attn_mask,
                                                           mask_w)  # [N,L_t,D], [N, L_t]
            x_fft, fft_mask_patch = self.random_masking(x_fft, 0, attn_mask,
                                                           )  # [N,L_t,D], [N, L_t]
        else:
            time_mask_patch = None
            fft_mask_patch = None
        ret.update({'time_mask_patch': time_mask_patch})
        ret.update({'fft_mask_patch': fft_mask_patch})
        fft_cls_tokens = self.fft_cls_token.expand(B, -1, -1)
        x_fft = torch.cat((fft_cls_tokens, x_fft), dim=1)

        time_cls_tokens = self.cls_token.expand(B, -1, -1)
        x_time = torch.cat((time_cls_tokens, x_time), dim=1)

        if self.pos_embed is not None:
            x_time[:, 1:, ] = x_time[:, 1:, ] + self.pos_embed.repeat(1, self.max_channels, 1)
            x_time[:, 0, :] = x_time[:, 0, :] + self.cls_token_pos_embed
            x_fft[:, 1:, ] = x_fft[:, 1:, ] + self.pos_embed.repeat(1, self.max_channels, 1)
            x_fft[:, 0, :] = x_fft[:, 0, :] + self.cls_token_pos_embed
        x_fft[:, 1:, ] = x_fft[:, 1:, ] + self.channel_embed.repeat_interleave(self.num_patches, dim=1)
        x_time[:, 1:, ] = x_time[:, 1:, ] + self.channel_embed.repeat_interleave(self.num_patches, dim=1)

        if self.actual_channels is not None:
            x_time_actual = rearrange(x_time[:, 1:, ], 'B (C P) D -> B C P D', C=self.max_channels)
            x_fft_actual = rearrange(x_fft[:, 1:, ], 'B (C P) D -> B C P D', C=self.max_channels)
            x_time_actual = rearrange(x_time_actual[:, self.actual_channels.to(x_time.device)],
                                      'B C P D -> B (C P) D', C=len(self.actual_channels))
            x_fft_actual = rearrange(x_fft_actual[:, self.actual_channels.to(x_time.device)],
                                     'B C P D -> B (C P) D', C=len(self.actual_channels))
            assert x_time_actual.shape == x_fft_actual.shape
            assert x_time_actual.shape[1] == len(self.actual_channels)*self.num_patches
            x_time = torch.cat([x_time[:, 0, :].unsqueeze(1), x_time_actual], dim=1)
            x_fft = torch.cat([x_fft[:, 0, :].unsqueeze(1), x_fft_actual], dim=1)

        x = torch.cat([x_time, x_fft], dim=1)
        assert x.shape[1] == self.max_time_len*2 + 2
        x = self.pos_drop(x)
        ret.update({'x': x})
        ret.update({'x_len': x_time.shape[1]})
        ret.update({'fft_len': x_fft.shape[1]})
        return ret

    def get_fft(self, _x: torch.Tensor, attn_mask: torch.Tensor, aug=False):
        res = []
        # ids_keep = self.choose_channels.to(_x.device)
        n_fft = 256
        hop_length = self.hop_length
        win_length = 200
        window = torch.hann_window(win_length, device=_x.device)
        x_fft = _x
        # x_fft = torch.gather(_x, dim=1, index=ids_keep.unsqueeze(0).unsqueeze(-1).repeat(_x.shape[0], 1,
        #                                                                                      self.epoch_duration
        #                                                                                  * self.fs))
        attn_mask_fft = attn_mask.clone()
        # print(' x_fft = torch.gathe', x_fft.shape)
        # for i in range(self.num_patches):
        #     res.append(torch.log(1 + torch.fft.fft(x_fft[:, :, self.patch_size * i:self.patch_size * (i + 1)],
        #                                            dim=-1, norm='ortho').abs()))
        for c in self.choose_channels:
            spec = torch.stft(x_fft[:, c], n_fft, hop_length, win_length, window, return_complex=False)
            magnitude = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)[:, :100, 1:]
            log_magnitude = 10 * torch.log10(magnitude + 1e-8)
            log_magnitude = log_magnitude.transpose(-2, -1)
            mean = log_magnitude.mean(dim=-1)
            std = log_magnitude.std(dim=-1)
            # print('stft: nan and sum')
            # print(c)
            # print(torch.isnan(log_magnitude).sum())
            # print(torch.isnan(mean).sum())
            # print(torch.isnan(std).sum())
            # print(std[torch.where(std<0.01)], torch.where(std<0.01))

            res.append((log_magnitude - mean.unsqueeze(-1)) / std.unsqueeze(-1))
            # print(torch.isnan((log_magnitude - mean.unsqueeze(-1)) / std.unsqueeze(-1)).sum())

        res = torch.stack(res, dim=1)
        if aug:
            ft = FFT_Transform()
            res = ft(res)
        # print('res = torch.concatenate(res, dim=-1)', torch.isnan(res).sum())
        return res, attn_mask_fft

class FilterbankShape(object):

    def lin_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a linear-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        #lowmel = self.hz2mel(lowfreq)
        #highmel = self.hz2mel(highfreq)
        #melpoints = np.linspace(lowmel,highmel,nfilt+2)
        hzpoints = torch.linspace(lowfreq,highfreq,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*hzpoints/samplerate)

        fbank = torch.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = torch.transpose(fbank)
        fbank.astype(torch.float32)
        return fbank

def backbone_base_patch200(pretrained=False, **kwargs):
    patch_size = kwargs.pop("patch_size", 200)
    model = MultiWayTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, tfffn_start_layer_index=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def backbone_large_patch200(pretrained=False, **kwargs):
    patch_size = kwargs.pop("patch_size", 200)
    model = MultiWayTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=16,
        mlp_ratio=4, qkv_bias=True, tfffn_start_layer_index=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def backbone_base_plus_patch200(pretrained=False, **kwargs):
    patch_size = kwargs.pop("patch_size", 100)
    model = MultiWayTransformer(
        patch_size=patch_size, embed_dim=1280, depth=16, num_heads=16,
        mlp_ratio=4, qkv_bias=True, tfffn_start_layer_index=13,
        use_abs_pos_emb=True, need_relative_position_embed=False,
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def backbone_base_plus_patch200_without(pretrained=False, **kwargs):
    patch_size = kwargs.pop("patch_size", 100)
    model = MultiWayTransformer(
        patch_size=patch_size, embed_dim=1280, depth=16, num_heads=16,
        mlp_ratio=4, qkv_bias=True, tfffn_start_layer_index=16,
        use_abs_pos_emb=True, need_relative_position_embed=False,
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
backbone_base_patch200 = backbone_base_patch200
backbone_large_patch200 = backbone_large_patch200
backbone_base_plus_patch200 = backbone_base_plus_patch200
backbone_base_plus_patch200_without = backbone_base_plus_patch200_without