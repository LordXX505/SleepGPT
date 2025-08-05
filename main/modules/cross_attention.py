import torch.nn as nn
import torch
from timm.models.layers import DropPath
from . import multiway_transformer

class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kvdim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        assert qdim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = qdim // num_heads
        self.scale = head_dim**-0.5
        self.use_cls_token = use_cls_token
        self.qkv_bias = qkv_bias

        self.q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.kv = nn.Linear(kvdim, qdim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(qdim, qdim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, query, memory, rel_pos_bias=None, attn_mask=None, with_mask=False
    ):
        # attn_mask: B x N
        B, N, C = query.shape
        L = memory.shape[1]

        q = (
            self.q(query)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k, v = (
            self.kv(memory)
            .reshape(B, L, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), -65504.0)

        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias

        attn_wo_softmax = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention = attn

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_wo_softmax


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        enc_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_cls_token=True,
        use_triton=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = multiway_transformer.Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_pos_emb=False,
            use_triton=use_triton
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            enc_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cls_token=use_cls_token,
        )

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = multiway_transformer.Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, memory, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x = self.attn(x, relative_position_bias=rel_pos_bias, mask=attn_mask)
        x = ori_x + self.drop_path(x)

        ori_x = x
        x = self.norm2(x)
        x, attn = self.cross_attn(
            x,
            memory,
            rel_pos_bias=rel_pos_bias,
            attn_mask=attn_mask,
        )

        x = ori_x + self.drop_path(x)

        ori_x = x
        x = self.drop_path(self.mlp(self.norm3(x)))
        x = ori_x + self.drop_path(x)

        return x, attn

