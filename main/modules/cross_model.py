from . import cross_attention
from . import vit
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange
import torch.nn.functional as F


class Cross_Attn_Spindle_Model(nn.Module):
    def __init__(self,
                 qdim,
                 kvdim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_cls_token=True,
                 drop_path=None,
                 num_queries=400,
                 decoder_depth=6,
                 act_layer=nn.GELU,
                 seq_len=2000,
                 ):
        super().__init__()
        self.pe = vit.PositionalEncoding(out_features=qdim)
        self.decoder = nn.ModuleList()
        if drop_path is not None:
            drop_path_rate = [x.item() for x in (torch.linspace(0.00, drop_path, decoder_depth))]
        else:
            drop_path_rate = None
        for i in range(decoder_depth):
            self.decoder.append(cross_attention.DecoderBlock(qdim, kvdim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                             attn_drop=attn_drop, drop_path=drop_path_rate[i],
                                                             act_layer=act_layer),)
        self.last = nn.Linear(qdim, seq_len//num_queries)
        self.last_act = act_layer()
        self.prob = nn.Linear(seq_len, seq_len)
        self.num_queries = num_queries
        self.qdim = qdim

    def forward(self, memory):
        time_c3, fft_c3 = memory
        memory = torch.cat([time_c3, fft_c3], dim=1)
        B, L, C = memory.shape
        q = torch.zeros(self.num_queries, self.qdim).unsqueeze(0).repeat(B, 1, 1).to(memory.device)
        x = self.pe(q)
        for layer in self.decoder:
            x, attn = layer(x, memory)
        x = self.last_act(self.last(x)).reshape(B, -1)
        x = F.sigmoid(self.prob(x))
        return x
