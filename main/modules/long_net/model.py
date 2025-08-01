import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from timm.models.layers import DropPath
from .customize_dialated_attn import DilatedAttention
from pytorch_lightning.utilities.rank_zero import rank_zero_info

# helpers
def exists(val):
    return val is not None

class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=1536,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# residual
# normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(
            max_seq_len, device=device, dtype=self.inv_freq.dtype
        )
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# Assuming necessary imports like RotaryEmbedding, SwiGLU, etc. are present


def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class ParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        dilated_ratios: int = 2,
        segment_lengths: int = 64,
        heads=8,
        ff_mult=4,
        dropout=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)

        # Attention and feedforward dimensions
        ff_inner_dim = dim * ff_mult
        self.heads = heads
        self.scale = dim_head**-0.5

        # Dilated attention for LongNet
        self.attn = DilatedAttention(
            dim=dim,
            num_heads=heads,
            segment_lengths=segment_lengths,
            dilated_ratios=dilated_ratios,
            dropout=dropout,
        )

        # Feed-forward network
        self.mlp = FeedForward(dim=dim, hidden_dim=ff_inner_dim, dropout=dropout)

        # Optional DropPath for regularization
        self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn, ot = self.attn(x, x, x)
        x = residual + self.drop_path(attn)

        # Residual 2: Feed-forward
        residual = x
        x = self.norm(x)
        ff = self.mlp(x)
        x = residual + self.drop_path(ff)

        return x


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult=4,
        dilated_ratios: int = 2,
        segment_lengths: int = 64,
        dropout=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.feedforward = (FeedForward(dim, dim, dropout=0.1),)

        self.layers = nn.ModuleList([
            ParallelTransformerBlock(
                dim=dim,
                dim_head=dim_head,
                dilated_ratios=dilated_ratios,
                segment_lengths=segment_lengths,
                heads=heads,
                ff_mult=ff_mult,
                dropout=dropout,
                drop_path=drop_path
            ) for _ in range(depth)
        ])

    def forward(self, x):
        x_list = []
        for block in self.layers:
            x = block(x)
            x_list.append(x)
        return x_list


# classes


class LongNetTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_patches,
        in_chans=1536,
        dim_head=64,
        heads=8,
        ff_mult=4,
        dilated_ratios: int = 2,
        segment_lengths: int = 64,
        dropout=0.1,
        drop_path=0.0,
        global_pool=False,
        sleep_stage=False,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=dim, )
        self.transformer = Transformer(
            dim, depth, heads, dim_head, ff_mult, dilated_ratios, segment_lengths, dropout=dropout, drop_path=drop_path
        )
        self.norm =  LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer('pos_embed', torch.zeros(1, num_patches + 1, dim), persistent=False)  # fixed sin-cos embedding
        self.global_pool = global_pool
        self.sleep_stage = sleep_stage
        rank_zero_info(f'in_chans: {in_chans}, embedim: {dim}, num_patches: {num_patches}')
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_list = self.transformer(x)
        outcomes = []
        for x_item in x_list:
            if self.global_pool:
                if self.sleep_stage is not True:
                    x_item = x_item[:, 1:, :].mean(dim=1)  # global average pooling
                    outcome = self.norm(x_item)
                else:
                    x_item_withoutcls = x_item[:, 1:, :]
                    x_item = rearrange(x_item_withoutcls, 'B (L P) D -> (B L) P D', P=15).mean(dim=1)
                    outcome = self.norm(x_item)
            else:
                x_item = self.norm(x_item)
                outcome = x_item[:, 0]
            outcomes.append(outcome)

        return outcomes


# autoregressive wrapper


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_thres=0.9,
        **kwargs,
    ):
        b, t, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_token = out == eos_token

                if is_eos_token.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]
        return out

    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        return F.cross_entropy(rearrange(logits, "b c n -> b n c"), x_labels)
