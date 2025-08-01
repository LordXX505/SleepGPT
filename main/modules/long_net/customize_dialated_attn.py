import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from .utils import XPOS, RelativePositionBias

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class DilatedAttention(nn.Module):
    def __init__(self, dim, num_heads, segment_lengths, dilated_ratios, dropout=0.1, device: str = "cpu",
                 flash_attention=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.segment_lengths = segment_lengths  # List of segment lengths
        self.dilated_ratios = dilated_ratios  # List of dilation ratios
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.flash_attention = flash_attention
        self.attention = None
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_module = torch.nn.Dropout(dropout)
        assert self.dilated_ratios is not None, f'{self.dilated_ratios}'

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None):
        if not self.flash_attention:
            q *= self.scale
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            # if attn_mask is not None:
            #     attn_weights = torch.nan_to_num(attn_weights)
            #     attn_mask = attn_mask.bool()
            #     # print(mask)
            #     attn_weights = attn_weights.masked_fill(~attn_mask[:, None, :], float("-inf"))

            if key_padding_mask is not None:
                attn_weights = rearrange(attn_weights, '(b h) t s -> b h t s', h=self.num_heads)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = rearrange(attn_weights, 'b h t s -> (b h) t s')

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            # Compute log-sum-exp for LSE
            attn_weights_logsumexp = torch.logsumexp(attn_weights, dim=-1)  # (batch_size * num_heads, seq_len)
            attn_weights_logsumexp = rearrange(attn_weights_logsumexp, '(b h) l -> b h l', h=self.num_heads)

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)
            # Convert attn_weights to probabilities
            # if torch.isnan(attn_probs).any():
            #     print("Warning: Found NaN in attention probabilities, setting them to 0.0.")
            #     attn_probs = torch.nan_to_num(attn_probs)
            #
            attn = torch.bmm(attn_probs, v)
            attn = rearrange(attn, '(b h) l d -> b l (h d)', h=self.num_heads)
        else:
            attn_output, attn_weights_logsumexp = self.attention(q, k, v)
            attn = rearrange(attn_output, 'b l h d -> b l (h d)')

        return attn, attn_weights_logsumexp

    def dense_to_sparse(self, x, ratio, attn_matrix):
        length = x.size(1)
        padding = padding_to_multiple_of(length, ratio)
        head_padding = padding_to_multiple_of(self.num_heads, ratio)
        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value=0.)
            attn_matrix = F.pad(x, (0, 0, 0, head_padding, 0, padding), value=0.)
        x = rearrange(x, 'b (l r1) (r2 h) d -> b l h d r1 r2', r1=ratio, r2=ratio)
        attn_matrix = rearrange(attn_matrix, 'b (l r1) (r2 h)  -> b l h  r1 r2', r1=ratio, r2=ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        attn_matrix = torch.diagonal(attn_matrix, offset=0, dim1=3, dim2=4)
        x = rearrange(x, 'b l h d r -> b l (r h) d')
        attn_matrix = rearrange(attn_matrix, 'b l h  r -> b l (r h) ')
        if head_padding > 0:
            x = x[:, :, :self.num_heads]
            attn_matrix = attn_matrix[:, :, :self.num_heads]
        return x, attn_matrix

    def sparse_to_dense(self, out, lse, ratio):
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value=0.)
            lse = F.pad(lse, (0, 0, 0, head_padding), value=-1e8)

        out = rearrange(out, 'b l (r h) d -> b l h d r', r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, 'b l h d r1 r2 -> b (r2 h) (l r1) d', r1=ratio, r2=ratio)

        lse = rearrange(lse, 'b (r h) l -> b l h r', r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse == 0, -1e8)
        lse = rearrange(lse, 'b l h r1 r2 -> b (r2 h) (l r1) 1', r1=ratio, r2=ratio)

        if head_padding > 0:
            out = out[:, :self.num_heads]
            lse = lse[:, :self.num_heads]

        return out, lse

    def gathering(self, x, dr, sl, offset=0):
        curr_x = x
        if offset > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, offset % sl, 0), value=0.)
        seq_len = curr_x.size(1)
        _sl = sl
        sl = min(sl, seq_len)
        padding = padding_to_multiple_of(seq_len, sl)
        attn_matrix = torch.ones(curr_x.size(0), curr_x.size(1), curr_x.size(2), device=curr_x.device)
        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value = 0.)
            attn_matrix = F.pad(attn_matrix, (0, 0,  0, padding), value = 0.)
        curr_x = rearrange(curr_x, 'b (n g) h d -> (b n) g h d', g=sl)
        attn_matrix = rearrange(attn_matrix, 'b (n g) h  -> (b n) g h ', g=sl)
        curr_x, attn_matrix  = self.dense_to_sparse(curr_x, dr, attn_matrix)
        curr_x = rearrange(curr_x, 'b l h d -> (b h) l d')
        attn_matrix = rearrange(attn_matrix, 'b l h  -> (b h) l ')
        return curr_x, attn_matrix

    def scattering(self, outs, lses, seq_len, bsz, offset=0):
        assert len(outs) == len(lses)
        assert len(outs) % len(self.dilated_ratios) == 0
        all_outs, all_lses = [], []
        drs = self.dilated_ratios
        if len(outs) > len(drs):
            drs = drs * (len(outs) // len(drs))

        for dr, o, lse in zip(drs, outs, lses):
            o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)
            o, lse = self.sparse_to_dense(o, lse, dr)
            o = rearrange(o, '(b n) h g d -> (b h) (n g) d', b=bsz)
            lse = rearrange(lse, '(b n) h g 1 -> (b h) (n g) 1', b=bsz)
            o = o[:, offset:offset + seq_len]
            lse = lse[:, offset:offset + seq_len]

            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0)
            max_lse = max_lse.max(0)[0]
            all_lses = [torch.exp(lse - max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = 0
        for o, lse in zip(all_outs, all_lses):
            out += o * lse.type_as(o)
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.num_heads)

        return out

    def forward(self, query, key, value, key_padding_mask=None, rel_pos=None):
        """
        Forward pass for dilated attention.
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.dim, f"Expected {self.dim}, but got {embed_dim}"

        # Project Q, K, V
        q = rearrange(self.q_proj(query), "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(self.k_proj(key), "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(self.v_proj(value), "b l (h d) -> b l h d", h=self.num_heads)
        # Generate padding mask if needed


        outs, lses = [], []

        # Iterate over segment lengths and dilation ratios
        for sl, dr in zip(self.segment_lengths, self.dilated_ratios):
            # Pad to match segment length
            ki, attn_mask = self.gathering(k, dr, sl, offset=0,)
            vi, attn_mask = self.gathering(v, dr, sl, offset=0,)
            qi, attn_mask = self.gathering(q, dr, sl, offset=0,)

            out, lse = self.attention_ops(qi, ki, vi, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                          rel_pos=rel_pos)

            outs.append(out)
            lses.append(lse)

        attn = self.scattering(outs, lses, tgt_len, bsz, offset=0,)
        attn = self.out_proj(attn)
        return attn, None