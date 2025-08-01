import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from . import cross_attention
from . import FPN
from . import vit
from . import cross_model

def parse_layer_range(layer_range_str):
    """
    Parse a string defining a layer range into a list of integers.

    Args:
    -----
    layer_range_str: str
        String defining the layer range, e.g., "5-11" or "4-6".

    Returns:
    --------
    list[int]:
        List of integers representing the selected layers.
    """
    if '-' in layer_range_str:
        start, end = map(int, layer_range_str.split('-'))
        return list(range(start, end + 1))  # Include the end layer
    else:
        raise ValueError(f"Invalid format for layer range: {layer_range_str}")

class Pooler(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, out_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(input=x.clamp(min=eps).pow(p), kernel_size=3).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Conv_embed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, channels=4, reshape=True, use_gem=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels

        self.reshape = reshape
        self.out_channels = out_channels
        self.use_gem = use_gem
        kernel_size = kernel_size
        if use_gem:
            self.gem = GeM()
            kernel_size = kernel_size // 3
        self.conv = nn.Conv1d(in_channels=in_channels * self.channels, out_channels=self.channels * out_channels,
                              kernel_size=kernel_size, groups=self.channels)

    def forward(self, x):
        b, c, d = x.shape
        if self.reshape:
            x = rearrange(x, "B (C P) D -> B (C D) P", C=self.channels)
        if self.use_gem:
            x = self.gem(x)
        x = self.conv(x).reshape(b, self.channels, -1)
        assert x.shape == (b, self.channels, self.out_channels)
        return x


class Attn(nn.Module):
    def __init__(self, hidden_size, out_size, channels=4, reshape=False, return_alpha=False,
                 double=True, channel_wise=False):
        super().__init__()

        self.reshape = reshape
        self.channels = channels
        self.double = double
        self.channel_wise = channel_wise

        self.fc_norm = nn.LayerNorm(eps=1e-6, normalized_shape=out_size)
        self.return_alpha = return_alpha
        if self.reshape and self.double:
            self.hidden_size = hidden_size * 2
            self.out_size = out_size * 2
        else:
            self.hidden_size = hidden_size
            self.out_size = out_size
        if channel_wise is False:
            self.w_ha = nn.Linear(self.hidden_size, self.out_size, bias=True)
            self.w_at = nn.Linear(self.out_size, 1, bias=False)
        else:
            for i in range(channels):
                setattr(self, f"w_ha_{i}", nn.Linear(self.hidden_size, self.out_size, bias=True))
                setattr(self, f"w_at_{i}", nn.Linear(self.out_size, 1, bias=False))

    def forward(self, x, time_split=None, attn_mask=None):
        if time_split == -1:
            b, c, p, d = x.shape
            assert c == self.channels
        else:
            b, c, d = x.shape
        softdim = 1
        if self.reshape is True:
            assert time_split is not None
            if self.channel_wise:
                # x_time = x[:, :time_split].reshape(b, self.channels, -1, d)
                # x_fft = x[:, time_split:].reshape(b, self.channels, -1, d)
                x = x.reshape(b, self.channels, -1, d)
                softdim = 1
            else:
                x_time = x[:, :time_split].reshape(b * self.channels, -1, d)
                x_fft = x[:, time_split:].reshape(b * self.channels, -1, d)
                x = torch.cat([x_time, x_fft], dim=-1)
                softdim = 2
        if self.channel_wise:
            a_states = []
            alpha = []
            assert x.shape[1] == self.channels
            for i in range(self.channels):
                a_states_temp = torch.tanh(getattr(self, f"w_ha_{i}")(x[:, i]))
                alpha_temp = torch.softmax(getattr(self, f"w_at_{i}")(a_states_temp), dim=softdim).view(x.size(0), 1,
                                                                                                        -1)
                a_states.append(a_states_temp)
                alpha.append(alpha_temp)
            a_states = torch.stack(a_states, dim=1)
            alpha = torch.stack(alpha, dim=1)
            assert a_states.shape == (b, c, p, self.out_size), f"a_states.shape:{a_states.shape}, x.shape:{x.shape}"
            a_states = a_states.reshape(b * self.channels, -1, self.out_size)
            alpha = alpha.reshape(b * self.channels, alpha.shape[2], alpha.shape[3])
        else:
            a_states = torch.tanh(self.w_ha(x))
            attn_matrix = self.w_at(a_states)
            # print(f'head attn shape: {attn_matrix.shape}, {attn_mask.shape}')
            attn_matrix = attn_matrix.masked_fill(~attn_mask.unsqueeze(-1).bool(), float("-inf"))
            alpha = torch.softmax(attn_matrix, dim=softdim).view(x.size(0), 1, -1)
        if self.return_alpha:
            return alpha
        if self.reshape is True:
            assert self.channel_wise is False
            x = torch.bmm(alpha, a_states).view(b, self.channels, -1)
        else:
            x = torch.bmm(alpha, a_states).view(-1, self.channels, self.out_size)
        if self.reshape is True:
            x = self.fc_norm(x.reshape(b, self.channels, 2, -1))
            x_time = x[:, :, 0]
            x_fft = x[:, :, 1]
            assert x_time.shape == (b, self.channels, self.out_size // 2), f"{x_time.shape}"
            assert x_fft.shape == (b, self.channels, self.out_size // 2), f"{x_fft.shape}"
            if self.return_alpha:
                return x_time, x_fft, alpha
            else:
                return x_time, x_fft
        else:
            x = self.fc_norm(x)
            if self.return_alpha:
                return x, alpha
            else:
                return x


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class ITCHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


class Event_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.GELU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.norm = nn.LayerNorm(h[-1])
        self.act = act_layer()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(self.norm(x))
        return F.sigmoid(x)


class Event_Head(nn.Module):
    def __init__(self, hidden_size, patch_size, weight=None, decoder_depth=6, enc_dim=512, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, dpr=None, norm_layer=nn.LayerNorm, num_layers=1,
                 seq_len=10, num_queries=400, FPN_resnet=False,
                 Use_FPN=None):
        super().__init__()
        # print(f"h:{h}")
        self.layers = num_layers
        self.Use_FPN = Use_FPN
        if Use_FPN == 'Trans':
            self.decoder = vit.EventDecoderTransformer(hidden_size, patch_size, weight=None,
                                                       decoder_depth=decoder_depth,
                                                       enc_dim=enc_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                       qkv_bias=qkv_bias, drop_rate=drop_rate,
                                                       attn_drop_rate=attn_drop_rate, dpr=dpr,
                                                       norm_layer=norm_layer, num_layers=num_layers, seq_len=seq_len
                                                       )
        elif Use_FPN == 'Cross':
            assert seq_len % num_queries == 0, f'seq_len%num_queries is not 0, seq_len: {seq_len}, num_queries: {num_queries}'
            self.decoder = cross_model.Cross_Attn_Event_Model(enc_dim, kvdim=hidden_size, num_heads=num_heads,
                                                              qkv_bias=qkv_bias,
                                                              seq_len=seq_len, drop_path=dpr,
                                                              decoder_depth=decoder_depth, num_queries=num_queries)
        elif Use_FPN == 'FPN':
            self.decoder = FPN.FPN(depth=decoder_depth, resnet=FPN_resnet)
        elif Use_FPN == 'Swin':
            self.decoder = FPN.FPN(depth=decoder_depth, resnet=FPN_resnet)
        else:
            self.decoder = Event_MLP(input_dim=hidden_size * 2, hidden_dim=enc_dim, output_dim=patch_size,
                                     num_layers=decoder_depth)

    def forward(self, x):
        raise NotImplementedError


class Apnea_Head(Event_Head):
    def __init__(self, hidden_size, patch_size, decoder_depth=6, enc_dim=512, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, dpr=None, norm_layer=nn.LayerNorm, num_layers=1,
                 seq_len=10, Use_FPN=None):
        super().__init__(hidden_size=hidden_size, patch_size=patch_size, weight=None, decoder_depth=decoder_depth,
                         enc_dim=enc_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, dpr=dpr,
                         norm_layer=norm_layer, num_layers=num_layers,
                         seq_len=seq_len,
                         Use_FPN=Use_FPN)

    def forward(self, x):
        # print(f"x.shape:{x.shape}")
        time_c3, fft_c3 = x
        B = time_c3.shape[0]
        if self.Use_FPN == 'FPN' or self.Use_FPN == 'MLP':
            x = torch.cat([time_c3, fft_c3], dim=-1)
            x = self.decoder(x)
        elif self.Use_FPN == 'Cross':
            x = self.decoder(x)
            return x
        else:
            x = self.decoder(x)
        x = x.reshape(B, -1)
        return x


class Spindle_Head(Event_Head):
    def __init__(self, hidden_size, patch_size, weight=None, decoder_depth=6, enc_dim=512, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, dpr=None, norm_layer=nn.LayerNorm, num_layers=1,
                 seq_len=10, num_queries=400, FPN_resnet=False,
                 Use_FPN=None):
        super().__init__(hidden_size=hidden_size, patch_size=patch_size, weight=None, decoder_depth=decoder_depth,
                         enc_dim=enc_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, dpr=dpr,
                         norm_layer=norm_layer, num_layers=num_layers,
                         seq_len=seq_len,
                         Use_FPN=Use_FPN)
    def forward(self, x):
        # print(f"x.shape:{x.shape}")
        time_c3, fft_c3 = x
        B = time_c3.shape[0]
        if self.Use_FPN == 'FPN' or self.Use_FPN == 'MLP':
            x = torch.cat([time_c3, fft_c3], dim=-1)
            x = self.decoder(x)
        elif self.Use_FPN == 'Cross':
            x = self.decoder(x)
            return x
        else:
            x = self.decoder(x)
        x = x.reshape(B, -1)
        return x


class Stage_Head(nn.Module):
    def __init__(self, hidden_size, weight=None):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        x = self.fc(x)
        return x


class Masked_decoder(nn.Module):
    def __init__(self, hidden_size, patch_size, num_patch, channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_patch = num_patch
        # self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        # # self.activation = nn.ReLU()
        # self.activation = nn.GELU()
        # self.LayreNorm = nn.LayerNorm(hidden_size)
        # self.decoder = nn.Linear(hidden_size, patch_size, bias=True)
        self.decoder = nn.Conv1d(in_channels=hidden_size * self.channels,
                                 out_channels=patch_size * self.channels,
                                 groups=self.channels,
                                 kernel_size=1)

    def forward(self, x):
        # x = self.linear(x)
        # x = self.activation(x)
        # x = self.LayreNorm(x)
        # x = self.decoder(x)
        x = x[:, 1:, :]
        B, L, C = x.shape
        # print(f'-----------Masked_decoder forward    B:{B}, L:{L}, C:{C}-----------')
        x = rearrange(x, 'B (C P) D -> B (C D) P', C=self.channels)
        # x = x.reshape(B, self.num_patch, C)
        x = self.decoder(x)
        x = rearrange(x, 'B (C D) P -> B (C P) D', C=self.channels)

        return x


class Masked_decoder2(nn.Module):
    def __init__(self, hidden_size, patch_size, num_patch, channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_patch = num_patch
        # self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        # # self.activation = nn.ReLU()
        # self.activation = nn.GELU()
        # self.LayreNorm = nn.LayerNorm(hidden_size)
        # self.decoder = nn.Linear(hidden_size, patch_size, bias=True)
        self.decoder = nn.Conv1d(in_channels=hidden_size * self.channels,
                                 out_channels=patch_size * self.channels,
                                 groups=self.channels,
                                 kernel_size=1)

    def forward(self, x):
        # x = self.linear(x)
        # x = self.activation(x)
        # x = self.LayreNorm(x)
        # x = self.decoder(x)
        x = x[:, 1:, :]
        B, L, C = x.shape
        x = rearrange(x, 'B (C P) D -> B (C D) P', C=self.channels)
        x = self.decoder(x)
        x = rearrange(x, 'B (C D) P -> B (C P) D', C=self.channels)

        return x

class LongnetClassificationHead(nn.Module):
    def __init__(self, dim, num_classes, selected_layers):
        """
        Classification head to select embeddings from specific Transformer layers
        and feed them into a classifier.

        Args:
        -----
        dim: int
            Dimensionality of the input embeddings.
        num_classes: int
            Number of output classes for classification.
        selected_layers: list[int]
            Indices of layers from which embeddings are taken, e.g., [5, 11].
        """
        super().__init__()
        if isinstance(selected_layers, str):
            selected_layers = parse_layer_range(selected_layers)
        self.selected_layers = selected_layers
        self.classifier = nn.Linear(dim * len(selected_layers), num_classes)

    def forward(self, layer_outputs):
        """
        Forward pass for the classification head.

        Args:
        -----
        layer_outputs: list[torch.Tensor]
            List of embeddings from all Transformer layers, where each Tensor has shape (batch_size, dim).

        Returns:
        --------
        torch.Tensor:
            Classification logits of shape (batch_size, num_classes).
        """
        # Select embeddings from specified layers
        selected_embeddings = [layer_outputs[i] for i in self.selected_layers]

        # Concatenate embeddings from selected layers
        combined_embeddings = torch.cat(selected_embeddings, dim=-1)  # Shape: (batch_size, dim * num_selected_layers)
        # Pass through the classifier
        logits = self.classifier(combined_embeddings)

        return logits