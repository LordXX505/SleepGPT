import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
            kernel_size = kernel_size//3
        self.conv = nn.Conv1d(in_channels=in_channels*self.channels, out_channels=self.channels*out_channels,
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

        if return_alpha is False:
            self.fc_norm = nn.LayerNorm(eps=1e-6, normalized_shape=out_size)
        self.return_alpha = return_alpha
        if self.reshape and self.double:
            self.hidden_size = hidden_size*2
            self.out_size = out_size*2
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

    def forward(self, x, time_split=None):
        if time_split==-1:
            b, c, p, d = x.shape
            assert c==self.channels
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
                x_time = x[:, :time_split].reshape(b*self.channels, -1, d)
                x_fft = x[:, time_split:].reshape(b*self.channels, -1, d)
                x = torch.cat([x_time, x_fft], dim=-1)
                softdim = 2
        if self.channel_wise:
            a_states = []
            alpha = []
            assert x.shape[1]==self.channels
            for i in range(self.channels):
                a_states_temp = torch.tanh(getattr(self, f"w_ha_{i}")(x[:, i]))
                alpha_temp = torch.softmax(getattr(self, f"w_at_{i}")(a_states_temp), dim=softdim).view(x.size(0), 1, -1)
                a_states.append(a_states_temp)
                alpha.append(alpha_temp)
            a_states = torch.stack(a_states, dim=1)
            alpha = torch.stack(alpha, dim=1)
            assert a_states.shape == (b, c, p, self.out_size), f"a_states.shape:{a_states.shape}, x.shape:{x.shape}"
            a_states = a_states.reshape(b*self.channels, -1, self.out_size)
            alpha = alpha.reshape(b*self.channels, alpha.shape[2], alpha.shape[3])
        else:
            a_states = torch.tanh(self.w_ha(x))
            alpha = torch.softmax(self.w_at(a_states), dim=softdim).view(x.size(0), 1, -1)
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
            assert x_time.shape == (b, self.channels, self.out_size//2), f"{x_time.shape}"
            assert x_fft.shape == (b, self.channels, self.out_size//2), f"{x_fft.shape}"
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


class Spindle_Head(nn.Module):
    def __init__(self, hidden_size, patch_size, weight=None):
        super().__init__()
        self.decoder = nn.Conv1d(in_channels=2*hidden_size,
                                 out_channels=2*patch_size,
                                 kernel_size=1)
    def forward(self, x):
        B = x.shape[0]
        x = self.decoder(x)
        x = rearrange(x, 'B (C J) T -> B T C J', J=2)
        x = torch.softmax(x, dim=-1)
        x = x.reshape(B, -1, 2)
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
        self.decoder = nn.Conv1d(in_channels=hidden_size*self.channels,
                                 out_channels=patch_size*self.channels,
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
        self.decoder = nn.Conv1d(in_channels=hidden_size*self.channels,
                                 out_channels=patch_size*self.channels,
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
