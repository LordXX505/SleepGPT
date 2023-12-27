import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=self.kernel_size, padding=padding,
                               bias=True)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=self.kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return self.relu(Y + identity)


class UnetLayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, padding=1, batchnorm=True):
        super(UnetLayer, self).__init__()
        conv1 = torch.nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding)
        if batchnorm:
            batchnormd = torch.nn.BatchNorm1d(out_channel)
        else:
            batchnormd = None
        if batchnorm:
            self.layer = nn.Sequential(conv1, batchnormd, nn.ReLU(inplace=True))
        else:
            self.layer = nn.Sequential(conv1, nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

class UnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, padding=1, batchnorm=True, depth=None):
        super(UnetBlock, self).__init__()
        self.block = nn.Sequential()
        for i in range(depth):
            self.block.append(UnetLayer(in_channel=in_channel, out_channel=out_channel, stride=stride,
                                        kernel_size=kernel_size, padding=padding, batchnorm=batchnorm))
    def forward(self, x):
        return self.block(x)

class FPN(nn.Module):
    def __init__(self, up_sample=None, in_channels=2 * 768, feature_scale=2, kernel_size=None, stride=None,
                 num_layers=None, res_kernel_size=None, resnet=False, depth=None):
        super().__init__()
        self.feature_scale = feature_scale
        self.up_sample = up_sample
        filters = [in_channels, in_channels // 2, in_channels // 4, in_channels // 8, in_channels // 16]
        filters = [int(x // self.feature_scale) for x in filters]
        if num_layers is None:
            num_layers = [2, 2, 2, 4, 2]
        if kernel_size is None:
            kernel_size = [2, 4, 5, 5]
        if stride is None:
            stride = [2, 4, 5, 5]
        if res_kernel_size is None:
            res_kernel_size = [3, 5, 11, 11, 11]
        if up_sample is None:
            for i in range(len(kernel_size)):
                setattr(self, f'up_{i}', nn.ConvTranspose1d(in_channels=filters[i], out_channels=filters[i + 1],
                                                            kernel_size=kernel_size[i], stride=stride[i]))
        else:
            for i in range(len(kernel_size)):
                setattr(self, f'up_{i}', partial(F.interpolate, scale_factor=stride[i], mode='linear'))
        for i in range(len(filters)):
            setattr(self, f'conv_{i}', nn.ModuleList())
            for j in range(num_layers[i]):
                if resnet is False:
                    getattr(self, f'conv_{i}').append(UnetBlock(in_channel=filters[i], out_channel=filters[i],
                                                                kernel_size=res_kernel_size[i],
                                                                padding=res_kernel_size[i] // 2, depth=depth))
                else:
                    getattr(self, f'conv_{i}').append(BasicBlock(in_channel=filters[i], out_channel=filters[i],
                                                                 kernel_size=res_kernel_size[i],
                                                                 padding=res_kernel_size[i] // 2))

        self.last = nn.Conv1d(in_channels=filters[-1], out_channels=2, kernel_size=1)
        self.layers = len(num_layers)
        self.mlp = nn.Sequential(nn.Linear(in_channels, filters[0]),
                                 nn.LayerNorm(filters[0]))

    def forward(self, x, trans=True):
        x = self.mlp(x)
        if trans:
            x = torch.transpose(x, dim0=1, dim1=2)
        for i in range(self.layers):
            for _, layer in enumerate(getattr(self, f'conv_{i}')):
                x = layer(x)
            if i != self.layers - 1:
                x = getattr(self, f'up_{i}')(x)
        x = self.last(x).permute(0, 2, 1)
        x = F.softmax(x, dim=-1)
        return x[..., 0]
