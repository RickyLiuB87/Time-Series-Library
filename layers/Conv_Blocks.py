import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp2d(nn.Module):
    def __init__(self, intra_chomp_size, inter_chomp_size, causal=True):
        super(Chomp2d, self).__init__()
        self.intra_chomp_size = intra_chomp_size
        self.inter_chomp_size = inter_chomp_size
        self.causal = causal

    def forward(self, x):
        if self.intra_chomp_size == 0 and self.inter_chomp_size == 0:
            return x
        if self.causal:
            return x[:, :, self.intra_chomp_size:, self.inter_chomp_size:].contiguous()
        else:
            return x[:, :, :-self.intra_chomp_size, :-self.inter_chomp_size].contiguous()
    

class Temporal_Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True, dilation=1, causal=True):
        super(Temporal_Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        chomps = []

        for i in range(self.num_kernels):
            kernel_size = 2 * i + 1
            padding = (kernel_size - 1) * dilation
            chomp_size = padding # Chomp size is equal to the padding size to maintain causality
            kernels.append(weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                     padding=(padding, padding), dilation=(dilation, dilation))))
            chomps.append(Chomp2d(chomp_size, chomp_size, causal=causal))

        self.kernels = nn.ModuleList(kernels)
        self.chomps = nn.ModuleList(chomps)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            conv_out = self.kernels[i](x)
            chomp_out = self.chomps[i](conv_out)
            res_list.append(chomp_out)
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
