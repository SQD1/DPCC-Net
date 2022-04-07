import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d



class DPF(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None):
        super(DPF, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.to_query = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.inter_channels)
        )

        self.to_key = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.inter_channels)
        )

        self.to_value = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.inter_channels)
        )


        self.point_wise = nn.Sequential(
            conv_nd(in_channels=2 * self.inter_channels, out_channels=self.out_channels, kernel_size=1, stride=1,padding=0),
            ModuleHelper.BNReLU(self.out_channels)
        )


    def forward(self, x1, x2):
        # x1 [B, C, 64, 64]
        # x2 [B, C, 64, 64]

        # [B, C, 64, 64]  ->  [B, C, HW]
        # x1 = rearrange(x1, 'b c h w -> b c (h w)')
        h = x1.shape[-2]

        q1 =self.to_query(x1)     # share weight     [B, C/2, H, W]
        q1 = rearrange(q1, 'b c h w -> b (h w) c')   #  [B, HW, C/2]

        k2 = self.to_key(x2)
        k2 = rearrange(k2, 'b c h w -> b c (h w)')   #  [B, C/2, HW]

        v1 = self.to_value(x1)
        v2 = self.to_value(x2)
        v1 = rearrange(v1, 'b c h w -> b (h w) c')   #  [B, HW, C/2]
        v2 = rearrange(v2, 'b c h w -> b (h w) c')   #  [B, HW, C/2]

        dots1 = einsum('b n c, b c m -> b n m', q1, k2)
        dots1 = (self.inter_channels**-.5) * dots1
        dots1 = F.softmax(dots1, dim=-1)             # dots1 [B, HW, HW]   dim=-1 表示对(HW,HW)的每一行进行softmax
        change_atten1 = 1 - dots1

        # q1 k2 v2
        out1 = einsum('b n n, b n c -> b n c', change_atten1, v2)   # [B, HW, C/2]
        out1 = rearrange(out1, 'b (h w) c -> b c h w', h=h)         # out1 [B, C/2, H, W]

        # q2 k1 v1
        out2 = einsum('b n n, b n c -> b n c', change_atten1, v1)  # [B, HW, C/2]
        out2 = rearrange(out2, 'b (h w) c -> b c h w', h=h)  # out2 [B, C/2, H, W]


        out = torch.cat([out1, out2], dim=1)
        out = self.point_wise(out)

        return out