import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
from model.backbone import resnet50_3layers
from model.DPF import DPF
from model.CCM import CCM
import numpy as np
from torch import einsum
from einops import rearrange


class DPCCNet(nn.Module):
    def __init__(self, fuse_out_channels=256, ocr_in_channels=768, ocr_mid_channels=256, ocr_key_channels=128, num_class=2, pretrained=True):
        super(DPCCNet, self).__init__()
        backbone_channels = [256, 512, 1024]  # resnet50 layer1-3的channels
        outc = ocr_in_channels // 2
        self.backbone = resnet50_3layers(pretrained=pretrained)
        # fuse_out_channels = 256 或 128
        # ocr_in_channels = 3 * fuse_out_channels
        self.fuse1 = DPF(in_channels=backbone_channels[0], out_channels=fuse_out_channels)
        self.fuse2 = DPF(in_channels=backbone_channels[1], out_channels=fuse_out_channels)
        self.fuse3 = DPF(in_channels=backbone_channels[2], out_channels=fuse_out_channels)


        self.OCR = CCM(in_channels= ocr_in_channels, mid_channels= ocr_mid_channels, key_channels=ocr_key_channels, num_classes=num_class)

        self.head = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, ocr_mid_channels,    # 256 -> 256
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ocr_mid_channels, num_class,
                      kernel_size=1, stride=1, padding=0, bias=True)           # 512 -> 2
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, x1, x2):   # x1 / x2    [B, 3, 256, 256]
#         print(x1)
        x1_layer1, x1_layer2, x1_layer3 = self.backbone(x1)
        x2_layer1, x2_layer2, x2_layer3 = self.backbone(x2)

        feats_fuse1 = self.fuse1(x1_layer1, x2_layer1)   # [fuse_out_channels 64 64]
        feats_fuse2 = self.fuse2(x1_layer2, x2_layer2)   # [fuse_out_channels 32 32]
        feats_fuse3 = self.fuse3(x1_layer3, x2_layer3)   # [fuse_out_channels 16 16]

        feats_fuse2 = self.upsample2(feats_fuse2)    # [64 64]
        feats_fuse3 = self.upsample2(feats_fuse3)    # [64 64]

        feats_fuseall = torch.cat([feats_fuse1, feats_fuse2, feats_fuse3], dim=1) # [fuse_out_channels*3, 64, 64]

        feats_ocr, sim_map, out_aux = self.OCR(feats_fuseall)

        out = self.head(feats_ocr)
        out = self.upsample4(out)
        out_aux = self.upsample4(out_aux)

        return out, out_aux, sim_map


# net = DPCCNet(fuse_out_channels=256, ocr_in_channels=768, ocr_mid_channels=256, ocr_key_channels=128, num_class=2, pretrained=True)
# a = torch.rand([7,3,256,256])
# b = torch.rand([7,3,256,256])
# out, out_aux, sim_map = net(a,b)
# print(out.shape, out_aux.shape, sim_map.shape)