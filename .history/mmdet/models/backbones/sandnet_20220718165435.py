from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from mmdet.utils import get_root_logger
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES

from mmdet.models.backbones.basic import *
from mmdet.models.backbones.blocks import *
from mmdet.models.backbones.blocks_repvgg import *


def dwconv33(in_channels, stride=1, padding=1, deploy=False):
    return OREPA(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, 
                 groups=in_channels, deploy=deploy)

def pwconv11(in_channels, out_channels, deploy=False):
    return OREPA_1x1(in_channels, out_channels, kernel_size=1, stride=1, padding=0, deploy=deploy)


class SepConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, padding=1, dilation=1, deploy=False):
        super(SepConvBR, self).__init__()

        self.pwconv = pwconv11(in_channels, out_channels, deploy=deploy)
        self.dwconv = dwconv33(out_channels, stride=stride, padding=padding, deploy=deploy)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.relu(x)

        x = self.dwconv(x)

        return x
            
            
class SandPart1(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, padding=1, dilation=1, deploy=False):
        super(SandPart1, self).__init__()

        self.identity= nn.Identity()
        self.unuse_part1 = (in_channels == out_channels)
        
        if not self.unuse_part1:
            self.dwconv = dwconv33(in_channels, stride=stride, padding=padding, deploy=deploy)
            self.pwconv = pwconv11(in_channels, out_channels, deploy=deploy)
        
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.unuse_part1:
            return self.identity(x)
        else:
            x = self.dwconv(x)
            x = self.relu(x)
        
            x = self.pwconv(x)

            return x
        
        
class SandPart2(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, padding=1, dilation=1, deploy=False):
        super(SandPart2, self).__init__()

        self.pwconv = pwconv11(in_channels, out_channels, deploy=deploy)
        self.dwconv = dwconv33(out_channels, stride=stride, padding=padding, deploy=deploy)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.relu(x)

        x = self.dwconv(x)

        return x


class StemBlock(nn.Module):
    def __init__(self, channel, out_channel, deploy=False):
        super(StemBlock, self).__init__()

        # self.stem_left = ConvK3(channel, channel + 8, stride=2)
        self.stem_left = ConvBN(channel, channel + 8, 3, stride=2, padding=1, deploy=deploy, 
                                nonlinear=nn.ReLU(inplace=True))
        
        self.stem_right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fuse = OREPA(out_channel, out_channel, 3, stride=1, padding=1, deploy=deploy, 
                          nonlinear=nn.ReLU(inplace=True))

    def forward(self, x):
        feat_left = self.stem_left(x)
        feat_right = self.stem_right(x)
        
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        
        return feat
        
        
class Sand_BottleNeck(nn.Module):

      def __init__(self, in_channels, mid_channels, expansion, stride=1, padding=1, dilation=1, iden_mapping=True, 
                   deploy=False):
         super(Sand_BottleNeck, self).__init__()
         self.iden_mapping = iden_mapping
         
         self.unuse_skip_connect = (mid_channels == in_channels)
         
         self.expansion = expansion
         
         self.conv1 = SandPart1(in_channels, mid_channels, stride=1, 
                                    padding=padding, dilation=dilation, deploy=deploy)
         self.conv2 = SandPart2(mid_channels, mid_channels*self.expansion, stride=stride, 
                                    padding=padding, dilation=dilation, deploy=deploy)
         self.relu = nn.ReLU(inplace=True)

         if not self.iden_mapping:
            self.shortcut = SepConvBR(in_channels, mid_channels*self.expansion, stride=stride, deploy=deploy)
         if self.unuse_skip_connect:
            self.shortcut = nn.Identity()
        
      def forward(self, x):
          shortcut = x
          res=self.conv1(x)
          res=self.conv2(res)
          if self.unuse_skip_connect:
            return self.relu(res)
          else:
            if not self.iden_mapping:
                shortcut = self.shortcut(x)
            return res + shortcut


@BACKBONES.register_module()
class SandNet(nn.Module):
     def __init__(self, bottleneck=Sand_BottleNeck, 
                        in_channels=3, 
                        block_cfg=None,
                        out_stride=32,
                        orepa_deploy=False):
        super(SandNet, self).__init__()

        if block_cfg is None:
            midchannel_comb = [32, 32, 32]
            depth_comb = [3,5,3]
            # orepa_deploy = False
        else:
            midchannel_comb = block_cfg['midchannel']
            depth_comb = block_cfg['depth']
            # orepa_deploy = block_cfg['deploy']
            
        self.out_stride = out_stride
        self.deploy = orepa_deploy
        
        assert self.out_stride in [8,16,32]
        if self.out_stride==8:
           self.last_two_stride = [1,1]
           self.last_two_dilation = [2,4]
        elif self.out_stride==16:
           self.last_two_stride = [2,1]
           self.last_two_dilation = [1,2]
        else:
           self.last_two_stride = [2,2]
           self.last_two_dilation = [1,1]

        # stage 1:
        # self.layer1 = ConvK3(in_channels, 12, stride=2)
        self.layer1 = ConvBN(in_channels, 12, 3, stride=2, padding=1, deploy=self.deploy, 
                                nonlinear=nn.ReLU(inplace=True))

        # stage 2: stride 4 feature map
        self.layer2 = StemBlock(12, 32, deploy=self.deploy)
   
        # contrcut block using ''bottleneck'' class
        self.block3 = self._make_block(bottleneck, midchannel_comb[0], 3,
                                       depth_comb[0], 
                                       midchannel_comb[0], 
                                       stride=2, 
                                       deploy=self.deploy)

        self.block4 = self._make_block(bottleneck, midchannel_comb[0]*3, 4,
                                       depth_comb[1], 
                                       midchannel_comb[1], 
                                       stride=self.last_two_stride[0],
                                       dilation=self.last_two_dilation[0],
                                       deploy=self.deploy)

        self.block5 = self._make_block(bottleneck, midchannel_comb[1]*4, 4,
                                       depth_comb[2], 
                                       midchannel_comb[2], 
                                       stride=self.last_two_stride[1],
                                       dilation=self.last_two_dilation[1],
                                       deploy=self.deploy)

     def _make_block(self, 
                     bottleneck, 
                     in_channels, 
                     expansion,
                     depth, 
                     mid_channels, 
                     stride=1, 
                     dilation=1,
                     deploy=False):
        layers = []
        iden_mapping = False if stride > 1 or dilation>1 or in_channels == mid_channels else True
        layers.append(bottleneck(in_channels, mid_channels, expansion, stride=stride,  dilation=dilation, 
                                 iden_mapping=iden_mapping, deploy=deploy))

        in_channels = mid_channels * expansion
        for i in range(1, depth):#
            layers.append(bottleneck(in_channels, mid_channels, expansion, stride=1, dilation=dilation,
                                     iden_mapping=True, deploy=deploy))

        return nn.Sequential(*layers)

     def forward(self, x):
        out2 = self.layer2(self.layer1(x))
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        return tuple([out3, out4, out5])