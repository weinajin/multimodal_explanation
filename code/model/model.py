import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

#########################
#######   BRATS #########
#########################
from .blocks import ConvBlock # for plain vgg like 3D network

from monai.networks.nets import densenet121 # for tumorsyn

class GeneNet(nn.Module):# [bat size, C, d, w, h
    def __init__(self, in_features, num_class, depth = 2, init='kaimingNormal', dropout_block = [0,0,0,0,0,0], dropout_fc=0):
        super(GeneNet, self).__init__()
        self.conv1 = ConvBlock(in_features=in_features, out_features=16, num=1, dropout_block = dropout_block[0], pool=True) # /2
        self.conv2 = ConvBlock(in_features=16, out_features=32, num=depth, dropout_block = dropout_block[1], pool=True) # /4
        self.conv3 = ConvBlock(in_features=32, out_features=64, num=depth, dropout_block = dropout_block[2],pool=True) # /8
        self.conv4 = ConvBlock(in_features=64, out_features=128, num=depth, dropout_block = dropout_block[3],pool=True) # /16
        self.conv5 = ConvBlock(in_features=128, out_features=128, num=depth, dropout_block = dropout_block[4],pool=True) # /32
        self.conv6 = ConvBlock(in_features=128, out_features=128, num=depth, dropout_block = dropout_block[5],pool=True) # /64
        self.dense = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(in_features=4096, out_features=num_class, bias=True)
        )
        # initialize
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        N, __, __, __, __ = conv6.size()
        out = self.dense(conv6.view(N,-1))
        return out


