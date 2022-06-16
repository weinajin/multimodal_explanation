import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num, dropout_block, pool=True):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm3d(out_features, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, num):
            layers.append(nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm3d(out_features, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        if dropout_block > 0:
            layers.append(nn.Dropout(dropout_block))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_features, attn_features, subsample=True):
        super(SelfAttentionBlock, self).__init__()
        self.in_features = in_features
        self.attn_features = attn_features
        self.subsample = subsample
        self.g = nn.Conv3d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv3d(in_channels=attn_features, out_channels=in_features, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm3d(in_features, affine=True, track_running_stats=True)
        )
        if subsample:
            self.g = nn.Sequential(
                self.g,
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
            self.phi = nn.MaxPool3d(kernel_size=2, stride=2)
    def forward(self, x):
        N, __, Dx, Wx, Hx = x.size()
        g_x = self.g(x).view(N,self.attn_features,-1).permute(0,2,1) # N x D'W'H' x C'
        theta_x = x.view(N,self.in_features,-1).permute(0,2,1) # N x DWH x C
        if self.subsample:
            phi_x = self.phi(x).view(N,self.in_features,-1) # N x C x D'W'H'
        else:
            phi_x = x.view(N,self.in_features,-1) # N x C x DWH
        c = torch.bmm(theta_x, phi_x) # N x DWH x D'W'H'
        y = torch.bmm(F.softmax(c,dim=-1), g_x) # N x DWH x C'
        y = y.permute(0,2,1).contiguous().view(N,self.attn_features,Dx,Wx,Hx)
        z = self.W(y)
        return z + x

'''
3D version CBAM attention

Reference Paper:
Convolutional Block Attention Module https://arxiv.org/pdf/1807.06521.pdf

Reference code:
https://github.com/kobiso/CBAM-keras
'''
class CBAMAttentionBlock(nn.Module):
    def __init__(self, in_features, reduction=4):
        super(CBAMAttentionBlock, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        self.max_pool = nn.AdaptiveMaxPool3d(output_size=(1,1,1))
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels=in_features, out_channels=in_features//reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_features//reduction, out_channels=in_features, kernel_size=1, padding=0, bias=True)
        )
        # spatial attention
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        # channel attention
        avg_pool = self.mlp(self.avg_pool(x)) # N x C x 1 x 1 x 1
        max_pool = self.mlp(self.max_pool(x)) # N x C x 1 x 1 x 1
        channel_attn = torch.sigmoid(avg_pool+max_pool)
        x_channel_attn = x.mul(channel_attn)
        # spatial attention
        ch_avg_pool = torch.mean(x_channel_attn, dim=1, keepdim=True) # N x 1 x D x H x W
        ch_max_pool, __ = torch.max(x_channel_attn, dim=1, keepdim=True) # N x 1 x D x H x W
        c = self.conv(torch.cat((ch_avg_pool,ch_max_pool), 1))
        spatial_attn = torch.sigmoid(c) # N x 1 x D x H x W
        output = x_channel_attn.mul(spatial_attn)
        return c, output
