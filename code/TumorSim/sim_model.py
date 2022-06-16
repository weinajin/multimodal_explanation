import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torchvision import models
from collections import namedtuple
from .Inpaint_base_model import *


class Inpaint_generator(nn.Module):
    def __init__(self):
        super(Inpaint_generator, self).__init__()

        model = [
            Conv(in_ch=5, out_ch=32, K=3, S=2, P=1),
            Conv(in_ch=32, out_ch=128, K=3, S=2, P=1)
        ]
        for _ in range(3):
            model += [ResidualBlock(128)]
        self.model_initial = nn.Sequential(*model)

        model = [Conv(in_ch=1, out_ch=4, K=3, S=2, P=1),
                 Conv(in_ch=4, out_ch=16, K=3, S=2, P=1)]
        self.model_Mask = nn.Sequential(*model)

        model = [Conv(in_ch=144, out_ch=256, K=3, S=1, P=1),
                 ResidualBlock(256),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     256, 128, 3, 1, 1),
                 ResidualBlock(128),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     128, 64, 3, 1, 1),
                 ResidualBlock(64),
                 Conv(in_ch=64, out_ch=1, K=7, S=1, P=3),
                 ]
        self.model_F = nn.Sequential(*model)

        model = [Conv(in_ch=144, out_ch=256, K=3, S=1, P=1),
                 ResidualBlock(256),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     256, 128, 3, 1, 1),
                 ResidualBlock(128),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     128, 64, 3, 1, 1),
                 ResidualBlock(64),
                 Conv(in_ch=64, out_ch=1, K=7, S=1, P=3),
                 ]
        self.model_T1 = nn.Sequential(*model)

        model = [Conv(in_ch=144, out_ch=256, K=3, S=1, P=1),
                 ResidualBlock(256),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     256, 128, 3, 1, 1),
                 ResidualBlock(128),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     128, 64, 3, 1, 1),
                 ResidualBlock(64),
                 Conv(in_ch=64, out_ch=1, K=7, S=1, P=3),
                 ]
        self.model_T1c = nn.Sequential(*model)

        model = [Conv(in_ch=144, out_ch=256, K=3, S=1, P=1),
                 ResidualBlock(256),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     256, 128, 3, 1, 1),
                 ResidualBlock(128),
                 nn.Upsample(scale_factor=2, mode='nearest'), Conv(
                     128, 64, 3, 1, 1),
                 ResidualBlock(64),
                 Conv(in_ch=64, out_ch=1, K=7, S=1, P=3),
                 ]
        self.model_T2 = nn.Sequential(*model)

    def forward(self, brain_blank, M):
        x = self.model_initial(torch.cat((brain_blank, M), 1))
        M = self.model_Mask(M)
        x = torch.cat((x, M), 1)
        F, T1, T1c, T2 = self.model_F(x), self.model_T1(
            x), self.model_T1c(x), self.model_T2(x)
        return torch.cat((F, T1, T1c, T2), 1)


class Inpaint_discriminator(nn.Module):
    def __init__(self, input_nc=5):
        super(Inpaint_discriminator, self).__init__()

        model = [SN_Conv(in_ch=input_nc, out_ch=32, K=3, S=2, P=1)]
        model += [SN_Conv(in_ch=32, out_ch=64, K=3, S=2, P=1)]
        model += [SN_Conv(in_ch=64, out_ch=256, K=3, S=1, P=1)]
        for _ in range(3):
            model += [SN_ResidualBlock(256)]
        model += [SN_Conv(in_ch=256, out_ch=64, K=3, S=1, P=1)]
        model += [SN_Conv(in_ch=64, out_ch=1, K=3, S=1,
                          P=1, activation=nn.Sigmoid())]

        self.model = nn.Sequential(*model)
        self.Avg = nn.AvgPool2d(kernel_size=64)

    def forward(self, tumor, M):
        x = self.model(torch.cat((tumor, M), 1))
        return self.Avg(x).view(x.size()[0], -1)


class Tumor_shape(nn.Module):
    def __init__(self, input_nc=2):
        super(Tumor_shape, self).__init__()

        # Down sampling
        model = [Conv(in_ch=input_nc, out_ch=32, K=5, S=1,
                      P=2, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=32, out_ch=64, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=64, out_ch=128, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=128, out_ch=256, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]

        for _ in range(3):
            model += [SN_ResidualBlock(256, activation=nn.LeakyReLU())]

        # Upsampling
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(32),
                  nn.LeakyReLU(inplace=True)]

        model += [Conv(in_ch=32, out_ch=1, K=5, S=1,
                       P=2, activation=nn.LeakyReLU())]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Tumor_grade(nn.Module):
    def __init__(self, input_nc=2):
        super(Tumor_grade, self).__init__()
        # Down sampling
        model = [Conv(in_ch=input_nc, out_ch=32, K=5, S=1,
                      P=2, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=32, out_ch=64, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=64, out_ch=128, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=128, out_ch=256, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        for _ in range(3):
            model += [SN_ResidualBlock(256, activation=nn.LeakyReLU())]
        # Upsampling
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(32),
                  nn.LeakyReLU(inplace=True)]
        model += [Conv(in_ch=32, out_ch=1, K=5, S=1,
                       P=2, activation=nn.LeakyReLU())]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
