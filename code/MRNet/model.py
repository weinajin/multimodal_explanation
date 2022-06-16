import pdb
import torch
import torch.nn as nn

from torchvision import models

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

class TripleMRNet(nn.Module):
    def __init__(self, backbone="resnet18", training=True):
        super().__init__()
        self.backbone = backbone
        if self.backbone == "resnet18":
            resnet = models.resnet18(pretrained=training)
            modules = list(resnet.children())[:-1]
            self.axial_net = nn.Sequential(*modules)
            for param in self.axial_net.parameters():
                param.requires_grad = False
        elif self.backbone == "alexnet":
            self.axial_net = models.alexnet(pretrained=training)

        if self.backbone == "resnet18":
            resnet = models.resnet18(pretrained=training)
            modules = list(resnet.children())[:-1]
            self.sagit_net = nn.Sequential(*modules)
            for param in self.sagit_net.parameters():
                param.requires_grad = False
        elif self.backbone == "alexnet":
            self.sagit_net = models.alexnet(pretrained=training)
        
        if self.backbone == "resnet18":
            resnet = models.resnet18(pretrained=training)
            modules = list(resnet.children())[:-1]
            self.coron_net = nn.Sequential(*modules)
            for param in self.coron_net.parameters():
                param.requires_grad = False
        elif self.backbone == "alexnet":
            self.coron_net = models.alexnet(pretrained=training)

        self.gap_axial = nn.AdaptiveAvgPool2d(1)
        self.gap_sagit = nn.AdaptiveAvgPool2d(1)
        self.gap_coron = nn.AdaptiveAvgPool2d(1)
       
        if self.backbone == "resnet18":
            self.classifier = nn.Linear(3*512, 1)
        elif self.backbone == "alexnet":
            self.classifier = nn.Linear(3*256, 1)

    def forward(self, vol_axial, vol_sagit, vol_coron):
        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)

        if self.backbone == "resnet18":
            vol_axial = self.axial_net(vol_axial)
            vol_sagit = self.sagit_net(vol_sagit)
            vol_coron = self.coron_net(vol_coron)
        elif self.backbone == "alexnet":
            vol_axial = self.axial_net.features(vol_axial)
            vol_sagit = self.sagit_net.features(vol_sagit)
            vol_coron = self.coron_net.features(vol_coron)

        vol_axial = self.gap_axial(vol_axial).view(vol_axial.size(0), -1)
        x = torch.max(vol_axial, 0, keepdim=True)[0]
        vol_sagit = self.gap_sagit(vol_sagit).view(vol_sagit.size(0), -1)
        y = torch.max(vol_sagit, 0, keepdim=True)[0]
        vol_coron = self.gap_coron(vol_coron).view(vol_coron.size(0), -1)
        z = torch.max(vol_coron, 0, keepdim=True)[0]

        w = torch.cat((x, y, z), 1)
        out = self.classifier(w)
        return out

    # def forward(self, vol_axial, vol_sagit, vol_coron):
    #     print(vol_axial.requires_grad)
    #     bz = vol_axial.shape[0]
    #     out_bz = []
    #     for i in range(bz):
    #         # vol_axial = vol_axial_bz[i]
    #         # vol_coron = vol_coron_bz[i]
    #         # vol_sagit = vol_sagit_bz[i]
    #         print(vol_axial[i].requires_grad)
    #
    #         if self.backbone == "resnet18":
    #             vol_axial_0 = self.axial_net(vol_axial[i])
    #             vol_sagit_0 = self.sagit_net(vol_sagit[i])
    #             vol_coron_0 = self.coron_net(vol_coron[i])
    #         elif self.backbone == "alexnet":
    #             for param in self.axial_net.parameters():
    #                 print(param.requires_grad ,'para')
    #             vol_axial_0 = self.axial_net.features(vol_axial[i])
    #             vol_sagit_0 = self.sagit_net.features(vol_sagit[i])
    #             vol_coron_0 = self.coron_net.features(vol_coron[i])
    #         # print(vol_axial[i].shape)
    #         vol_axial_0 = self.gap_axial(vol_axial_0).view(vol_axial_0.size(0), -1)
    #         # print(vol_axial_0.shape, 'vol_axial_0')
    #         x = torch.max(vol_axial_0, 0, keepdim=True)[0]
    #         # print(x.shape, 'x')
    #         vol_sagit_0 = self.gap_sagit(vol_sagit_0).view(vol_sagit_0.size(0), -1)
    #         y = torch.max(vol_sagit_0, 0, keepdim=True)[0]
    #         vol_coron_0 = self.gap_coron(vol_coron_0).view(vol_coron_0.size(0), -1)
    #         z = torch.max(vol_coron_0, 0, keepdim=True)[0]
    #
    #         w = torch.cat((x, y, z), 1)
    #         # print('w', w.shape)
    #         out = self.classifier(w)
    #         # print('out', out)
    #         #out_bz[i] = out.item()
    #         out_bz.append(out)
    #     out_bz = torch.stack(out_bz)
    #     print(out_bz.requires_grad, 'out')
    #
    #     return out_bz


    # def forward(self, vol_axial, vol_sagit, vol_coron):
    #
    #     print(vol_axial.requires_grad)
    #     vol_axial = torch.squeeze(vol_axial, dim=0)
    #     vol_sagit = torch.squeeze(vol_sagit, dim=0)
    #     vol_coron = torch.squeeze(vol_coron, dim=0)
    #     print(vol_axial.requires_grad)
    #
    #     if self.backbone == "resnet18":
    #         vol_axial = self.axial_net(vol_axial)
    #         vol_sagit = self.sagit_net(vol_sagit)
    #         vol_coron = self.coron_net(vol_coron)
    #     elif self.backbone == "alexnet":
    #         vol_axial = self.axial_net.features(vol_axial)
    #         vol_sagit = self.sagit_net.features(vol_sagit)
    #         vol_coron = self.coron_net.features(vol_coron)
    #
    #     vol_axial = self.gap_axial(vol_axial).view(vol_axial.size(0), -1)
    #     x = torch.max(vol_axial, 0, keepdim=True)[0]
    #     vol_sagit = self.gap_sagit(vol_sagit).view(vol_sagit.size(0), -1)
    #     y = torch.max(vol_sagit, 0, keepdim=True)[0]
    #     vol_coron = self.gap_coron(vol_coron).view(vol_coron.size(0), -1)
    #     z = torch.max(vol_coron, 0, keepdim=True)[0]
    #
    #     w = torch.cat((x, y, z), 1)
    #     out = self.classifier(w)
    #     print(out.requires_grad, 'out')
    #     return out