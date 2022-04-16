# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DSC_3D(nn.Module):      # Depthwise Separable Convolution
    def __init__(self, in_ch, out_ch):
        super(DSC_3D, self).__init__()
        self.depth_conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)



class attention_fusion(nn.Module):
    def __init__(self, features, r, M=2):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            r: the radio for compute d, the length of z.
        """
        super(attention_fusion, self).__init__()
        d = int(features / r)
        self.features = features
        self.relu = nn.ReLU(inplace=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv3d(features, d, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.ModuleList([])
        for i in range(M):
            self.conv2.append(nn.Sequential(
                nn.Conv3d(d, features, kernel_size=1, stride=1, padding=0, bias=True),
            ))
        #self.conv3 = nn.Conv3d(d, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = nn.ModuleList([])
        for i in range(M):
            self.conv4.append(nn.Sequential(
                nn.Conv3d(d, 1, kernel_size=1, stride=1, padding=0, bias=True)
            ))
        self.softmax = nn.Softmax(dim=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.compress = ChannelPool()
        self.conv5 = nn.Conv3d(features, d, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):

        fea1 = x1.unsqueeze_(dim=1)
        fea2 = x2.unsqueeze_(dim=1)
        feas = torch.cat([fea1, fea2], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_c = fea_U.mean(-1).mean(-1).mean(-1)
        fea_c = fea_c.unsqueeze_(dim=-1).unsqueeze_(dim=-1).unsqueeze_(dim=-1)
        fea_z = self.relu(self.conv1(fea_c))
        for i, conv in enumerate(self.conv2):
            vector = conv(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_channel = vector
            else:
                attention_channel = torch.cat([attention_channel, vector], dim=1)
        attention_channel = self.softmax(attention_channel)
        #print(attention_channel.shape)

        fea_s = self.relu1(self.conv5(fea_U))
        for i, conv in enumerate(self.conv4):
            feature = conv(fea_s).unsqueeze_(dim=1)
            if i == 0:
                attention_spaital = feature
            else:
                attention_spaital = torch.cat([attention_spaital, feature], dim=1)
        attention_spaital = self.softmax1(attention_spaital)
        #print(attention_spaital.shape)
        attention = attention_channel*attention_spaital
        attention = self.softmax2(attention)
        fea_v = (feas * attention).sum(dim=1)

        return fea_v

class Res3D(nn.Module):       # 3D Res_block using Depthwise Separable Convolution
    def __init__(self, inChans, outChans):
        super(Res3D, self).__init__()
        self.conv1 = DSC_3D(inChans, outChans)
        self.conv2 = DSC_3D(inChans, outChans)
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.bn2 = nn.BatchNorm3d(outChans)

    def forward(self, x):

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out = out1 + x
        out = self.relu(out)

        return out

class cnn(nn.Module):

    def __init__(self, argx):
        super(cnn, self).__init__()

        kernel_size = 3
        self.conv1 = BasicConv(1, 16, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True)
        self.conv2 = BasicConv(1, 16, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True)
        self.conv3 = DSC_3D(32, 1)
        self.conv11 = DSC_3D(16, 32)
        self.conv12 = DSC_3D(32, 32)
        self.conv21 = DSC_3D(16, 32)
        self.conv22 = DSC_3D(32, 32)

        self.res3D = Res3D(32, 32)

        self.fusion_block = attention_fusion(32, 4) # r = 4
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.bn11 = nn.BatchNorm3d(32)
        self.bn12 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(16)
        self.bn21 = nn.BatchNorm3d(32)
        self.bn22 = nn.BatchNorm3d(32)

    def forward(self, target1, target2):

        x1 = self.conv1(target1)
        x1 = self.relu(self.bn1(x1))
        y1 = self.conv2(target2)
        y1 = self.relu(self.bn2(y1))
        x1 = self.conv11(x1)
        x1 = self.relu(self.bn11(x1))
        y1 = self.conv21(y1)
        y1 = self.relu(self.bn21(y1))
        x1 = self.conv12(x1)
        x1 = self.relu(self.bn12(x1))
        y1 = self.conv22(y1)
        y1 = self.relu(self.bn22(y1))

        out = self.fusion_block(x1, y1)
        #out = x1 + y1  # ablation study

        out = self.res3D(out)
        out = self.conv3(out)
        mask = torch.sigmoid(out)

        return mask