#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:02:59 2018

@author: Taha Emara  @email: taha@emaraic.com
"""
import torch.nn as nn
import torch
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class ASPP(nn.Module):
    
    def __init__(self, inplanes, planes, rate, dw=False):
        super(ASPP, self).__init__()
        self.rate=rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            if dw:
                self.conv1 =SeparableConv2d(planes,planes,3,1,1)
            else:
                self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()
   
            #self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes,planes,kernel_size,1,padding,rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        #x = self.relu(x)
        if self.rate!=1:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SAP(nn.Module):
    
    def __init__(self, inplanes, planes):
        super(SAP, self).__init__()

        self.conv1 = SeparableConv2d(inplanes,planes,5,2,2)
        self.conv2 = SeparableConv2d(inplanes,planes,9,4,4)
        self.conv3 = SeparableConv2d(inplanes,planes,17,8,8)
        self.conv4 = SeparableConv2d(inplanes,planes,33,16,16)
        self.conv5 = SeparableConv2d(inplanes,planes,65,32,32)
        
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        #self.atrous_convolution = SeparableConv2d(inplanes,planes,kernel_size,1,padding,rate)
        
        
        self._init_weight()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv2(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        '''
        x4 = self.conv4(x)
        x4 = self.bn(x4)
        x4 = self.relu(x4)

        x5 = self.conv1(x)
        x5 = self.bn(x5)
        x5 = self.relu(x5)
        '''
        return x1, x2, x3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
