#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:20:32 2018

@author: Taha Emara  @email: taha@emaraic.com
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import sys
import os
sys.path.append(os.path.abspath('./'))
# from models.backbone_networks import MobileNetV2_prune as MobileNetV2
from models.backbone_networks import MobileNetV1
from models import aspp
from models.separableconv import SeparableConv2d 


def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()




class RT(nn.Module):
    
    def __init__(self, n_classes=59, pretrained=False):
        
        super(RT, self).__init__()
        print("PDESNet-MobileNet...")

        # self.mobile_features=MobileNetV2.MobileNetV2()
        # self.mobile_features=MobileNetV1.MobileNet_res()
        self.mobile_features=MobileNetV1.MobileNet()
        
        if pretrained:
            print('Loading pretrained model: model_best.pth.tar...')
            pretrained_path = os.path.join('pretrained_model', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.mobile_features.load_state_dict(new_state_dict)
        else:
            self.mobile_features.apply(weights_init)
        
        
        # rates = [1, 3, 6, 9]
        rates = [1, 5, 9]

        x_c = 1024
        out_c = 96
        low_level_c = 128
        self.aspp1 = aspp.ASPP(x_c, out_c, rate=rates[0], dw=True)
        self.aspp2 = aspp.ASPP(x_c, out_c, rate=rates[1], dw=True)
        self.aspp3 = aspp.ASPP(x_c, out_c, rate=rates[2], dw=True)
        # self.aspp4 = aspp.ASPP(x_c, out_c, rate=rates[3], dw=True)

        self.relu_prune = nn.ReLU()
        self.global_avg_pool_prune = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(x_c, out_c, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(out_c),
                                             nn.ReLU())
        #self.conv1 = nn.Conv2d(480+1280, 96, 1, bias=False)
        # self.conv1_prune =SeparableConv2d(x_c+out_c*5,out_c,1)
        self.conv1_prune =SeparableConv2d(x_c+out_c*3,out_c,1)
        self.bn1_prune = nn.BatchNorm2d(out_c)

        #adopt [1x1, 48] for channel reduction.
        #self.conv2 = nn.Conv2d(24, 32, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(32)
    
        self.last_conv_prune1 = nn.Sequential(
                                       SeparableConv2d(low_level_c+out_c,out_c,3,1,1),
                                       nn.BatchNorm2d(out_c),
                                       nn.ReLU())
                                       
        self.last_conv_prune2 = nn.Sequential(
                                       SeparableConv2d(out_c,out_c,3,1,1),
                                       nn.BatchNorm2d(out_c),
                                       nn.ReLU())
        
        self.deep_prune = nn.Conv2d(out_c, 1, kernel_size=1, stride=1)
        self.class_prune = nn.Conv2d(out_c, n_classes, kernel_size=1, stride=1)
        
    def trunk(self, input, input_size):
        x, low_level_features = self.mobile_features(input)
        # x: (b,1024,input_h/32,input_h/32)
        # low_level_features: (b, 128, input_h/4, input_w/4)
        # import ipdb; ipdb.set_trace()
        x1 = self.aspp1(x) #(b,96,input_h/32,input_h/32)
        x2 = self.aspp2(x) #(b,96,input_h/32,input_h/32)
        x3 = self.aspp3(x) #(b,96,input_h/32,input_h/32)
        # x4 = self.aspp4(x) #(b,96,input_h/32,input_h/32)
        # x5 = self.global_avg_pool_prune(x) #(b,96,1,1)
        # x5 = F.interpolate(x5, size=(int(math.ceil(input_size[-2]/32)),
                                # int(math.ceil(input_size[-1]/32))), mode='bilinear', align_corners=True) #(b,96,input_h/32,input_h/32)

        # x = torch.cat((x,x1, x2, x3, x4, x5), dim=1) #(b,1760,input_h/32,input_h/32)
        # x = torch.cat((x,x1, x2, x3, x5), dim=1) #(b,288,input_h/32,input_h/32)
        x = torch.cat((x,x1, x2, x3), dim=1) #(b,288,input_h/32,input_h/32)
        #print('after aspp cat',x.size())
        x = self.conv1_prune(x) #(b,96,input_h/32,input_h/32)
        x = self.bn1_prune(x)
        x = self.relu_prune(x)
        x = F.interpolate(x, size=(int(math.ceil(input_size[-2]/4)),
                                int(math.ceil(input_size[-1]/4))), mode='bilinear', align_corners=True) # (b, 96, input_h/4, input_w/4)
       # ablation=torch.max(low_level_features, 1)[1]
        #print('after con on aspp output',x.size())

        ##comment to remove low feature
        #low_level_features = self.conv2(low_level_features)
        #low_level_features = self.bn2(low_level_features)
        #low_level_features = self.relu(low_level_features)
        #print("low",low_level_features.size())
        
        x = torch.cat((x, low_level_features), dim=1) #(b, 120, input_h/4, input_w/4)
        #print('after cat low feature with output of aspp',x.size())
        x = self.last_conv_prune1(x) #(b, 96, input_h/4, input_w/4)
        return x
        
    def head(self, x, input_size):
    
        depth = self.last_conv_prune2(x) #(b, 96, input_h/4, input_w/4)
        depth = self.deep_prune(depth) #(b, 1, input_h/4, input_w/4)
        depth = F.interpolate(depth, size=(int(math.ceil(input_size[-2]/2)),
                                int(math.ceil(input_size[-1]/2))), mode='bilinear', align_corners=True) #(b, 1, input_h/2, input_w/2)

        class_mask = self.last_conv_prune2(x) #(b, 96, input_h/4, input_w/4)
        class_mask = self.class_prune(class_mask) #(b, 1, input_h/4, input_w/4)
        class_mask = F.interpolate(class_mask, size=(int(math.ceil(input_size[-2]/2)),
                                int(math.ceil(input_size[-1]/2))), mode='bilinear', align_corners=True) #(b, n_classes, input_h/2, input_w/2)
                                
        return depth, class_mask
        
    def forward(self, input):
        #import ipdb;ipdb.set_trace()
        input_size = input.size()
        
        x = self.trunk(input, input_size)
        
        depth, class_mask = self.head(x, input_size)
        
        return x#{'depth': depth, 'class_mask': class_mask}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
def flops(model,input_size):
    input = torch.rand(1, 3, input_size, input_size).cuda()
    flops, params = profile(model, inputs=(input, ))
    from thop import clever_format
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    #import pdb;pdb.set_trace()

def FPS(model,input_size):
    import time
    model.eval()

    for x in range(0,200):
        input = torch.randn(1, 3, input_size, input_size).cuda()
        with torch.no_grad():
            # import ipdb;ipdb.set_trace()
            output = model.forward(input)
        
    total=0
    for x in range(0,200):
        input = torch.randn(1, 3, input_size, input_size).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(input)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print('FPS:', str(200/total))
    print('ms:', str(1000*total/200))
    
if __name__ == '__main__':

    from thop import profile
    model = RT(n_classes=59, pretrained=False)
    model.cuda()
    input_size = 224
    # flops(model,input_size)
    FPS(model,input_size)
