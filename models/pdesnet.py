#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 12:00:52 2019

@author: Taha Emara  @email: taha@emaraic.com
"""
import torch 

from models import fast_mobilenetv1_aspp, fast_mobilenetv1_aspp_prune, fast_mobilenetv1_no_aspp, fast_mobilenetv1_aspp_maskinput
from models import fast_mobilenetv1_aspp_conv
from models import fastdepth, Espnet, Espnet_yangmei



class PDESNet():
    
    def build(backbone_network,n_classes,CONFIG=None,is_train=True):
    
        if backbone_network == 'fast_mobilenetv1_aspp':
            net = fast_mobilenetv1_aspp.RT(n_classes=n_classes)
        elif backbone_network == 'fast_mobilenetv1_aspp_prune':
            net = fast_mobilenetv1_aspp_prune.RT(n_classes=n_classes)
        elif backbone_network == 'fast_mobilenetv1_no_aspp':
            net = fast_mobilenetv1_no_aspp.RT(n_classes=n_classes)
        elif backbone_network == 'fast_mobilenetv1_aspp_conv':
            net = fast_mobilenetv1_aspp_conv.RT(n_classes=n_classes)
        elif backbone_network == 'fast_mobilenetv1_aspp_maskinput':
            net = fast_mobilenetv1_aspp_maskinput.RT(n_classes=n_classes)
        elif backbone_network == 'ESPNet_Encoder':
            net = Espnet.ESPNet_Encoder(classes=n_classes)
        elif backbone_network == 'ESPNet_Encoder_yangmei':
            net = Espnet_yangmei.ESPNet_Encoder(classes=n_classes)
        elif backbone_network == 'fastdepth':
            decoder = 'nnconv5'
            net = fastdepth.MobileNet(decoder, 224, in_channels=3)
        else:
            raise NotImplementedError
            
        #if modelpath is not None:
            #net.load_state_dict(torch.load(modelpath)['state_dict'])
            
        print("Using PDESNet with",backbone_network)
        return net
        
            
    

