# -*- coding: utf-8 -*-


import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import numpy as np
import yaml
from addict import Dict
import argparse

# PyTorch includes
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import cityscapes
from dataloaders import utils
from dataloaders import augmentation as augment
from models.pdesnet import PDESNet
from models import fast_mobilenetv1_aspp_prune
from models.fastdepth import MobileNetSkipAdd
from utils import loss as losses
from utils import iou_eval
import util

import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


Colors = np.array([[0,0,0], [0,255,255],[114, 255,   0], [255,   0, 105], [255, 131,   0],
                  [  0, 255, 202], [  0, 255, 149], [219,   0, 255], [255,   0,   0], [219, 255,   0],
                  [  0,  43, 255], [255,  79,   0], [114,   0, 255], [255,   0, 211], [255, 237,   0],
                  [255,  26,   0], [ 87, 255,   0], [  0,  70, 255], [246,   0, 255], [  0, 255,  70],
                  [  0, 255,  96], [255,   0, 131], [  0, 255,  17], [193,   0, 255], [255, 211,   0],
                  [ 61,   0, 255], [255,   0, 158], [  0, 202, 255], [255,   0,  26], [167, 255,   0],
                  [255,   0, 237], [  8, 255,   0], [ 61, 255,   0], [255, 105,   0], [255, 184,   0],
                  [ 35,   0, 255], [  0, 175, 255], [  0, 255, 228], [255,  52,   0], [255,   0,  79],
                  [140,   0, 255], [  0, 255, 175], [167,   0, 255], [193, 255,   0], [  0, 255, 123],
                  [246, 255,   0], [  0,  96, 255], [  0, 255, 255], [  8,   0, 255], [ 35, 255,   0],
                  [  0, 255,  43], [140, 255,   0], [255,   0, 184], [ 87,   0, 255], [  0,  17, 255], 
                  [  0, 228, 255], [255,   0,  52], [  0, 123, 255], [  0, 149, 255]])
                  
                  
def map_semantic_id_to_color(label):
    #import ipdb;ipdb.set_trace()
    return Colors[label]




ap = argparse.ArgumentParser()
ap.add_argument('--backbone_network', required=False,
                help = 'name of backbone network',default='mobilenet')#shufflenet, mobilenet, and darknet
ap.add_argument('--model_path_resume', required=False,
                help = 'path to a model to resume from',default='our_model/EPFL_checkpoint_mobilenet.pth.tar')
ap.add_argument('--steps-plot', type=int, default=1000)
ap.add_argument('--epochs-vis', type=int, default=1)

args = ap.parse_args()
backbone_network=args.backbone_network
model_path_resume=args.model_path_resume

NUM_CLASSES = 59

testBatch = 1  # Testing batch size
nValInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 1  # Store a model every snapshot epochs


save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))



with torch.cuda.device(0): 

    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    np.random.seed(2020)
    torch.backends.cudnn.benchmark = True
    
    CONFIG=Dict(yaml.load(open("config/training.yaml")))    
    
    net= fast_mobilenetv1_aspp_prune.RT()#MobileNetSkipAdd()
    torch.cuda.set_device(device=CONFIG.GPU_ID)
    ckpt = torch.load(model_path_resume, map_location=lambda storage, loc: storage)
    #ckpt = torch.load(model_path_resume)
    net.load_state_dict(ckpt['state_dict'])
    #net.load_state_dict(ckpt)
    #load_state_dict(net.state_dict(), ckpt)
    net.cuda()
    
    composed_transforms = transforms.Compose([
        augment.Scale((240, 320), 1),
        #augment.RandomHorizontalFlip(),
        augment.ToTensor()])

    dataset_path='./demoImages'#CONFIG.DATASET
    CAD60_val = cityscapes.Cityscapes_test(root=dataset_path, transform=composed_transforms)
    #CAD60_val = cityscapes.Cityscapes(root=dataset_path, split='Test', transform=composed_transforms)
        
    valloader = DataLoader(CAD60_val, batch_size=testBatch, shuffle=False, num_workers=4)
    
    num_img_vl = len(valloader)
    net.eval()
    image_dir = './Show/demoImages'
    depth_dir = './Show/demoImages_people_depth'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    idx = 0
    for ii, sample_batched in enumerate(valloader):
        print(ii+1,'/', num_img_vl)
        inputs = sample_batched
        inputs = Variable(inputs, requires_grad=True)
        inputs = inputs.cuda()
        with torch.no_grad():
            output = net.forward(inputs)
        semantics, predepths = output['class_mask'], output['depth']
        #semantics, predepths = output
        semantics = F.interpolate(semantics, scale_factor=2, mode='bilinear', align_corners=True)
        predepths = F.interpolate(predepths, scale_factor=2, mode='bilinear', align_corners=True)
        #import ipdb;ipdb.set_trace()
        #predepths[depths==0] = 0
        batchSize = inputs.size(0)
        for i in range(batchSize):
            #import ipdb;ipdb.set_trace()
            idx += 1
            image = inputs[i]
            #gtdepth = depths.unsqueeze(1)[i]
            predsemantic = semantics.max(1)[1].unsqueeze(1).data[i][0].cpu().numpy()
            predsemantic[predsemantic!=1] = 0
            
            #gtsemantic = labels.data[i][0].cpu().numpy().astype(np.int)
            #gtsemantic[gtsemantic!=1] = 0
            #semanticShowGT = map_semantic_id_to_color(gtsemantic)
            semanticShowPRE = map_semantic_id_to_color(predsemantic)
            preddepth = predepths.unsqueeze(1)[i]
    
            #plt.imsave(os.path.join(image_dir, str(idx)+'_GTdepth' + '.png'), gtdepth.cpu().detach().numpy().squeeze(), cmap='rainbow')
            plt.imsave(os.path.join(image_dir, str(idx)+'_PREdepth' + '.png' ), preddepth.cpu().detach().numpy().squeeze(), cmap='rainbow')
            cv2.imwrite(os.path.join(image_dir, str(idx)+'_PREsemantic' + '.png'), semanticShowPRE)
            #cv2.imwrite(os.path.join(image_dir, str(idx)+'_GTsemantic' + '.png'), semanticShowGT)
            cv2.imwrite(os.path.join(image_dir, str(idx)+'_RGB' + '.png'), image.permute(1,2,0).cpu().detach().numpy().squeeze()*255)
            
            #preddepth[gtdepth==0]=0
            #plt.imsave(os.path.join(image_dir, str(idx)+'_PREdepth_set0' + '.png' ), preddepth.cpu().detach().numpy().squeeze(), cmap='rainbow')
            
            showdepthPRE = cv2.imread(os.path.join(image_dir, str(idx)+'_PREdepth' + '.png' ))
            showdepthPRE[predsemantic!=1]=0
            cv2.imwrite(os.path.join(depth_dir, str(idx)+'_PRE_semantic_depth' + '.png'), showdepthPRE)