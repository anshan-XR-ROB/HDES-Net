#!/usr/bin/env python3
"""
Created on Sat Oct 20 00:18:43 2018

@author: Taha Emara  @email: taha@emaraic.com
"""

import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from dataloaders.utils import listFiles

class Cityscapes(data.Dataset):

    def __init__(self, root='path/to/datasets/cityscapes', split="Train", transform=None):
        """
        Cityscapes dataset folder has two folders, 'leftImg8bit' folder for images and 'gtFine_trainvaltest' 
        folder for annotated images with fine annotations 'labels'.
        """
        self.root = root
        self.split = split #train, validation, and test sets
        self.transform = transform
        self.files = {}
        self.n_classes = 59

        print("Using CAD dataset")
        self.images_path = os.path.join(self.root, self.split, 'RGB')
        self.labels_path = os.path.join(self.root, self.split, 'Semantic_label')
        self.depths_path = os.path.join(self.root, self.split, 'Depth')
          
            
        #print(self.images_path)
        self.files[split] = listFiles(rootdir=self.images_path, suffix='.png')#list of the pathes to images

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1] #not to train
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        #print(self.class_map)
        
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images.path))

        print("Found %d %s images" % (len(self.files[split]), split))
        #import ipdb;ipdb.set_trace()
        
    
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        image_path = self.files[self.split][index].rstrip()
        #print(image_path)
        label_path = os.path.join(self.labels_path,os.path.basename(image_path))
        depth_path = os.path.join(self.depths_path,os.path.basename(image_path))
                            
        _img = Image.open(image_path).convert('RGB')
        _tmp = np.array(Image.open(label_path))
        _tmp = self.map(_tmp)

        _target = Image.fromarray(_tmp)

        _depth = Image.open(depth_path)
        
        sample = {'image': _img, 'label': _target, 'depth': _depth}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def encode_segmap(self, mask):
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def map(self, mask):
        h,w = mask.shape
        map_ids = np.array([0,1,-1,2,-1,-1,-1,-1,-1,-1,-1,3,-1,-1,4,5,6,7,-1,-1,-1,-1,-1,-1,-1,8,9,10,11,12,13,
                       14,-1,15,16,17,18,19,-1,20,21,22,23,-1,24,25,26,27,28,29,30,31,-1,32,33,34,35,36,37,
                       38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,-1,56,57,58,-1])
        mask = map_ids[mask.reshape(-1)].reshape((h,w))
        return mask.astype(np.uint8)


class Cityscapes_test(data.Dataset):

    def __init__(self, root='path/to/datasets/cityscapes', transform=None):
        """
        Cityscapes dataset folder has two folders, 'leftImg8bit' folder for images and 'gtFine_trainvaltest' 
        folder for annotated images with fine annotations 'labels'.
        """
        self.root = root
        self.transform = transform
        self.files = {}
        self.n_classes = 59

        print("Using dataset")
          
            
        #print(self.images_path)
        self.files = listFiles(rootdir=self.root, suffix='.png')#list of the pathes to images

        
        self.ignore_index = 255
        
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image_path = self.files[index].rstrip()

        _img = Image.open(image_path).convert('RGB')
        _img = _img.resize((320, 240), Image.BILINEAR)
        _img = np.array(_img).astype(np.float32).transpose((2, 0, 1))
        _img = torch.from_numpy(_img).float().div(255)
        
        return _img


