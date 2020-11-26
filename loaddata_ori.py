import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import os


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataroot):
    
        image_file = 'RGB'
        depth_file = 'Depth'
        Semantic_file = 'Semantic_label'
    
        img_list = os.listdir(os.path.join(dataroot, image_file))
        self.image_names = [os.path.join(dataroot, image_file, i) for i in img_list]
        self.depth_names = [os.path.join(dataroot, depth_file, i) for i in img_list]
        self.Semantic_names = [os.path.join(dataroot, Semantic_file, i) for i in img_list]
        
        
        __imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])
        }
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        
        
        
        if 'Train' in dataroot:
            '''
            self.transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            # RandomRotate(5),
                                            # CenterCrop([304, 228], [152, 114], [152, 114]),
                                            # CenterCrop([304, 224], [152, 112], [152, 112]),
                                            # CenterCrop([320, 240], [160, 120], [160, 120]),
                                            ToTensor(dataroot),
                                            # Lighting(0.1, __imagenet_pca[
                                                # 'eigval'], __imagenet_pca['eigvec']),
                                            # ColorJitter(
                                                # brightness=0.4,
                                                # contrast=0.4,
                                                # saturation=0.4,
                                            # ),
                                            # Normalize(__imagenet_stats['mean'],
                                                        # __imagenet_stats['std'])
                                            ])
        '''
            if 'CAD' in dataroot:
                self.transform=transforms.Compose([
                                                Scale(240, 0.5),
                                                RandomHorizontalFlip(),
                                                ToTensor(dataroot),
                                                ])
            else:
                self.transform=transforms.Compose([
                                                Scale(-1, 0.5),
                                                RandomHorizontalFlip(),
                                                ToTensor(dataroot),
                                                ])
            
                                            
        else:
            '''
            self.transform=transforms.Compose([
                                            Scale(240),
                                            # CenterCrop([304, 224], [152, 112], [152, 112]),
                                            # CenterCrop([320, 240], [160, 120], [160, 120]),
                                            ToTensor(dataroot),
                                            # Normalize(__imagenet_stats['mean'],
                                                        # __imagenet_stats['std'])
                                            ])
            '''
            if 'CAD' in dataroot:
                self.transform=transforms.Compose([
                                                Scale(240, 0.5),
                                                ToTensor(dataroot),
                                                ])
            else:
                self.transform=transforms.Compose([
                                                Scale(-1, 0.5),
                                                ToTensor(dataroot),
                                                ])
                                                
        
        self.map_ids = np.array([0,1,-1,2,-1,-1,-1,-1,-1,-1,-1,3,-1,-1,4,5,6,7,-1,-1,-1,-1,-1,-1,-1,8,9,10,
                                11,12,13,14,-1,15,16,17,18,19,-1,20,21,22,23,-1,24,25,26,27,28,29,30,
                                31,-1,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
                                51,52,53,54,55,-1,56,57,58,-1])
        
    def Semantic_process(self, Semantic):
        img = np.array(Semantic) #(240,320) uint8
        # print('-------------before----------------')
        # print(img.max())
        # print(img.min())
        img_shape = img.shape
        img = img.reshape((-1))
        img_new = np.uint8(self.map_ids[img])
        img_new = img_new.reshape(img_shape)
        # print('-------------after----------------')
        # print(img_new.max())
        # print(img_new.min())
        img_new = Image.fromarray(img_new)
        return img_new
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        depth_name = self.depth_names[idx]
        Semantic_name = self.Semantic_names[idx]

        image = Image.open(image_name)
        image = image.convert('RGB')
        depth = Image.open(depth_name)
        Semantic = Image.open(Semantic_name)
        
        Semantic = self.Semantic_process(Semantic)

        sample = {'image': image, 'depth': depth, 'Semantic': Semantic}

        if self.transform:
            sample = self.transform(sample)
        

        return sample

    def __len__(self):
        return len(self.image_names)

        
        
def getTrainingData(dataroot, batch_size=64, is_refine=False):
    
    transformed_training = depthDataset(dataroot=os.path.join(dataroot, 'Train'))

    dataloader_training = DataLoader(transformed_training, batch_size,drop_last=True,
                                     shuffle=True, num_workers=32, pin_memory=False)

    return dataloader_training


def getTestingData(dataroot, batch_size=64):

    transformed_testing = depthDataset(dataroot=os.path.join(dataroot, 'Test'))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing

