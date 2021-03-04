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

    def __init__(self, csv_file, transform=None):
        #import ipdb;ipdb.set_trace()
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame[0][idx]
        depth_name = self.frame[1][idx]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}
        '''
        ## 检查数据
        image_debug = np.array(image)[:,:, (2, 1, 0)]
        depth_debug = np.array(depth)
        import cv2
        debug_file = 'debug_file'
        if not os.path.exists(debug_file):
            os.system('mkdir -p '+debug_file)
        image_debug_name = os.path.join(debug_file, image_name.split('/')[-1])
        depth_debug_name = os.path.join(debug_file, depth_name.split('/')[-1])
        cv2.imwrite(image_debug_name,  image_debug)
        cv2.imwrite(depth_debug_name,  depth_debug)
        '''
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)

class depthDataset_refine(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataroot, transform=None):
        #import ipdb;ipdb.set_trace()
        self.images = [os.path.join(dataroot, str(i)+'.jpg') for i in range(795)]
        self.depths = [os.path.join(dataroot, str(i)+'.png') for i in range(795)]
        self.transform = transform
        
        #image_name = self.images[0]
        #depth_name = self.depths[0]

        #image = Image.open(image_name)
        #depth = Image.open(depth_name)

        #sample = {'image': image, 'depth': depth}

        #if self.transform:
            #sample = self.transform(sample)
            
    def __getitem__(self, idx):
        image_name = self.images[idx]
        depth_name = self.depths[idx]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)
        
        
def getTrainingData(batch_size=64, is_refine=False):
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
    if is_refine:
        transformed_training = depthDataset_refine(dataroot='./data/nyu2_refine_labeled',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [304, 228]),
                                            Scale((224,224)),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))
    else:
        transformed_training = depthDataset(csv_file='./data/nyu2_train.csv',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [304, 228]),
                                            Scale((224,224)),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training


def getTestingData(batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           Scale((224,224)),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing

