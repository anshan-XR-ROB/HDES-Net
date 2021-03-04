import argparse

import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata_ori as loaddata
import loaddata_fastdepth
import os
import numpy as np
import cv2
from models.pdesnet import PDESNet
from iouEval import iouEval, getColorEntry
import util
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load_model_my(model, checkpoint):
    model_dict = model.state_dict()
    # import ipdb; ipdb.set_trace()
    pretrained_dict = torch.load(checkpoint)['state_dict']
    
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    
    return model

    
def main(checkpoint, backbone_network, n_classes, save_file, dataroot):
    #import ipdb;ipdb.set_trace()
    
    model = PDESNet.build(backbone_network,n_classes,is_train=False)
    # if 'yangmei' in backbone_network:
        # model = model.cuda()
    # else:
        # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    try:
        model.load_state_dict(torch.load(checkpoint)['state_dict'])#(torch.load(checkpoint)['state_dict'])
    except:
        model.load_state_dict(torch.load(checkpoint))
    print('Loading checkpoint ' + checkpoint)
    # else:
        # model.apply(weights_init)
    
    cudnn.benchmark = True
    
    if backbone_network.lower() == 'fastdepth':
        # train_loader = loaddata_fastdepth.getTrainingData(batch_size)
        test_loader = loaddata_fastdepth.getTestingData(1)
    else:
        # train_loader = loaddata.getTrainingData(dataroot,batch_size)
        test_loader = loaddata.getTestingData(dataroot, 1)

    
    test(test_loader, model, n_classes, save_file)
        

 
def test(test_loader, model, n_classes, save_file):
    
    model.eval()
    iouEvalVal = iouEval(n_classes)
    
    totalNumber = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    img_num = len(test_loader)
    for i, sample_batched in enumerate(test_loader):
        
        
        image, depth, Semantic = sample_batched['image'], sample_batched['depth'], sample_batched['Semantic']
        
        depth = depth.cuda()
        image = image.cuda()
        Semantic = Semantic.cuda().long()
        image = torch.autograd.Variable(image, volatile=True)
        depth = torch.autograd.Variable(depth, volatile=True)
        Semantic = torch.autograd.Variable(Semantic, volatile=True)

        output = model(image)
        # import ipdb; ipdb.set_trace()
        pre_depth = output['depth']
        pre_class_mask = output['class_mask']
        
        save_result(i, image, depth, Semantic, pre_depth, pre_class_mask, n_classes, save_file)
        
        errorSum, totalNumber = eval_depth(pre_depth, depth, errorSum, totalNumber)
        # import ipdb; ipdb.set_trace()
        iouEvalVal.addBatch(pre_class_mask.max(1)[1].unsqueeze(1).data, Semantic.data)
        
        if i % 100 == 0:
            print(str(i)+' / '+str(img_num))
            # show_eval_result(errorSum, totalNumber, iouEvalVal)
        # if i > 10:
            # break
    
    show_eval_result(errorSum, totalNumber, iouEvalVal)
    
    

def show_eval_result(errorSum, totalNumber, iouEvalVal):
    #只显示两个指标
    # averageError = util.averageErrors(errorSum, totalNumber)
    # averageError['RMSE'] = np.sqrt(averageError['MSE'])
    # line_depth = 'RMSE: ' + str(averageError['RMSE']) + '   Detal1: ' + str(averageError['DELTA1'])
    # print(line_depth)
    #显示全部指标
    averageError = util.averageErrors(errorSum, totalNumber)
    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    averageError_keys = averageError.keys()
    for key in averageError_keys:
        line_depth = key+': ' + str(averageError[key])
        print(line_depth)
    
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    iouPeople = getColorEntry(iou_classes[1])+'{:0.2f}'.format(iou_classes[1]*100) + '\033[0m'
    line_mask = "EPOCH IoU on VAL set: "+str(iouStr)+"% "+ "   People IoU: "+ str(iouPeople)+"%"
    print (line_mask)
    
    
    
def eval_depth(pre_depth, depth, errorSum, totalNumber):
    # output = torch.nn.functional.upsample(pre_depth, size=[depth.size(2),depth.size(3)], mode='bilinear')

    #depth_edge = edge_detection(depth)
    #output_edge = edge_detection(output)
    
    pre_depth[pre_depth<0] = 0
    pre_depth[pre_depth>10] = 10
    
    batchSize = depth.size(0)
    totalNumber = totalNumber + batchSize
    errors = util.evaluateError(pre_depth, depth)
    errorSum = util.addErrors(errorSum, errors, batchSize)
    return errorSum, totalNumber
            
def normalize_img(img, n_classes, img_name):

    img_shape = img.shape
    if img_shape[1] == 3:
        img = np.array(img.squeeze().cpu()).transpose(1,2,0)[:,:,(2,1,0)] #(h,w,3) BGR
    elif img_shape[1] == n_classes:
        img = np.array(img.max(1)[1].squeeze().cpu())
    else:
        if 'pre' in img_name:
            img = img.squeeze().cpu().detach().numpy()
        else:
            img = np.array(img.squeeze().cpu())
            
    if 'mask' in img_name:
        # img[img>1] = 0
        img[img>1] = 1
    if 'depth' in img_name:
        img[img<0] = 0
        img[img>10] = 10
        img = np.uint8(img/10*255)
    else:
        img = np.uint8(((img-img.min())/(img.max()-img.min()))*255)
        
    if not 'image' in img_name:
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    
    return img
 
def save_result(i, image, depth, Semantic, pre_depth, pre_class_mask, n_classes, save_file):
    pre_depth[depth==0] = 0
    imgs = [image, depth, Semantic, pre_depth, pre_class_mask]
    img_names = ['image', 'depth_gt', 'mask_gt', 'depth_pre', 'mask_pre']
    for img_i, img in enumerate(imgs):
        img_name = img_names[img_i]
        img_numpy = normalize_img(img, n_classes, img_name)
        img_name = os.path.join(save_file, str(i)+'_'+img_name+'.jpg')
        if 'depth' in img_name:
            plt.imsave(img_name, img_numpy,cmap='rainbow')
            # cv2.imwrite(img_name, img_numpy)
        else:
            cv2.imwrite(img_name, img_numpy)


if __name__ == '__main__':
        
    # data_mode = 'data_CAD120'
    # data_mode = 'data_CAD60'
    data_mode = 'EPFL'
    
    n_classes = 58+1
    # backbone_network = 'mobilenet'
    # backbone_network = 'mobilenet_prune'
    # backbone_network = 'fast_mobilenetv1_aspp'
    backbone_network = 'fast_mobilenetv1_aspp_prune'
    # backbone_network = 'ESPNet_Encoder_yangmei'
    # backbone_network = 'fast_mobilenetv1_aspp_maskinput'
    # backbone_network = 'fastdepth'
    save_file = 'result'
    if os.path.exists(save_file):
        os.system('rm -r '+save_file)
    os.system('mkdir -p '+save_file)
    
   
    # checkpoint = 'model_out/fast_mobilenetv1_aspp/20200806-185758/checkpoint_mobilenet_99.pth.tar' #aspp_60
    # checkpoint = 'model_out/fast_mobilenetv1_aspp/20200806-185446/checkpoint_mobilenet_99.pth.tar' #aspp_120
    # checkpoint = 'model_out/fast_mobilenetv1_aspp/20200818-105502/checkpoint_mobilenet_299.pth.tar' #aspp_epfl
    # checkpoint = 'model_out/fast_mobilenetv1_aspp_prune/20200814-170629/checkpoint_mobilenet_99.pth.tar' #prune_60
    # checkpoint = 'model_out/fast_mobilenetv1_aspp_prune/20200817-145650/checkpoint_mobilenet_99.pth.tar' #prune_120
    checkpoint = 'our_model/EPFL_checkpoint_mobilenet.pth.tar' #prune_epfl
    # checkpoint = None
    
    dataroot = os.path.join('data', data_mode)
    main(checkpoint, backbone_network, n_classes, save_file, dataroot)

