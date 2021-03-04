import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata_ori as loaddata
import loaddata_fastdepth
import util
import os
import numpy as np
import sobel
from models.pdesnet import PDESNet
from models.VNL_loss import VNL_Loss
from datetime import datetime as dat
import warnings
from iouEval import iouEval, getColorEntry
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--backbone_network', 
                help = 'name of backbone network',default='mobilenet_prune')#shufflenet, mobilenet, and darknet

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')##\4e-5
parser.add_argument('--save_folder', default='weights/depth_normal/liteseg_759_SGD_poly_lr1e-4_refine_vnl/')


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
    
def main(ckpt_dir, checkpoint, lr, start_epoch, epochs, batch_size, backbone_network, n_classes, adjust_epoch, dataroot, opt):
    #import ipdb;ipdb.set_trace()
    global args
    args = parser.parse_args()
    model = PDESNet.build(backbone_network,n_classes,args,is_train=True)
        
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
        
    if checkpoint:
        try:
            model.load_state_dict(torch.load(checkpoint)['state_dict'])#(torch.load(checkpoint)['state_dict'])
            print('Load all parameters')
        except:
            model = load_model_my(model, checkpoint)
            print('Load some parameters')
    # else:
        # model.apply(weights_init)
    
    cudnn.benchmark = True
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    ##原始liteseg使用SGD，初始lr为1e-7
    if opt == 'Adam':
        lr = 5e-4
        optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if backbone_network.lower() == 'fastdepth':
        train_loader = loaddata_fastdepth.getTrainingData(batch_size)
        test_loader = loaddata_fastdepth.getTestingData(1)
    else:
        train_loader = loaddata.getTrainingData(dataroot, batch_size)
        test_loader = loaddata.getTestingData(dataroot, 1)

    for epoch in range(start_epoch, epochs):
        
        gap = epochs//(adjust_epoch+1)
        if epoch % gap == (gap-1) and opt == 'SGD':
            # lr = adjust_learning_rate_poly(optimizer, epoch, epochs, lr)
            # lr = adjust_learning_rate(optimizer, epoch, epochs, lr)
            lr = lr*0.1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if opt == 'Adam' and epoch % 100 == 0 and epoch != 0:
            lr = lr*0.5
            optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
        
        train(train_loader, model, optimizer, epoch, ckpt_dir, lr)
        test(test_loader, model, n_classes, epoch, ckpt_dir)
        
    save_checkpoint({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')

L1loss = nn.L1Loss()
criterion_mask = nn.CrossEntropyLoss()
    
    
def criterion_depth(pre_depth, gt_depth):
    pre_depth = pre_depth.view(-1)[gt_depth.view(-1)!=0]
    gt_depth = gt_depth.view(-1)[gt_depth.view(-1)!=0]
    
    loss_depth = F.smooth_l1_loss(pre_depth, gt_depth, size_average=True)
    # loss_depth = L1loss(pre_depth, gt_depth)
    return loss_depth
    
def criterion_depth_weight(pre_depth, gt_depth, Semantic):
    
    masks = Semantic.cpu().float() #(b,h,w)
    
    kernel = np.ones((5,5),np.uint8)
    batch_size = len(masks)
    for id in range(batch_size):
        mask = np.uint8(masks[id].numpy())
        mask[mask>0] = 255
        mask_erosion = cv2.erode(mask,kernel,iterations = 1) #腐蚀
        mask_dilation = cv2.dilate(mask,kernel,iterations = 1) #膨胀
        mask_edge = mask_dilation-mask_erosion
        mask_edge[mask_edge>0] = 5
        mask_edge[mask_edge==0] = 1
        # print(len(np.where(mask_edge>1)[0]))
        mask_edge = torch.from_numpy(mask_edge).float().cuda()
        masks[id] = mask_edge
        
    # print(masks.shape)
    # print(gt_depth.shape)
    masks = masks.view(-1)[gt_depth.view(-1)!=0]
    pre_depth = pre_depth.view(-1)[gt_depth.view(-1)!=0]
    gt_depth = gt_depth.view(-1)[gt_depth.view(-1)!=0]
    
    loss_depth = F.smooth_l1_loss(pre_depth, gt_depth, weight=masks, size_average=True)
    return loss_depth

    
def train(train_loader, model, optimizer, epoch, ckpt_dir, lr):
    
    # vnl_loss = VNL_Loss(input_size = np.array([114, 152]))
    batch_time = AverageMeter()
    losses = AverageMeter()
    

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    start_epoch = time.time()
    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        
        image, depth, Semantic = sample_batched['image'], sample_batched['depth'], sample_batched['Semantic']
        
        '''
        ## 检查数据
        image_debug = np.transpose(np.array(image[0]),(1,2,0))[:,:, (2, 1, 0)]
        image_debug = (image_debug-image_debug.min())/(image_debug.max()-image_debug.min())
        image_debug = np.uint8(image_debug*255)
        
        depth_debug = np.array(depth[0][0])
        # print(depth_debug.shape)
        depth_debug = (depth_debug-depth_debug.min())/(depth_debug.max()-depth_debug.min())
        depth_debug = np.uint8(depth_debug*255)
        import cv2
        debug_file = 'debug_file'
        if os.path.exists(debug_file):
            os.system('rm -r '+debug_file)
        os.system('mkdir -p '+debug_file)
        image_debug_name = os.path.join(debug_file, str(i)+'_image.jpg')
        depth_debug_name = os.path.join(debug_file, str(i)+'_depth.jpg')
        cv2.imwrite(image_debug_name,  image_debug)
        cv2.imwrite(depth_debug_name,  depth_debug)
        
        import ipdb;ipdb.set_trace()
        '''

        depth = depth.cuda()
        image = image.cuda()
        Semantic = Semantic.cuda().long().squeeze()
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        Semantic = torch.autograd.Variable(Semantic)

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()

        output = model(image)
        # import ipdb; ipdb.set_trace()
        pre_depth = output['depth']
        pre_class_mask = output['class_mask']

        '''
        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        #depth_normal = F.normalize(depth_normal, p=2, dim=1)
        #output_normal = F.normalize(output_normal, p=2, dim=1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        
        loss_vnl = vnl_loss(depth/5, output/5)
        

        loss = loss_depth + loss_normal + (loss_dx + loss_dy) 
        '''
        # import ipdb; ipdb.set_trace()
        loss_depth = criterion_depth(pre_depth, depth)
        # loss_depth = criterion_depth_weight(pre_depth, depth, Semantic)
        # import ipdb; ipdb.set_trace()
        # print(Semantic.max())
        loss_mask = criterion_mask(pre_class_mask, Semantic)
        loss = loss_depth + loss_mask
        # loss = loss_mask
        # import ipdb; ipdb.set_trace()
        losses.update(loss.data, image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
   
        '''
        print('Epoch: [{0}][{1}/{2}]\t'
          'Depth: {3:.4f}\t'
          'Normal: {4:.4f}\t'
          'Dx: {5:.4f}\t'
          'Dy: {6:.4f}\t'
          #'VNL: {7:.4f}\t'
          'Loss: {loss.val:.4f} ({loss.avg:.4f})'
          .format(epoch, i, len(train_loader), loss_depth.item(), loss_normal.item(), loss_dx.item(), loss_dy.item(), loss=losses))
        '''
        if i%500 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'D_loss: {3:.4f}\t'
            'M_loss: {4:.4f}\t'
            'epoch_loss: {5:.4f}\t'
            'avg_loss: {6:.4f}\t'
            'lr: {7:.6f}'
            .format(epoch, i, len(train_loader), loss_depth.item(), loss_mask.item(), loss.item(), losses.avg, lr))
          
    end_epoch = time.time()
    print('Epoch %d last %4f minutes.' % (epoch, (end_epoch-start_epoch)/60))
    filename = os.path.join(ckpt_dir, 'checkpoint_mobilenet_' + str(epoch) + '.pth.tar')
    save_checkpoint({'state_dict': model.state_dict()}, filename)
 
def test(test_loader, model, n_classes, epoch, ckpt_dir):
    file = os.path.join(ckpt_dir, 'val.txt')
    model.eval()
    iouEvalVal = iouEval(n_classes)

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

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
        
        #depth_edge = edge_detection(depth)
        #output_edge = edge_detection(output)

        errorSum, totalNumber = eval_depth(pre_depth, depth, errorSum, totalNumber)
        
        iouEvalVal.addBatch(pre_class_mask.max(1)[1].unsqueeze(1).data, Semantic.data)
    
    show_eval_result(errorSum, totalNumber, iouEvalVal, epoch, file)
    
    

def show_eval_result(errorSum, totalNumber, iouEvalVal, epoch, file):

    averageError = util.averageErrors(errorSum, totalNumber)
    model_name = 'checkpoint_'+str(epoch)
    print(model_name)
    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    line_depth = 'RMSE: ' + str(averageError['RMSE']) + '   Detal1: ' + str(averageError['DELTA1'])
    print(line_depth)
    
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    iouPeople = getColorEntry(iou_classes[1])+'{:0.2f}'.format(iou_classes[1]*100) + '\033[0m'
    line_mask = "EPOCH IoU on VAL set: "+str(iouStr)+"% "+ "   People IoU: "+ str(iouPeople)+"%"
    print (line_mask) 
    
    
    f = open(file,'a+')
    f.write(model_name + '\n')
    f.write(line_depth + '\n')
    f.write(line_mask + '\n')
    f.close()
    
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
    
def adjust_learning_rate(optimizer, epoch, epochs, lr, adjust_epoch):
    # lr = lr * (0.1 ** (epoch // 5))
    
    # lr = lr * (0.1 ** (epoch // (epochs//(adjust_epoch+1))))
    lr = lr * 0.1

    # for param_group in optimizer.param_groups:
        # param_group['lr'] = lr

    return lr
    
def adjust_learning_rate_poly(optimizer, epoch, epochs, lr):
    ##原始liteseg的lr更新
    lr_ = lr * ((1 - float(epoch) / epochs) ** 0.9)
    print('(poly lr policy) learning rate: ', lr_)
    
    return lr_

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    torch.save(state, filename)


if __name__ == '__main__':
    #opt = 'Adam'  ##EPFL
    opt = 'SGD'  ##CAD
    data_mode = 'data_CAD60'
    #data_mode = 'data_CAD120'
    #data_mode = 'EPFL'
    print('The dataset is : '+data_mode)
    adjust_epoch = 4 #一共调整几次学习率,opt = 'SGD'时才会用到
    # adjust_epoch = 2 #一共调整几次学习率
    lr = 1e-2 #opt = 'SGD'时才会用到
    # lr = 1e-4
    start_epoch = 0
    epochs = 300 ##CAD100,EPFL300
    # epochs = 100
    # epochs = 50
    # batch_size = 16
    batch_size = 64
    n_classes = 58+1
    # backbone_network = 'mobilenet'
    # backbone_network = 'mobilenet_prune'
    #backbone_network = 'fast_mobilenetv1_aspp'
    backbone_network = 'fast_mobilenetv1_aspp_prune'
    # backbone_network = 'fast_mobilenetv1_aspp_conv'
    #backbone_network = 'fast_mobilenetv1_no_aspp'
    # backbone_network = 'fast_mobilenetv1_aspp_maskinput'
    # backbone_network = 'ESPNet_Encoder'
    # backbone_network = 'fastdepth'
    
    date_name = dat.strftime(dat.now(), '%Y%m%d-%H%M%S')
    ckpt_dir = 'model_out'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_dir = os.path.join(ckpt_dir, backbone_network)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_dir = os.path.join(ckpt_dir,date_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print('Models saved in: '+ckpt_dir)
    # checkpoint = 'model_out/fast_mobilenetv1_aspp/20200806-185758/checkpoint_mobilenet_99.pth.tar'
    # checkpoint = 'model_out/fast_mobilenetv1_aspp/20200807-180736/checkpoint_mobilenet_99.pth.tar'
    checkpoint = None
    dataroot = os.path.join('data', data_mode)
    main(ckpt_dir, checkpoint, lr, start_epoch, epochs, batch_size, backbone_network, n_classes, adjust_epoch, dataroot, opt)

