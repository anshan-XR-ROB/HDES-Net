import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

class MobileNet(nn.Module):
    def __init__(self, relu6=True):
        super(MobileNet, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2, relu6), 
            conv_dw( 32,  64, 1, relu6),
            conv_dw( 64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6),
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        for i in range(14):
            x = self.model[i](x)
            if i == 3:
                low_level_features = x
        return x, low_level_features
        
class MobileNet_res(nn.Module):
    def __init__(self, relu6=True):
        super(MobileNet_res, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2, relu6), 
            conv_dw( 32,  64, 1, relu6),
            conv_dw( 64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6), #(128,56,56)
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6), #(1024,7,7)
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        # res_ind = [3,5,7,8,9,10,11]
        res_ind = [3,5,7]
        for i in range(14):
            out = self.model[i](x)
            if i in res_ind:
                x = x + out
            else:
                x = out
            if i == 3:
                low_level_features = x
        return x, low_level_features

def main():
    import torchvision.models
    model = MobileNet(relu6=True)
    model = torch.nn.DataParallel(model).cuda()
    model_filename = os.path.join('results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
    if os.path.isfile(model_filename):
        print("=> loading Imagenet pretrained model '{}'".format(model_filename))
        checkpoint = torch.load(model_filename)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded Imagenet pretrained model '{}' (epoch {}). best_prec1={}".format(model_filename, epoch, best_prec1))

def flops(model,input_size):
    input = torch.rand(1, 3, input_size, input_size).cuda()
    flops, params = profile(model, inputs=(input, ))
    from thop import clever_format
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    #import pdb;pdb.set_trace()

def FPS(model,input_size):
    model.eval()
    import time
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
    # model = MobileNet()
    model = MobileNet_res()
    model.cuda()
    input_size = 224
    # flops(model,input_size)
    # model_name = model.name()
    # print('==== model name: '+model_name+'   decoder: '+decoder+' =========')
    FPS(model,input_size)