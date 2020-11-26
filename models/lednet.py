import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
import math


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    # flatten
    x = x.view(batchsize,-1,height,width)
    
    return x
    

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()
		
        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		nn.BatchNorm2d(out_ch, eps=1e-3),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)


# after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock (nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock,self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        oup_inc = chann//2
        
        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)       
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)       
        
    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input) #按通道分成两份

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
    
    
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1,output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out,2)



class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,32)

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()

        for x in range(0, 3):
            self.layers1.append(SS_nbt_module(32, 0.03, 1))
        

        self.layers1.append(DownsamplerBlock(32,64))
        

        for x in range(0, 2):
            self.layers1.append(SS_nbt_module(64, 0.03, 1))
  
        self.layers2.append(DownsamplerBlock(64,128))

        for x in range(0, 1):    
            self.layers2.append(SS_nbt_module(128, 0.3, 1))
            self.layers2.append(SS_nbt_module(128, 0.3, 2))
            self.layers2.append(SS_nbt_module(128, 0.3, 5))
            self.layers2.append(SS_nbt_module(128, 0.3, 9))
            
        for x in range(0, 1):    
            self.layers2.append(SS_nbt_module(128, 0.3, 2))
            self.layers2.append(SS_nbt_module(128, 0.3, 5))
            self.layers2.append(SS_nbt_module(128, 0.3, 9))
            self.layers2.append(SS_nbt_module(128, 0.3, 17))
                    

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)
        self.depth_conv = nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        
        output = self.initial_block(input)

        for layer in self.layers1:
            output = layer(output)
        low_feature = output  #1/4*1/4*64
        
        for layer in self.layers2:
            output = layer(output)

        if predict:
            semantic = self.output_conv(output)
            depth = self.depth_conv(output)
            #depth = F.interpolate(depth, size=(int(math.ceil(input.size()[-2]/4)),
            #                    int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
            return semantic, depth

        return low_feature, output

class Interpolate(nn.Module):
    def __init__(self,size,mode):
        super(Interpolate,self).__init__()
        
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
    def forward(self,x):
        x = self.interp(x,size=self.size,mode=self.mode,align_corners=True)
        return x
        

class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
        # midddle branch
        self.mid = nn.Sequential(
		Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)
		
        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)
		
        self.down3 = nn.Sequential(
		Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
		Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
	)
		
        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)
	
        self.depth = nn.Conv2d(1, out_ch, kernel_size=1, stride=1)
    def forward(self, x):
        
        h = x.size()[2]
        w = x.size()[3]
        
        #b1 = self.branch1(x)
        # b1 = Interpolate(size=(h, w), mode="bilinear")(b1)
        #b1= interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)
	
        mid = self.mid(x)
		
        x1 = self.down1(x) #1/16,1
        x2 = self.down2(x1) #1/32,1
        x3 = self.down3(x2) #1/64,1
        # x3 = Interpolate(size=(h // 4, w // 4), mode="bilinear")(x3)
        x3= interpolate(x3, size=x2.size()[2:], mode="bilinear", align_corners=True)	 #1/32,1
        x2 = self.conv2(x2) 
        x = x2 + x3
        # x = Interpolate(size=(h // 2, w // 2), mode="bilinear")(x)
        x= interpolate(x, size=x1.size()[2:], mode="bilinear", align_corners=True)
       		
        x1 = self.conv1(x1)
        x = x + x1
        # x = Interpolate(size=(h, w), mode="bilinear")(x)
        x= interpolate(x, size=(h, w), mode="bilinear", align_corners=True)  #1/8,1
        		
        #x = torch.mul(x, mid)

        x = x + mid
        x = self.depth(x)
       
       
        return x
          

class ASPP(nn.Module):
    
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate=rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            #self.conv1 =SeparableConv2d(planes,planes,3,1,1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()
   
            #self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes,planes,kernel_size,1,padding,rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        
        

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        #x = self.relu(x)
        if self.rate!=1:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.relu1(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
        
        

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = APN_Module(in_ch=128,out_ch=1)

        self.conv = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(256, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

        rates = [1, 3, 6, 9]
        self.aspp1 = ASPP(256, 96, rate=rates[0])
        self.aspp2 = ASPP(256, 96, rate=rates[1])
        self.aspp3 = ASPP(256, 96, rate=rates[2])
        self.aspp4 = ASPP(256, 96, rate=rates[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(256, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
                                             
        self.conv1 =SeparableConv2d(480+256,96,1)
        self.bn1 = nn.BatchNorm2d(96)
        
        self.last_conv = nn.Sequential(#nn.Conv2d(24+96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(64+96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       #nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU())
        
        self.semantic = nn.Conv2d(96, num_classes, kernel_size=1, stride=1)
        # self.upsample = Interpolate(size=(512, 1024), mode="bilinear")
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
  
    def forward(self, low_feature, input):
        
        depth = self.apn(input)
        depth = interpolate(depth, size=(212, 256), mode="bilinear", align_corners=True)
        
        semantic = self.conv(input)
        x1 = self.aspp1(semantic)
        x2 = self.aspp2(semantic)
        x3 = self.aspp3(semantic)
        x4 = self.aspp4(semantic)
        x5 = self.global_avg_pool(semantic)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        semantic = torch.cat((semantic,x1, x2, x3, x4, x5), dim=1)

        semantic = self.conv1(semantic)
        semantic = self.bn1(semantic)
        semantic = self.relu(semantic)
        semantic = F.interpolate(semantic, size=(low_feature.size()[-2],
                                low_feature.size()[-1]), mode='bilinear', align_corners=True)
                                
        semantic = torch.cat((semantic, low_feature), dim=1)
        semantic = self.last_conv(semantic)
        semantic = self.semantic(semantic)
        semantic = F.interpolate(semantic, size=(212, 256), mode='bilinear', align_corners=True)
        return semantic, depth


# LEDNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=True):
        #import ipdb;ipdb.set_trace()
        if only_encode:
            return self.encoder.forward(input, predict=True)  #semantic,depth
        else:
            low_feature, output = self.encoder(input)    
            return self.decoder.forward(low_feature, output)

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
    model = Net(num_classes=59)
    model.cuda()
    input_size = 224
    # flops(model,input_size)
    FPS(model,input_size)
            
