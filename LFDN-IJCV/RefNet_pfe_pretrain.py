import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class ChannelSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class Conv_spa(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_spa, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):   
        N,c,uv,h,w = x.shape
        x = x.permute(0,2,1,3,4).reshape(N*uv,c,h,w)  
        out = self.op(x)
        #print(out.shape)
        out = out.reshape(N,uv,32,h,w).permute(0,2,1,3,4)
        return out

class Conv_ang(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, angular, bias):
        super(Conv_ang, self).__init__()
        self.angular = angular
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):    
        N,c,uv,h,w = x.shape
        x = x.permute(0,3,4,1,2).reshape(N*h*w,c,self.angular,self.angular)   
        out = self.op(x)
        out = out.reshape(N,h,w,32,uv).permute(0,3,4,1,2)
        return out

class Conv_epi_h(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_epi_h, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):   
        N,c,uv,h,w = x.shape
        x = x.permute(0,3,1,2,4).reshape(N*h,c,uv,w)
        out = self.op(x)
        out = out.reshape(N,h,32,uv,w).permute(0,2,3,1,4)
        return out

class Conv_epi_v(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_epi_v, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        N,c,uv,h,w = x.shape
        x = x.permute(0,4,1,2,3).reshape(N*w,c,uv,h)
        out = self.op(x)
        out = out.reshape(N,w,32,uv,h).permute(0,2,3,4,1)
        return out
    

class Autocovnlayer(nn.Module):
    def __init__(self,dence_num,component_num,angular,bs):
        super(Autocovnlayer, self).__init__()
        self.dence_num = dence_num
        self.component_num = component_num
        self.dence_weight = nn.Parameter(torch.rand(dence_num),requires_grad=True) 
        self.component_weight = nn.Parameter(torch.rand(component_num),requires_grad=True)  
        self.angular = angular
        self.kernel_size = 3

        self.naslayers = nn.ModuleList([
           Conv_spa(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_ang(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, angular = self.angular, bias = bs),
           Conv_epi_h(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_epi_v(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs)
        ])
        self.Conv_all = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1, padding=1, bias = bs)
        self.softmax1 = nn.Softmax(1)
        self.softmax0 = nn.Softmax(0) 
        self.Conv_mixdence = nn.Conv2d(in_channels = 64*self.dence_num, out_channels=64, kernel_size=1, stride=1, padding=0, bias = False)
        self.Conv_mixnas = nn.Conv2d(in_channels = 32*4, out_channels=64, kernel_size=1, stride=1, padding=0, bias = False)     ## 1*1 paddding!!
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x,temperature_1,temperature_2):
        x = torch.stack(x,dim = 0) 
        [fn, N, C, uv, h, w] = x.shape  
        ## generate 2 noise    dim of noise !!!       
        dence_weight_soft = torch.ones((self.dence_num,N))[:,:,None,None,None,None].cuda()
        component_weight_gumbel = torch.ones((self.component_num,N))[:,:,None,None,None,None].cuda()
        
        x = x * dence_weight_soft
        x = x.permute([1,3,0,2,4,5]).reshape([N*uv,fn*C,h,w])    
        x = self.relu(self.Conv_mixdence(x))                              
        x_mix = x.reshape([N,uv,C,h,w]).permute([0,2,1,3,4])   
        layer_label = 0
        nas = []
        for layer in self.naslayers:
            nas_ = layer(x_mix)
            nas.append(nas_)
        
        nas = torch.stack(nas,dim = 0)  
        nas = nas * component_weight_gumbel            
        nas = nas.permute([1,3,0,2,4,5]).reshape([N*uv,self.component_num*32,h,w])   
        nas = self.relu(self.Conv_mixnas(nas))
        ####### add a spa conv  #######
        nas = self.Conv_all(nas)
        nas = nas.reshape(N,uv,C,h,w).permute(0,2,1,3,4)
        nas = self.relu(nas + x_mix)
        return nas

def make_autolayers(opt,bs):
    layers = []
    for i in range( opt.sasLayerNum ):
        layers.append(Autocovnlayer(i+1, opt.component_num, opt.angResolution, bs))
    return nn.Sequential(*layers)
    
class RefNet(nn.Module):
    def __init__(self, opt, bs):        
        super(RefNet, self).__init__()
        self.angResolution = opt.angResolution
        self.lfNum = opt.angResolution * opt.angResolution
        self.epochNum = opt.epochNum
        self.temperature_1 = opt.temperature_1
        self.temperature_2 = opt.temperature_2
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
        self.dence_autolayers = make_autolayers(opt,bs)
        self.sptialSE = SpatialSELayer(32)
        self.channelSE = ChannelSELayer3D(32*2,2)
        self.syn_conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias = bs)

    def forward(self,input,sampleLF,epoch):
        N,u,v,h,w = input.shape   #[b,u,v,c,x,y]
        _,c,_,_ = sampleLF.shape
        if epoch <= 3800:    # T  1 ==> 0.1 
            temperature_1 = self.temperature_1 * (1 - epoch / 4000)
            temperature_2 = self.temperature_2 * (1 - epoch / 4000)
        else:
            temperature_1 = 0.05
            temperature_2 = 0.05
        # feature extraction sample
        feat1 = self.relu(self.conv1(sampleLF))
        feat1 = self.sptialSE(feat1).unsqueeze(2)
        feat1 = feat1.expand(-1,-1,u*v,-1,-1)
        # feature extraction LF
        feat2 = input.reshape(N*u*v,1,h,w) 
        feat2 = self.relu(self.conv0(feat2)) 
        feat2 = feat2.reshape(N,u*v,32,h,w).permute(0,2,1,3,4) 
        feat = torch.cat([feat2,feat1],1)
        feat = self.channelSE(feat)
        feat = [feat]
        for index, layer in enumerate(self.dence_autolayers):
            feat_ = layer(feat,temperature_1,temperature_2)   
            feat.append(feat_)
        feat = self.syn_conv2(feat[-1].permute(0,2,1,3,4).reshape(N*self.lfNum,64,h,w))  
        out = feat.reshape(N,u,v,h,w)
        return out
    
    
    
    