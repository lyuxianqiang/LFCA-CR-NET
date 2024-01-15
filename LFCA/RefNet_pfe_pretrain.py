import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np
import math

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
        out = out.reshape(N,uv,16,h,w).permute(0,2,1,3,4)
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
        out = out.reshape(N,h,w,16,uv).permute(0,3,4,1,2)
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
        out = out.reshape(N,h,16,uv,w).permute(0,2,3,1,4)
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
        out = out.reshape(N,w,16,uv,h).permute(0,2,3,4,1)
        return out
    

class Autocovnlayer(nn.Module):
    def __init__(self,dence_num,component_num,angular,bs):
        super(Autocovnlayer, self).__init__()
        self.dence_num = dence_num
        self.component_num = component_num
        self.dence_weight = nn.Parameter(torch.rand(dence_num),requires_grad=True) #[N,*,c,u,v,h,w]
        self.component_weight = nn.Parameter(torch.rand(component_num),requires_grad=True)  #[N,*,c,,,]
        self.angular = angular
        self.kernel_size = 3

        self.naslayers = nn.ModuleList([
           Conv_spa(C_in = 32, C_out = 16, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_ang(C_in = 32, C_out = 16, kernel_size = self.kernel_size, stride = 1, padding = 1, angular = self.angular, bias = bs),
           Conv_epi_h(C_in = 32, C_out = 16, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_epi_v(C_in = 32, C_out = 16, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs)
        ])
        self.Conv_all = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
        self.softmax1 = nn.Softmax(1)
        self.softmax0 = nn.Softmax(0) 
        self.Conv_mixdence = nn.Conv2d(in_channels = 32*self.dence_num, out_channels=32, kernel_size=1, stride=1, padding=0, bias = False)
        self.Conv_mixnas = nn.Conv2d(in_channels = 16*4, out_channels=32, kernel_size=1, stride=1, padding=0, bias = False)     ## 1*1 paddding!!
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x,temperature_1,temperature_2):
        x = torch.stack(x,dim = 0) 
        [fn, N, C, uv, h, w] = x.shape
        dence_weight_soft = torch.ones((self.dence_num,N))[:,:,None,None,None,None].cuda()
        component_weight_gumbel = torch.ones((self.component_num,N))[:,:,None,None,None,None].cuda()
        
        x = x * dence_weight_soft
        x = x.permute([1,3,0,2,4,5]).reshape([N*uv,fn*C,h,w])    
        x = self.relu(self.Conv_mixdence(x))                               
        x_mix = x.reshape([N,uv,32,h,w]).permute([0,2,1,3,4])   
        layer_label = 0
        nas = []
        for layer in self.naslayers:
            nas_ = layer(x_mix)
            nas.append(nas_)
        
        nas = torch.stack(nas,dim = 0)   
        nas = nas * component_weight_gumbel           
        #print("nas-shape:",nas.shape)
        nas = nas.permute([1,3,0,2,4,5]).reshape([N*uv,self.component_num*16,h,w])
        nas = self.relu(self.Conv_mixnas(nas))           
        ####### add a spa conv
        nas = self.Conv_all(nas)
        #print("outshape0:",nas.shape)
        nas = nas.reshape(N,uv,32,h,w).permute(0,2,1,3,4)
        nas = self.relu(nas + x_mix)
        #print("outshape1:",nas.shape)
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
        self.measurementNum = opt.measurementNum
        self.lfNum = opt.angResolution * opt.angResolution
        self.epochNum = opt.epochNum
        self.temperature_1 = opt.temperature_1
        self.temperature_2 = opt.temperature_2
        
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
        self.conv1 = nn.Conv2d(in_channels=self.measurementNum, out_channels=self.lfNum, kernel_size=3, stride=1, padding=1, bias = bs)
        
        #self.altblock = make_Altlayer(opt)
        self.dence_autolayers = make_autolayers(opt,bs)
        #self.syn_conv1 = nn.Conv3d(in_channels=64, out_channels=self.angResolution * self.angResolution, kernel_size=(self.measurementNum,3,3), stride=1, padding=(0,1,1))
        self.syn_conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias = bs)
        

    def forward(self, input, epoch):
        N,uvInput,_,h,w = input.shape   #[N,uvInput,1,h,w]
        if epoch <= 3800:    # T  1 ==> 0.1 
            temperature_1 = self.temperature_1 * (1 - epoch / 4000)
            temperature_2 = self.temperature_2 * (1 - epoch / 4000)
        else:
            temperature_1 = 0.05
            temperature_2 = 0.05
            
        # feature extraction
        feat = input.reshape(N*uvInput,1,h,w)
        feat = self.relu(self.conv0(feat)) 
        
        # LF feature extration
        feat = feat.reshape(N,uvInput,32,h,w).permute(0,2,1,3,4).reshape(N*32,uvInput,h,w) 
        feat = self.relu(self.conv1(feat))     
        feat = feat.reshape(N,32,self.lfNum,h,w)  
        
        # autoConv
        feat = [feat]
        for index, layer in enumerate(self.dence_autolayers):
            #print("feat-shape:",feat.shape)
            feat_ = layer(feat,temperature_1,temperature_2)   
            feat.append(feat_)
        feat = self.syn_conv2(feat[-1].permute(0,2,1,3,4).reshape(N*self.lfNum,32,h,w))
        out = feat.reshape(N,self.lfNum,1,h,w)
        return out
    
    