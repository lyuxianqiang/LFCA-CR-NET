import torch
import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import scipy.io as scio
from RefNet_pfe_test import RefNet
from Functions import weights_init

warnings.filterwarnings("ignore")
plt.ion()

class StageBlock(torch.nn.Module):
    def __init__(self, opt, bs):
        super(StageBlock,self).__init__()
        # Regularization sub-network
        self.refnet=RefNet(opt,bs)
        self.refnet.apply(weights_init)
    def forward(self, mResidual,sampleLF,epoch):
        lfRedisual = self.refnet(mResidual,sampleLF,epoch) 
        return lfRedisual

        
def CascadeStages(block, opt, bs):
    blocks = torch.nn.ModuleList([])
    for _ in range(opt.stageNum):
        blocks.append(block(opt, bs))
    return blocks       

class MainNet(torch.nn.Module):
    def __init__(self,opt):
        super(MainNet,self).__init__()
        self.kernelSize=[opt.angResolution,opt.angResolution]
        self.angularnum = opt.angResolution 
        # global average
        self.avglf = torch.nn.AvgPool2d(kernel_size=self.kernelSize,stride = None, padding = 0)   
        self.proj_init = torch.nn.Conv2d(in_channels=1,out_channels=7,kernel_size=self.kernelSize,bias=False)
        torch.nn.init.xavier_uniform_(self.proj_init.weight.data)
        self.initialRefnet=RefNet(opt, True)
        self.initialRefnet.apply(weights_init)     
        # Iterative stages
        self.iterativeRecon = CascadeStages(StageBlock, opt, False)

    def forward(self, noiself, epoch):
        b,u,v,x,y=noiself.shape
        avgLF = self.avglf(noiself.permute(0,3,4,1,2).reshape(b,x*y,u,v))
        avgLF = avgLF.reshape(b,x,y,1).permute(0,3,1,2) 
        projLF = self.proj_init(noiself.permute(0,3,4,1,2).reshape(b*x*y,1,u,v))
        projLF = projLF.reshape(b,x,y,7).permute(0,3,1,2)
        sampleLF = torch.cat([avgLF,projLF],1)
        # Initialize LF 
        out = self.initialRefnet(noiself,sampleLF,epoch) 

        # Reconstructing iteratively
        for stage in self.iterativeRecon:
            avgLF = self.avglf(out.permute(0,3,4,1,2).reshape(b,x*y,u,v))
            avgLF = avgLF.reshape(b,x,y,1).permute(0,3,1,2)   
            projLF = self.proj_init(out.permute(0,3,4,1,2).reshape(b*x*y,1,u,v))
            projLF = projLF.reshape(b,x,y,7).permute(0,3,1,2)
            sampleLF = torch.cat([avgLF,projLF],1)
            out = out + stage(out,sampleLF,epoch)
        return out 
