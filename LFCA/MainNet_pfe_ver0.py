import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import scipy.io as scio
from RefNet_pfe_ver0 import RefNet
from Functions import weights_init

warnings.filterwarnings("ignore")
plt.ion()

class StageBlock(torch.nn.Module):
    def __init__(self, opt, bs):
        super(StageBlock,self).__init__()
        # Regularization sub-network
        self.refnet=RefNet(opt,bs)
        self.refnet.apply(weights_init)
        
    def forward(self, mResidual, epoch):
        lfRedisual = self.refnet(mResidual, epoch) 
        return lfRedisual

        
def CascadeStages(block, opt, bs):
    blocks = torch.nn.ModuleList([])
    for _ in range(opt.stageNum):
        blocks.append(block(opt, bs))
    return blocks       
           
               
# Main Network construction
class MainNet(torch.nn.Module):
    def __init__(self,opt):
        super(MainNet,self).__init__()
        # self.channelNum = opt.channelNum
        if opt.measurementNum == 1:    
            self.kernelSize=[opt.angResolution,opt.angResolution]
        if opt.measurementNum == 2:    
            self.kernelSize=[opt.angResolution,opt.angResolution]
        if opt.measurementNum == 4:    
            self.kernelSize=[opt.angResolution,opt.angResolution]
            
        # Shot layer
        self.proj_init=torch.nn.Conv2d(in_channels=opt.channelNum,out_channels=opt.measurementNum,kernel_size=self.kernelSize,bias=False)
        torch.nn.init.xavier_uniform_(self.proj_init.weight.data)
        # Initialize LF from measurements
        self.recon=torch.nn.ConvTranspose2d(in_channels=opt.channelNum,out_channels=opt.channelNum,kernel_size=self.kernelSize,bias=False)
        torch.nn.init.xavier_uniform_(self.recon.weight.data)
        self.initialRefnet=RefNet(opt, True)
        self.initialRefnet.apply(weights_init)
        # Iterative stages
        self.iterativeRecon = CascadeStages(StageBlock, opt, False)

        
    def forward(self, lf, epoch):
        b,u,v,c,x,y=lf.shape
        # Shot
        degLF=self.proj_init(lf.permute(0,4,5,3,1,2).reshape(b*x*y,c,u,v))
        _,m,_,_ = degLF.shape
        degLF = degLF.reshape(b,x,y,m,c).permute(0,3,4,1,2)
        
        # Initialize LF from measurements
        initLF = self.initialRefnet(degLF, epoch)
        out=initLF
        # Reconstructing iteratively
        for stage in self.iterativeRecon:
            mResidual = degLF -self.proj_init(out.reshape(b,u,v,c,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,c,u,v)).reshape(b,x,y,m,c).permute(0,3,4,1,2)
            out = out + stage(mResidual,epoch)
        return out.reshape(b,u,v,c,x,y)

