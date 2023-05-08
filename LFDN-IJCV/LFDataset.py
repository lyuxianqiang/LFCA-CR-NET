import torch
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch

# Loading data
class LFDataset(Dataset):
    """Light Field dataset."""
    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        dataSet = scio.loadmat(opt.dataPath) 
        self.lfSet = dataSet['lf'].transpose(4,0,1,2,3)
        self.noiselfSet = dataSet['noilf_{}'.format(opt.noiselevel)].transpose(4,0,1,2,3) 
        self.patchSize=opt.patchSize

    def __getitem__(self, idx):
        lf=self.lfSet[idx] 
        noiself = self.noiselfSet[idx]   
        H = self.lfSet.shape[3]
        W = self.lfSet.shape[4]
        lfPatch, noiselfPatch=ExtractPatch(lf, noiself, H, W, self.patchSize)
        lfPatch= torch.from_numpy(lfPatch.astype(np.float32)/255)
        noiselfPatch= torch.from_numpy(noiselfPatch.astype(np.float32)/255)
        sample = {'lf':lfPatch,'noiself':noiselfPatch}
        return sample
        
    def __len__(self):
        return self.lfSet.shape[0]



