import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch

# Loading data
class LFDataset(Dataset):
    """Light Field dataset."""

    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        dataSet = h5py.File(opt.dataPath) #[c,y,x,v,u,ind] 
        self.lfSet = dataSet.get('lf')[:].transpose(5,4,3,0,2,1)  #[ind, u, v, c, x, y]
        self.lfSize = dataSet.get('lfSize')[:].transpose(1,0) #[ind, H,W] The spatial resolution of LF
        self.patchSize=opt.patchSize

    def __getitem__(self, idx):
        lf=self.lfSet[idx] #[u, v, c, x, y]
        H,W=self.lfSize[idx] #[H,W]
        lfPatch=ExtractPatch(lf, H, W, self.patchSize) #[u v c x y]
        lfPatch= torch.from_numpy(lfPatch.astype(np.float32)/255)
        sample = {'lf':lfPatch}
        return sample
        
    def __len__(self):
        return self.lfSet.shape[0]



