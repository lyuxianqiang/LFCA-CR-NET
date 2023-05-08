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
        self.lfNameSet = dataSet['LF_name']
        self.patchSize=opt.patchSize

    def __getitem__(self, idx):
        lf=self.lfSet[idx] 
        noiself = self.noiselfSet[idx]   
        lfPatch= torch.from_numpy(lf.astype(np.float32)/255)
        noiselfPatch= torch.from_numpy(noiself.astype(np.float32)/255)
        LF_name = ''.join([chr(self.lfNameSet[idx][0][0][i]) for i in range(self.lfNameSet[idx][0][0].shape[0])]) 
        sample = {'lf':lfPatch,'noiself':noiselfPatch,'lfname':LF_name}
        return sample
        
    def __len__(self):
        return self.lfSet.shape[0]






