from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch
warnings.filterwarnings("ignore")
plt.ion()

# Loading data
class LFDatatest(Dataset):
    """Light Field dataset."""

    def __init__(self, opt):
        super(LFDatatest, self).__init__()     
        dataSet = scio.loadmat(opt.testPath)
        self.LFSet = dataSet['lf']  #[ind, u, v, x, y, c]
        self.lfNameSet = dataSet['LF_name'] #[ind, 1] LF name represented by ASCII
        
    def __getitem__(self, idx):
        LF=self.LFSet[idx] #[u, v, x, y, c]
        lfName=''.join([chr(self.lfNameSet[idx][0][0][i]) for i in range(self.lfNameSet[idx][0][0].shape[0])]) 

        LF= torch.from_numpy(LF.astype(np.float32)/255)
        sample = {'LF':LF,'lfName':lfName}
        return sample
        
    def __len__(self):
        return self.LFSet.shape[0]



