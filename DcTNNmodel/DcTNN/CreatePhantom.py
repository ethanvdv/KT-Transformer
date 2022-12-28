import numpy as np
from phantominator import shepp_logan
import torch

def CreatePhantom(N, numCh=1):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)

    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = torch.unsqueeze(ph, 0)
    ph = torch.cat([ph]*numCh, 0)

    ph = torch.unsqueeze(ph, 0)

    return ph