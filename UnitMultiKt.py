import torch
import numpy as np
from phantominator import shepp_logan
import matplotlib.pyplot as plt
from einops import rearrange

def MultiplicationKT(mat,Shift, N):

    shifts = ((Shift*torch.arange(N))%N)
    
    _, _, n_rows, n_cols = mat.shape
    arange1 = (shifts).view((n_rows, 1)).repeat((1, n_cols))
    arange2 = torch.transpose(arange1,0,1) 
    
    arange1 = rearrange(arange1, 'h w-> 1 1 h w')
    arange2 = rearrange(arange2, 'h w -> 1 1 h w')
    out1 = torch.gather(mat, 2, arange1)

    return torch.gather(out1, 3, arange2)


N = 176
numCh = 1

#From Marlon's Code for loading Shepp_Logan

# Generate phantom for testing
ph2 = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
# As pytorch tensor
ph = torch.tensor(ph2.copy(), dtype=torch.float)
# Create channel dim (with dummy data for the channel dimension)
ph = torch.unsqueeze(ph, 0)
ph = torch.cat([ph]*numCh, 0)
# Create batch dim
ph = torch.unsqueeze(ph, 0)

Shift = 5

output = MultiplicationKT(ph,Shift,N)

plt.imsave(f'shifting.png',np.abs(output[0,0,:, :]))
plt.imsave(f'original.png',np.abs(ph[0,0,:, :]))
