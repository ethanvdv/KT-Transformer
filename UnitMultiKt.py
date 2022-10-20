import torch
import numpy as np
from phantominator import shepp_logan
import matplotlib.pyplot as plt
from einops import rearrange, repeat

def MultiplicationKT(mat,Shift, N):

    shifts = ((Shift*torch.arange(N))%N)

    _, _, _, n_cols = mat.shape
    
    arange1 = repeat(shifts,'h -> h c', c = n_cols)
    
    # arange2 = rearrange(arange1,'h w -> w h')
    # not sure which method is more efficient

    arange1 = rearrange(arange1, 'h w-> 1 1 h w')
    arange2 = rearrange(arange1, '1 1 h w -> 1 1 w h')

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
