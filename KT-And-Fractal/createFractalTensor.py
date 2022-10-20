import torch
import numpy as np
from phantominator import shepp_logan
import matplotlib.pyplot as plt
from einops import rearrange, repeat

def MultiplicationKT(mat,Shift, N):

    shifts = ((Shift*torch.arange(N))%N)

    _, _, _, n_cols = mat.shape
    
    arange1 = repeat(shifts,'h -> h c', c = n_cols)
    
    '''not sure which method is more efficient'''
    
    # arange2 = rearrange(arange1,'h w -> w h')
    # arange2 = rearrange(arange1, '1 1 h w -> 1 1 h w')
    
    arange1 = rearrange(arange1, 'h w-> 1 1 h w')
    arange2 = rearrange(arange1, '1 1 h w -> 1 1 w h')

    out1 = torch.gather(mat, 2, arange1)

    return torch.gather(out1, 3, arange2)


# Fractal Version

N = 176
data = np.zeros((N,N))
h, w =data.shape
    
rows = np.arange(h)
cols = np.arange(w)
a = 88 -15
b = 88 +15
data[a:b, a:b] = 1

data[a:b,a] = 0
data[a, a:b] = 0
values = np.arange(a,b)
for i in values:
    data[i,i] = 0
    data[i,-i] = 0    

data = torch.tensor(data.copy(), dtype=torch.float)
data = rearrange(data, 'h w-> 1 1 h w')

factorslist = [5, 7, 25, 35, -5, -7, -25, -35, 3, 59, -3, -59]

output = torch.zeros_like(data)

for a in factorslist:
    print(f'Value = {a}')
    step = MultiplicationKT(data,a,N)
    output += step
    
#Plot
plt.figure(1)
plt.imshow(np.abs(data[0,0,:, :]))    

plt.figure(2)
plt.imshow(np.abs(output[0,0,:, :]))    
    

plt.show()