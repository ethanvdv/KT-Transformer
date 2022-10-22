import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from CreatePhantom import *

'''
The Multiplication Kaleidoscope Transform by White, in Tensor Form.

Note: The Transform is the opposite to the traditional transform.

i.e For a 1 smear MKT of the number 176, this requires the factors of 175 and 177.

175 has factors of 3 and 59.

We want to apply the "3" transform, we need to actually shift it by 59.

If we want to apply the "59" transform, we need to actually shift by 3.

It is a "psuedo" inverse because for some values, the error is very high.

This is due to the non-deterministic "scatter" function.

For Prime Grids, the transform ALWAYS has a correct inverse, except when Shift = 0,
and when Shift = N.

'''

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


def pseudoInvKaleidoscope(source, Shift, N):
    '''
    Scatter is non-deterministic 
    (from documentation)
    It is "close" but not guranteed to be correct, if anything it adds more noise to it
    '''
    shifts = ((Shift*torch.arange(N))%N)

    _, _, _, n_cols = source.shape
    
    arange1 = repeat(shifts,'h -> h c', c = n_cols)
    
    # arange2 = rearrange(arange1,'h w -> w h')
    # not sure which method is more efficient

    arange1 = rearrange(arange1, 'h w-> 1 1 h w')
    arange2 = rearrange(arange1, '1 1 h w -> 1 1 w h')
    
    x = source.scatter(3,arange2, source)
    
    output  = x.scatter(2,arange1, x)
    
    return output

N = 176

ph = CreatePhantom(N)

shift = 7

output = MultiplicationKT(ph,shift,N)
input = pseudoInvKaleidoscope(output, shift, N)


plt.figure(1)
plt.imshow(np.abs(ph[0,0,:, :]))

plt.figure(2)
plt.imshow(np.abs(output[0,0,:, :]))

plt.figure(3)
plt.imshow(np.abs(input[0,0,:,:]))
plt.show()



# plt.imsave(f'shifting.png',np.abs(output[0,0,:, :]))
# plt.imsave(f'original.png',np.abs(output[0,0,:, :]))

# plt.imsave(f'psuedoin.png',np.abs(input[0,0,:, :]))
