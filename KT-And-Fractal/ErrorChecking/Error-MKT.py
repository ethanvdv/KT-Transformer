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

valuesPseudo = []

# for i in range(-N, 0):
#     output = MultiplicationKT(ph,i,N)
#     input = pseudoInvKaleidoscope(output, i, N)
#     #Error of the inverse
#     hhh = torch.sum(torch.nonzero(ph-input))
#     valuesPseudo.append(hhh)


# for i in range(N):
#     output = MultiplicationKT(ph,i,N)
#     input = pseudoInvKaleidoscope(output, i, N)
#     #Error of the inverse
    
#     hhh = torch.sum(torch.nonzero(ph-input))
#     valuesPseudo.append(hhh)



for i in [5, 7, 25, 35, -5, -7, -25, -35, 3, 59, -3, -59]:
    output = MultiplicationKT(ph,i,N)
    input = pseudoInvKaleidoscope(output, i, N)
    #Error of the inverse
    
    hhh = torch.sum(torch.nonzero(ph-input))
    valuesPseudo.append(hhh)



print(len(valuesPseudo))



plt.figure(1)
plt.plot([5, 7, 25, 35, -5, -7, -25, -35, 3, 59, -3, -59], valuesPseudo,'r*')


plt.figure(2)
plt.imshow(np.abs(ph[0,0,:, :]))


plt.figure(3)
plt.imshow(np.abs(output[0,0,:, :]))

plt.figure(4)
plt.imshow(np.abs(input[0,0,:,:]))
plt.show()






# plt.imsave(f'shifting.png',np.abs(output[0,0,:, :]))
# plt.imsave(f'original.png',np.abs(output[0,0,:, :]))

# plt.imsave(f'psuedoin.png',np.abs(input[0,0,:, :]))
