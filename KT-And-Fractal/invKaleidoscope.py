from operator import index
import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from einops import rearrange, repeat
import torch
from PIL import Image
from torchvision.utils import save_image
import torch

def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def kaleidoscope(img, nu, sigma):
    '''
    Perform a nu,sigma-Kaleidoscope transform on img
    Modified from Jacobs code
    '''

    img = img // np.abs(sigma) # Normalise image


    # Perform kaleidoscope transform
    h, w = img.shape

    rows = np.arange(h)
    cols = np.arange(w)

    
    values1 = np.arange(h)
    values2 = np.arange(w)
    for r in rows:
        for c in cols:

            m1 = np.mod(kal_round(h / nu, sigma) * np.mod(r, nu) + sigma * (r // nu), h)
            m2 = np.mod(kal_round(w / nu, sigma) * np.mod(c, nu) + sigma * (c // nu), w)
            
            values1[r] = m1
            values2[c] = m2
    
    #Turn them into tensors
    rowschange = torch.tensor(values1.copy(), dtype=torch.int64)
    rowschange = repeat(rowschange,'h -> h c', c = h)

    #Need to add an extra dimension since gather removes ones
    rowschange = rearrange(rowschange, 'h w -> 1 1 h w 1')

    colschange = torch.tensor(values2.copy(), dtype=torch.int64)
    colschange = repeat(colschange,'h -> h c', c = w)
    
    #Need to add an extra dimension since gather removes ones
    colschange = rearrange(colschange, 'h w -> 1 1 w h 1')

    #Use extra layer for RGB image
    #Use extra layer for RGB image
    zerosvector = torch.zeros_like(colschange)
    
    changes = torch.cat((rowschange, colschange), 4)
    changes = torch.cat((changes, zerosvector), 4)
    
    #returns as (dummy) b h w c
    return changes

def tensorKaleidoscope(img, changes):
    
    #Apply the change in rows
    rows = torch.gather(img, 2, changes[:,:,:,:,0])

    #Apply the change in columns
    output = torch.gather(rows, 3, changes[:,:,:,:,1])
    
    return output

def undotensorKaleidoscope(source, changes2):
    '''
    Scatter is non-deterministic 
    (from documentation)
    It is "close" but not guranteed to be correct, if anything it adds more noise to it
    '''
    
    x = source.scatter(3,changes2[:,:,:,:,1],source)
    
    output  = x.scatter(2,changes2[:,:,:,:,0], x)
    
    return output


N = 176
numCh = 1

#From Marlon's Code for loading Shepp_Logan

ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
ph = torch.tensor(ph.copy(), dtype=torch.float)
ph = torch.unsqueeze(ph, 0)
ph = torch.cat([ph]*numCh, 0)
ph = torch.unsqueeze(ph, 0)


#Lets make the integer mask
indexes = np.arange(N*N)

#Reshape to square
indexes = np.reshape(indexes, [N,N])



#Chose nu and sigma
#downscaling
nu = 16

#Smear FactorW
sigma = 1

#Returns the indexes of the "rows" and the indexes of the "cols"
changes = kaleidoscope(indexes, nu, sigma)


# plt.imsave(f'{nu}-{sigma}.png',np.abs(changes[0,0,:,:, :].numpy().astype(np.uint8)))

output = tensorKaleidoscope(ph, changes)

input = undotensorKaleidoscope(output, changes)

#Error of the inverse
print(torch.sum(torch.nonzero(ph-input)))

plt.figure(1)
plt.imshow(np.abs(output[0,0,:, :]))


plt.figure(2)
plt.imshow(np.abs(changes[0,0,:,:, :]))

plt.figure(3)
plt.imshow(np.abs(input[0,0,:,:]))
plt.show()



"""Possibly used in future code"""
# invTensor = torch.tensor(np.arange(176).copy(), dtype=torch.int64)
# InvTensor3 = repeat(invTensor,'h -> h c', c = N)
# print(InvTensor3)
# print(InvTensor3.shape)
# InvTensor1 = rearrange(InvTensor3, 'h w -> 1 1 h w 1')
# # print(InvTensor1.shape)
# InvTensor2 = rearrange(InvTensor3, 'h w -> 1 1 w h 1')

# changes2 = torch.cat((InvTensor1, InvTensor2), 4)



