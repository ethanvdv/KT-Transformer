from operator import index
import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from einops import rearrange, repeat
import torch
from PIL import Image
from torchvision.utils import save_image
import torch

'''
This takes a while (About 3 hrs) to Run because of nested for loops.
 
Can definitely be improved/optimised

Can safely not plot nu > 88 because the transforms are too lossy
'''


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
    

    # Perform kaleidoscope transform
    h, w = img.shape
    
    #Wraps so a 0 transform is actually a "N" transform
    #Needed if we divide by zero
    if nu == 0:
        nu = h
        
    if sigma == 0:
        sigma = h
    


    # img = img // np.abs(sigma) # Normalise image


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

def pseudoInvKaleidoscope(source, changes2):
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

nonLossyTransformsNu = []
nonLossyTransformsSigma = []

nonLossyTransformsNu1000 = []
nonLossyTransformsSigma1000 = []



i = 0
for nu in range(N):
    for sigma in range(N):                   
        
        #Returns the indexes of the "rows" and the indexes of the "cols"
        changes = kaleidoscope(indexes, nu, sigma)

        output = tensorKaleidoscope(ph, changes)

        input = pseudoInvKaleidoscope(output, changes)

        hhh = torch.sum(torch.nonzero(ph-input))
        #Error of the inverse
        if hhh == 0:
            print(f"{i} - yes - {nu} - {sigma}")
            nonLossyTransformsNu.append(nu)
            nonLossyTransformsSigma.append(sigma)
            
        elif hhh < 100000:
            print(f"{i} - maybe - {nu} - {sigma}")
            nonLossyTransformsNu1000.append(nu)
            nonLossyTransformsSigma1000.append(sigma)
            
        else:
            print(i)
        
        i+=1
            
                

print(f'Number of "0" transforms {len(nonLossyTransformsNu)}')

print(f'Number of "Psuedo" transforms {len(nonLossyTransformsNu1000)}')

print(i)


plt.figure(1)
plt.plot(nonLossyTransformsNu,nonLossyTransformsSigma)
plt.xlabel('nu')
plt.ylabel('sigma')
plt.title("Transforms which have full inverses - Nu vs Sigma")


plt.figure(2)
plt.plot(nonLossyTransformsNu1000,nonLossyTransformsSigma1000)
plt.xlabel('nu')
plt.ylabel('sigma')
plt.title("Transforms which have full inverses - Nu vs Sigma")



plt.show()