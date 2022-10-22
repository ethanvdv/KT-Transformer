import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from einops import rearrange, repeat
import torch
from PIL import Image
from torchvision.utils import save_image
import torch
from CreatePhantom import *

def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def createTensor(values1, values2,N):
    '''
    Takes in two vectors. One is the changes in the rows index
    and the other is the change in the column index.
    
    Returns a RGB (ish) Tensor of size (1 1 N N 3)
    '''
    
    #Turn them into tensors
    rowschange = torch.tensor(values1.copy(), dtype=torch.int64)
    rowschange = repeat(rowschange,'h -> h c', c = N)

    #Need to add an extra dimension since gather removes ones
    rowschange = rearrange(rowschange, 'h w -> 1 1 h w 1')

    colschange = torch.tensor(values2.copy(), dtype=torch.int64)
    colschange = repeat(colschange,'h -> h c', c = N)
    
    #Need to add an extra dimension since gather removes ones
    colschange = rearrange(colschange, 'h w -> 1 1 w h 1')

    #Use extra layer for RGB image
    zerosvector = torch.zeros_like(colschange)
    
    changes = torch.cat((rowschange, colschange), 4)
    changes = torch.cat((changes, zerosvector), 4)
    
    #returns as (dummy) b h w c
    return changes


class Kaleidoscope:    
    
    def kaleidoscopeIndexes(N, nu, sigma):
        '''
        Get the Indexes for the "Full" Transform
        
        --- Modified from Jacobs code
        
        '''
        rows = np.arange(N)
        cols = np.arange(N)

        values1 = np.arange(N)
        values2 = np.arange(N)
        for r in rows:
            for c in cols:

                m1 = np.mod(kal_round(N / nu, sigma) * np.mod(r, nu) + sigma * (r // nu), N)
                m2 = np.mod(kal_round(N / nu, sigma) * np.mod(c, nu) + sigma * (c // nu), N)
                
                values1[r] = m1
                values2[c] = m2
        
        return createTensor(values1, values2, N)
    
    def MKTkaleidoscopeIndexes(shift, N):
              
        shifts = ((shift*np.arange(N))%N)

        #Need to pass two elements to be able to use the same function
        output = createTensor(shifts,shifts,N)
        
        return output
        
    def ApplyTransform(img, changes):
        '''
        Used in both the MKT and the Full KT
        '''
        #Apply the change in rows
        rows = torch.gather(img, 2, changes[:,:,:,:,0])

        #Apply the change in columns
        output = torch.gather(rows, 3, changes[:,:,:,:,1])
        
        return output
    
    def pseudoInvKaleidoscope(source, changes2,type=0):
        '''
        Scatter is non-deterministic 
        (from documentation)
        It is "close" but not guranteed to be correct.
        
        Can add more noise
        
        '''
        
        if type == 1:
            x = source.scatter(3,changes2[:,:,:,:,0], source)
    
            output  = x.scatter(2,changes2[:,:,:,:,1], x)
    
        x = source.scatter(3,changes2[:,:,:,:,1],source)

        output  = x.scatter(2,changes2[:,:,:,:,0], x)

        return output
    
    
    
    
    
    