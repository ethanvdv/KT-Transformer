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

class Kaleidoscope:    
       
    @torch.no_grad()
    def MKTkaleidoscopeIndexes(shift, N, case):
        if case == 1:
            shifts = np.ceil((shift*np.arange(N))%N)
        if case == 2:
            shifts = np.floor((shift*np.arange(N))%N)
            
        shifts = torch.tensor(shifts.copy(), dtype=torch.long)

        # _, _, _, n_cols = mat.shape
        
        arange1 = repeat(shifts,'h -> h c', c = N)
        
        # arange2 = rearrange(arange1,'h w -> w h')
        # not sure which method is more efficient

        arange1 = rearrange(arange1, 'h w-> 1 1 h w 1')
        arange2 = rearrange(arange1, '1 1 h w 1 -> 1 1 w h 1')
        zerosvector = torch.zeros_like(arange1)
        
        #Need to pass two elements to be able to use the same function
        # output = createTensor(shifts,shifts,N)
        
        output = torch.cat((arange1, arange2), 4)
        output = torch.cat((output, zerosvector), 4)
        
        return output
    
    def MKTkaleidoscopeIndexes2(shift, N):
        shifts = ((shift*np.arange(N))%N)
            
        shifts = torch.tensor(shifts.copy(), dtype=torch.long)
        
        arange1 = repeat(shifts,'h -> h c', c = N)
        
        arange1 = rearrange(arange1, 'h w-> 1 1 h w 1')
        arange2 = rearrange(arange1, '1 1 h w 1 -> 1 1 w h 1')
        zerosvector = torch.zeros_like(arange1)

        
        output = torch.cat((arange1, arange2), 4)
        output = torch.cat((output, zerosvector), 4)
        
        return output
        
    
    def ApplyKTTransform(input,changes):
        input.to('cuda')
        changes.to('cuda')

        input= input[:,:,changes[0,0,:,:,0],changes[0,0,:,:,1]]
        return input.to('cuda')
    
    def pseudoInvMKTransform(source, changes):
        x = torch.zeros_like(source)
        x[:,:,changes[0,0,:,:,0], changes[0,0,:,:,1]] = source
        return x
        
    
    def KalIndexes(N, nu, sigma):
        rows = torch.arange(N)

        # Make indexes
        m1 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(rows, nu) + sigma * (rows // nu), N)

        arange1 = repeat(m1,'h -> h c', c = N)
        arange1 = rearrange(arange1, 'h w-> 1 1 h w 1')
        arange2 = rearrange(arange1, '1 1 h w 1 -> 1 1 w h 1')
        zerosvector = torch.zeros_like(arange1)
        output = torch.cat((arange1, arange2), 4)
        indexes = torch.cat((output, zerosvector), 4)
        return indexes
    
    
    
    