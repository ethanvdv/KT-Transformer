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
            
    def MKTkaleidoscopeIndexes(shift, N):
        shifts = ((shift*np.arange(N))%N)
            
        shifts = torch.tensor(shifts.copy(), dtype=torch.long)
        
        arange1 = repeat(shifts,'h -> h c', c = N)
        
        arange1 = rearrange(arange1, 'h w-> 1 1 h w 1')
        arange2 = rearrange(arange1, '1 1 h w 1 -> 1 1 w h 1')
        zerosvector = torch.zeros_like(arange1)
        
        output = torch.cat((arange1, arange2), 4)
        output = torch.cat((output, zerosvector), 4)
        
        return output
        
    def KalIndexes(N, nu, sigma):
        rows = torch.arange(N)
        # Make indexes
        m1 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(rows, nu) + sigma * (torch.div(rows, nu, rounding_mode='floor')), N)
        
        arange1 = repeat(m1,'h -> h c', c = N)
        arange1 = rearrange(arange1, 'h w-> 1 1 h w 1')
        arange2 = rearrange(arange1, '1 1 h w 1 -> 1 1 w h 1')
        # zerosvector = torch.zeros_like(arange1)
        indexes = torch.cat((arange1, arange2), 4)
        # indexes = torch.cat((output, zerosvector), 4)
        return indexes.to('cuda'), torch.unique(m1).shape[0], 
    
    
    def KaleidoscopeShuffleIndexes(N, nu, sigma, F, G):
        

        rows = ((F*torch.arange(N))).to('cuda')
        cols = ((G*torch.arange(N))).to('cuda')
        
        # Make indexes
        m1 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(rows, nu) + sigma * (torch.div(rows, nu, rounding_mode='floor')), N).to('cuda')
        m2 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(cols, nu) + sigma * (torch.div(cols, nu, rounding_mode='floor')), N).to('cuda')

        
        arange1 = repeat(m1,'h -> h c', c = N)
        arange1 = rearrange(arange1, 'h w-> 1 1 h w 1')

        arange2 = repeat(m2,'h -> h c', c = N)
        arange2 = rearrange(arange2, 'h w-> 1 1 w h 1')

        zerosvector = torch.zeros_like(arange1)
        output = torch.cat((arange1, arange2), 4)
        indexes = torch.cat((output, zerosvector), 4)

        return indexes.to('cuda'), torch.unique(m1).shape[0], torch.unique(m2).shape[0]
    
    
    def KSIndexCheck(N, nu, sigma, F, G):
        
        rows = ((F*torch.arange(N)))
        cols = ((G*torch.arange(N)))
        
        # Make indexes
        m1 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(rows, nu) + sigma * (torch.div(rows, nu, rounding_mode='floor')), N).to('cuda')
        m2 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(cols, nu) + sigma * (torch.div(cols, nu, rounding_mode='floor')), N).to('cuda')

        return torch.unique(m1).shape[0], torch.unique(m2).shape[0]
    
    
    def ApplyKTTransform(input,changes):
        x = torch.zeros_like(input)
        x[:,:,changes[0,0,:,:,0], changes[0,0,:,:,1]] = input
        return x
        
    
    def pseudoInvKTransform(input, changes):
        input= input[:,:,changes[0,0,:,:,0],changes[0,0,:,:,1]]
        return input
    
   
    def FKSIndexCheck(N, nu, sigma, F):
        rows = ((F*torch.arange(N)))
        # Make indexes
        m1 = torch.remainder(kal_round(N / nu, sigma) * torch.remainder(rows, nu) + sigma * (torch.div(rows, nu, rounding_mode='floor')), N).to('cuda')

        return torch.unique(m1).shape[0]