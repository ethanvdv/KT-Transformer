import numpy as np
import matplotlib.pyplot as plt
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random
from phantominator import shepp_logan
def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph

N = 512
ph = CreatePhantom(N).to('cpu')  


nu, sigma, F, G, i, j = 16, 1, 1, 4, 0, 1

indexTensor = Kaleidoscope.AffineKSIndexes(N,nu,sigma,F,G,i,j)


x = Kaleidoscope.ApplyKTTransform(ph,indexTensor)
inv = Kaleidoscope.pseudoInvKTransform(x, indexTensor)

plt.figure(1)
plt.imshow(np.abs(x[0,0,:,:].numpy()))

plt.figure(2)
plt.imshow(np.abs(inv[0,0,:,:].numpy()))
plt.show()
