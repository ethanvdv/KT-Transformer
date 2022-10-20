import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from einops import rearrange, repeat
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
    
    
    
    return (values1, values2)


# def tensorKaleidoscope()



N = 176

#Lets make the integer mask
img = np.arange(N*N)

#Reshape to square
img = np.reshape(img, [N,N])

#Chose nu and sigma

#downscaling
nu = 15

#Smear Factor
sigma = 17

#Returns the indexes of the "rows" and the indexes of the "cols"
values1, values2 = kaleidoscope(img, nu, sigma)

#Turn them into tensors
rowschange = torch.tensor(values1.copy(), dtype=torch.int64)
rowschange = repeat(rowschange,'h -> h c', c = N)
rowschange = rearrange(rowschange, 'h w -> 1 1 h w')

colschange = torch.tensor(values2.copy(), dtype=torch.int64)
colschange = repeat(colschange,'h -> h c', c = N)
colschange = rearrange(colschange, 'h w -> 1 1 w h')




#From Marlon's Code for loading Shepp_Logan
numCh = 1
# Generate phantom for testing
ph2 = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
# As pytorch tensor
ph = torch.tensor(ph2.copy(), dtype=torch.float)
# Create channel dim (with dummy data for the channel dimension)
ph = torch.unsqueeze(ph, 0)
ph = torch.cat([ph]*numCh, 0)
# Create batch dim
ph = torch.unsqueeze(ph, 0)


#Apply the change in rows
output = torch.gather(ph, 2, rowschange)

#Apply the change in columns
output = torch.gather(output, 3, colschange)

plt.figure(1)
plt.imshow(np.abs(output[0,0,:, :]))
plt.show()




