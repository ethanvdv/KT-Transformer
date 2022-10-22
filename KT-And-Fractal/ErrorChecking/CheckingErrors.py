import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from einops import rearrange, repeat
import torch
import time

def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def kaleidoscope(img, nu, sigma, N):
    '''
    Perform a nu,sigma-Kaleidoscope transform on img
    Modified from Jacobs code
    '''
    if nu == 0:
        nu = N
        
    if sigma == 0:
        sigma = N
    
    
    img = img // np.abs(sigma) # Normalise image


    # Perform kaleidoscope transform
    h, w = img.shape

    rows = np.arange(h)
    # cols = np.arange(w)

    
    values1 = np.arange(h)
    # values2 = np.arange(w)
    
    
    for r in rows:
        # for c in cols:
        m1 = np.mod(kal_round(h / nu, sigma) * np.mod(r, nu) + sigma * (r // nu), h)
        
        # m2 = np.mod(kal_round(w / nu, sigma) * np.mod(c, nu) + sigma * (c // nu), w)
        """same transform, so we're not doing loops with in loops lets comment it out"""
            
        values1[r] = m1
        # values2[c] = m2
    
    u1 = np.unique(values1)

    # u2, _ = np.unique(values2, return_inverse=True)

    return len(u1)


N = 176

#Lets make the integer mask
indexes = np.arange(N*N)

#Reshape to square
indexes = np.reshape(indexes, [N,N])

value = N

aarray = np.arange(value*value)
aarrayes = np.reshape(aarray, [value,value])



for nu in np.arange(value):
    for sigma in np.arange(value):
        #Returns the indexes of the "rows"
        a= kaleidoscope(indexes, nu, sigma, N)
        print(a)
        aarrayes[nu,sigma] = a



plt.figure(1)
plt.imshow(aarrayes, cmap='inferno', vmin=0, vmax=N)
plt.title('Data Lost From Transform')
plt.xlabel('nu')
plt.ylabel('sigma')
plt.colorbar()

# plt.legend()

plt.show()

