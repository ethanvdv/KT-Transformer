import time
import numpy as np
from Kaleidoscope import Kaleidoscope
# from phantominator import shepp_logan
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from loadingim import GetDatasetFolder
import torch
from einops import rearrange

import numpy as np

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
    '''

    # img = img // np.abs(sigma) # Normalise image

    # Initialise new image
    imgNew = np.zeros_like(img, dtype = int)

    # Perform kaleidoscope transform
    _, h, w = img.shape

    rows = np.arange(h)
    cols = np.arange(w)

    for r in rows:
        for c in cols:

            m1 = np.mod(kal_round(h / nu, sigma) * np.mod(r, nu) + 
                        sigma * (r // nu), h)
            m2 = np.mod(kal_round(w / nu, sigma) * np.mod(c, nu) + 
                        sigma * (c // nu), w)

            imgNew[:,m1, m2] = img[:,r, c]

    return imgNew


device = 'cuda'

BASE_PATH = '/home/groups/deep-compute/OASIS/'
BATCH_SIZE = 50
offset = 3

torch.cuda.set_device('cuda:0')

ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)


N = 256
totaltime = 0
start = time.time()

mktindexes = Kaleidoscope.KalIndexes(nu=16,sigma=1, N=N)

# Save the indicies
# plt.imsave(f'kt.png', np.abs(mktindexes[0,0,:,:,:].cpu().numpy().astype(np.uint8)))

totalimages = 0 
for batch_index, ph in enumerate(dl):
    ph = rearrange(ph, 'b h w -> b 1 h w').contiguous()
    ph.to(device)
    x = Kaleidoscope.ApplyKTTransform(ph, mktindexes)
    totalimages += x.shape[0]
    
end = time.time()
elapsed = end - start
print("Execution took " + str(elapsed) + " seconds")
print("Average per batch " + str(elapsed/batch_index) + " seconds")
print("Average per image " + str(elapsed/totalimages) + " seconds")
print(totalimages)

totaltime += elapsed



ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)
N = 256


# batch_index = 1
totalimages = 0 
start = time.time()
for batch_index, ph in enumerate(dl):
    # if batch_index != 0:
    #     break
    ph.to(device)
    ph = kaleidoscope(ph,16,1)
    # ph = rearrange(ph, 'b h w -> b 1 h w').contiguous()
    
    totalimages += ph.shape[0]
    
end = time.time()
elapsed = end - start
print("Execution took " + str(elapsed) + " seconds")
print("Average per batch " + str(elapsed/batch_index) + " seconds")
print("Average per image " + str(elapsed/totalimages) + " seconds")
print(totalimages)
    
print("Done")

