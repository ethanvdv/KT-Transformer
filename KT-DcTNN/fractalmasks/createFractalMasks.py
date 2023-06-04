import numpy as np
import matplotlib.pyplot as plt
# from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fromMask = False

N = 256
data = np.zeros((N,N))

if fromMask: 
    #Load the Mask: 
    R = 2112
    sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/fractalmasks/mask_R" + str(R) + ".png")))
    # sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))
    a = 16
    # data[105-a:105+a, 105-a:107+a] = sampling_mask[105-a:106+a, 105-a:106+a]
    # data[105-a:106+a, 105-a:106+a] = sampling_mask[105-a:106+a, 105-a:106+a]
    data[128-a:129+a, 128-a:129+a] = sampling_mask[105-a:106+a, 105-a:106+a]
    # data[106-a:106+a, 106-a:106+a] = sampling_mask[107-a:107+a, 107-a:107+a] 
    # data = sampling_mask[1:,1:]
    # data[104-a:104+a, 104-a:104+a] = sampling_mask[104-a:104+a, 104-a:104+a]
    data[data > 100] = 255
    data[data < 100] = 0

else:
    # data = sampling_mask
    a = 8
    # data[128-a:129+a, 128-a:129+a] = 255
    data[:, 128-a:129+a] = 255
    data[128-a:129+a,:] = 255
    # data[106-a:107+a, 106-a:107+a] = 255
    # data[104-a:105+a, 104-a:105+a] = 255

x = torch.tensor(data.copy(), dtype=torch.float)
data = torch.tensor(data.copy(), dtype=torch.float)
data = rearrange(data,'h w -> 1 1 h w')

x = rearrange(x,'h w -> 1 1 h w')
plt.imsave(f'mask_Rasdfdsf1.png', np.abs(x[0,0,:,:]))
print(f"the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

shifts = [3, 5, 15, 17, 51, 85]z
count = 0
for shift in shifts:
    # if shift == 0:
    # if np.mod(208,20000) == 0:/
    # if (np.mod(N,shift) == 0):
    #     pass
    # else:
        count += 1
        mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
        x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
        # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:])) # save the image on each shift
        print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
print(count)

