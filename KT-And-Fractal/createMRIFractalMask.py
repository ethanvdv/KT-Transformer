import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random
from dc import *
from torchmetrics import StructuralSimilarityIndexMeasure

fractal = True
fromMask = True

N = 208
data = np.zeros((N,N))

if fractal == True:
    if fromMask: 
        #Load the Mask: 
        R = 81
        sampling_mask = np.array(ImageOps.grayscale(Image.open("ph.png")))
        # sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))
        a = 15
        data = sampling_mask
        # print(sampling_mask)
        # data[104-a:105+a, 104-a:105+a] = sampling_mask[104-a:105+a, 104-a:105+a]
        # data[104-a:104+a, 104-a:104+a] = sampling_mask[104-a:104+a, 104-a:104+a]
        data[data > 30] = 255
        # data[data < 150] = 0

    else:
        # data = sampling_mask
        a = 50
        data[104-a:105+a, 104-a:105+a] = 255

    x = torch.tensor(data.copy(), dtype=torch.float)
    data = torch.tensor(data.copy(), dtype=torch.float)
    data = rearrange(data,'h w -> 1 1 h w')

    x = rearrange(x,'h w -> 1 1 h w')
    plt.imsave(f'mask_Rasdfdsf1.png', np.abs(x[0,0,:,:]))
    print(f"the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

    """
    Factors of 207 and 209:

    shifts = [3, 9, 23, 69, 11, 19]

    """
    #Removing unused values
    # shifts = [3, 9, 23, 69, 11, 19]
    # shifts = [23, 11, 19]
    shifts = [8]
    # shifts = [3, 9, 23, 69, 11, 19]
    # shifts = range(N//2)
    # shifts = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    # shifts = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
    count = 0
    for shift in shifts:
            # if np.mod(shift,2) != 0:
                # if shift not in [3, 9, 23, 69, 11, 19]:
        # if (np.mod(N,shift) == 0):
        #     pass
        #     # if np.mod(shift,2) != 0:
        #         # if shift not in [3, 9, 23, 69, 11, 19]:
        # else: 
        if shift != 0:
            count += 1
            mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
            x = Kaleidoscope.ApplyMKTransform(data, mktindexes)
            # x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
            # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:])) # save the image on each shift
            print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")



    # print(f"The final percentage is {np.count_nonzero(value)}, {np.count_nonzero(value)/(N*N)}")

    plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)
