import numpy as np
import matplotlib.pyplot as plt
# from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fractal = False
fromMask = False

N = 256
data = np.zeros((N,N))

if fractal == True:
    if fromMask: 
        #Load the Mask: 
        R = 2112
        sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/Prime/fractalmasks/mask_R" + str(R) + ".png")))
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

    """
    Factors of 207 and 209:

    shifts = [3, 9, 23, 69, 11, 19]

    """
    #Removing unused values
    # shifts = range(N//2)
    # shifts = [10, 14, 15, 21]
    # shifts = 3, 5, 15, 17, 51, 85
    # shifts = [15]
    # shifts = [2, 3, 4, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 53, 106]
    #shifts = [3, 5, 15, 17, 51, 85]
    shifts = [15]
    
    # shifts = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    #shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 26, 28, 32, 36, 37, 42, 43, 51, 52, 63, 64, 65, 84, 85, 86, 126, 127, 128, 129, 130, 252, 253, 254, 255, 256, 257, 258, 259, 260]
    count = 0
    for shift in shifts:
        # if shift == 0:
        # if np.mod(208,20000) == 0:/
        # if (np.mod(N,shift) == 0):
        #     pass
        # else:
            count += 1
            mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
            # x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
            x = Kaleidoscope.ApplyMKTransform(data, mktindexes)
            mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=15, N=N)
            output = Kaleidoscope.pseudoInvMKTransform(x, mktindexes)
            # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:])) # save the image on each shift
            print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

    plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)

    plt.imsave(f'mask_R2.png', np.abs(output[0,0,:,:]), cmap='gray')
    print(f"The final percentage is {np.count_nonzero(output[0,0,:,:])}, {np.count_nonzero(output[0,0,:,:])/(N*N)}")
    print(count)
    
else: 
    # Gaussian Mask

    h, w =data.shape

    # r8 = N//16
    # print(r8)
    R = 6

    if R == 8:
        value = N//8
    if R == 6:
        # value = N//6 - r8
        value = N//6
        print(value)
    if R == 4:
        # value = N//4 - r8
        value = N//4

    indexs =[]
    newmasks = np.zeros((N,N))
    while len(np.unique(indexs)) < value:
        temp = random.gauss(N//2, N//8)
        temp = round(temp)

        if (temp > 0) and (temp < N):
            if (temp not in indexs):
                indexs.append(temp)
                newmasks[:,temp] = 255
    print(f'Number of Lines = {len(indexs)}')

    # newmasks[:,N//2 - r8//2:N//2 + r8//2] = 255 # add middle 8th
    # print(np.count_nonzero(newmasks[:,:]))
    plt.imsave(f'mask_R{R}.png', newmasks, cmap = 'gray')
    print(f"The percentage is {np.count_nonzero(newmasks)}, {(np.count_nonzero(newmasks)/(N*N))}")
