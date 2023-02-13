import numpy as np
import matplotlib.pyplot as plt
# from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fractal = True
fromMask = True

N = 256
data = np.zeros((N,N))

if fractal == True:
    if fromMask: 
        #Load the Mask: 
        R = 2574
        # sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/Prime/fractalmasks/mask_R" + str(R) + ".png")))
        sampling_mask = np.array(ImageOps.grayscale(Image.open("zf.png")))
        a = 10
        # data[128-a:129+a, 128-a:129+a] = sampling_mask[105-a:106+a, 105-a:106+a]
        # data[128-a:129+a, 128-a:129+a] = sampling_mask[128-a:129+a, 128-a:129+a]
        data = sampling_mask
        # data[data > 100] = 255
        # data[data < 100] = 0

    else:
        # data = sampling_mask
        a = 11
        data[128-a:129+a, 128-a:129+a] = 255

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
    shifts = [15]
    # shifts = [2, 3, 4, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 53, 106]
    #shifts = [3, 5, 15, 17, 51, 85]
    # shifts = [15, 17, 51, 85]
    # shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 26, 28, 32, 36, 37, 42, 43, 51, 52, 63, 64, 65, 84, 85, 86, 126, 127, 128, 129, 130, 252, 253, 254, 255, 256, 257, 258, 259, 260] 
    # shifts = [1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 17, 23, 32, 37, 43, 51, 64, 85, 86, 127, 128, 129, 253, 254, 255, 256, 257, 258, 259]
    # shifts = [2, 4, 8, 16, 32]
    # shifts = [16]
    count = 0
    for shift in shifts:
        if np.mod(N,shift) != 0:
            # mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
            # x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
            mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=15, N=N)
            output = Kaleidoscope.ApplyFullTransform(x, mktindexes)

            print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")


    plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)

else: 
    # Gaussian Mask

    h, w =data.shape

    r8 = N//8
    print(r8)
    R = 6

    if R == 8:
        value = 0
    if R == 6:
        value = N//6 - r8
    if R == 4:
        value = N//4 - r8

    indexs =[]
    newmasks = np.zeros((N,N))
    while len(np.unique(indexs)) < value:
        temp = random.gauss(N//2, N//4)
        temp = round(temp)

        if (temp > 0) and (temp < N):
            if (temp > (N//2 - r8//2)) and (temp < (N//2 + r8//2)):
                #make sure it's not going over previous lines
                pass
            else:
                indexs.append(temp)
                newmasks[:,temp] = 255
    print(f'Number of Lines = {len(indexs) + r8}')

    newmasks[:,N//2 - r8//2:N//2 + r8//2] = 255 # add middle 8th
    # print(np.count_nonzero(newmasks[:,:]))
    plt.imsave(f'mask_R{R}.png', newmasks, cmap = 'gray')
    print(f"The percentage is {np.count_nonzero(newmasks)}, {(np.count_nonzero(newmasks)/(N*N))}")
