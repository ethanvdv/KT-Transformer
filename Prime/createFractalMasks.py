import numpy as np
import matplotlib.pyplot as plt
# from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fractal = True
fromMask = True

N = 211
data = np.zeros((N,N))

if fractal == True:
    if fromMask: 
        #Load the Mask: 
        R = 0
        sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/Prime/fractalmasks/mask_R" + str(R) + ".png")))
        # sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))
        a = 10
        # data[105-a:105+a, 105-a:107+a] = sampling_mask[105-a:106+a, 105-a:106+a]
        # data[105-a:106+a, 105-a:106+a] = sampling_mask[105-a:106+a, 105-a:106+a]
        # data[104-a:105+a, 104-a:105+a] = sampling_mask[105-a:106+a, 105-a:106+a]
        data[106-a:106+a, 106-a:106+a] = sampling_mask[107-a:107+a, 107-a:107+a] 
        # data = sampling_mask
        # data[104-a:104+a, 104-a:104+a] = sampling_mask[104-a:104+a, 104-a:104+a]
        data[data > 100] = 255
        data[data < 100] = 0

    else:
        # data = sampling_mask
        a = 10
        data[106-a:106+a, 106-a:106+a] = 255
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
    shifts = [10, 14, 15, 21]
    # shifts = [2, 3, 4, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 53, 106]
    # shifts = [2, 3, 4, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 53, 106, -2, -3, -4, -5, -6, -7, -10, -14, -15, -21, -30, -35, -42, -70, -105, -53, -106]
    # shifts = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    count = 0
    for shift in shifts:
        # if shift == 0:
        # if np.mod(208,20000) == 0:/
        # if (np.mod(N,shift) == 0) or (np.mod(shift,2) == 0):
        count += 1
        mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
        x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
        # mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=-1*shift, N=N)
        # x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
        # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:])) # save the image on each shift
        print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

    
    # Add the central R8 Mask:     
    # x[0,0,:,104-13:104+13] = 255
    # value = np.abs(x[0,0,:,:])
    # value[value>0] = 2
    # value[value == 0] = 255
    # value[value == 2] = 0
    # plt.imsave(f'mask_R1.png', np.abs(value), cmap='gray')
    # print(f"The final percentage is {np.count_nonzero(value)}, {np.count_nonzero(value)/(N*N)}")

    plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)
else: 
    # Gaussian Mask

    h, w =data.shape

    r8 = N//8
    R = 8

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

        if (temp > 0) and (temp < 208):
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
