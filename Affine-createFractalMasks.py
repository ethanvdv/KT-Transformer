import numpy as np
import matplotlib.pyplot as plt
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fromMask = False

N = 256
for i in range(10):
    data = np.zeros((N,N))

    if fromMask: 
        #Load the Mask: 
        R = 1
        sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))

        # plt.figure(1)
        # plt.imshow(np.abs(sampling_mask[:,:]))
        # plt.title(f"Original Fractal")


        a = 10
        data[128-a:129+a, 128-a:129+a] = sampling_mask[111-a:112+a, 111-a:112+a]
        # data[111-a:112+a, 111-a:112+a] = sampling_mask[111-a:112+a, 111-a:112+a]
        data[data > 100] = 255
        data[data < 100] = 0

    else:
        a = 7
        data[129-a:129+a, 129-a:129+a] = 255

    x = torch.tensor(data.copy(), dtype=torch.float)
    x2 = torch.tensor(data.copy(), dtype=torch.float)
    data = torch.tensor(data.copy(), dtype=torch.float)
    data = rearrange(data,'h w -> 1 1 h w')

    x = rearrange(x,'h w -> 1 1 h w')
    x2 = rearrange(x2,'h w -> 1 1 h w')

    # plt.figure(2)
    # plt.imshow(np.abs(x[0,0,:,:]))
    plt.imsave(f'fractals/central.png',np.abs(x[0,0,:,:].numpy()))
    # plt.title(f"Central Pattern")

    print(f"The percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

    shifts = np.arange(N//2)
    # shifts = [1, 3, 5, 15, 17, 51, 85, 255]

    count = 0
    for nu in shifts:
        if nu != 0:
            count += 1
            mktindexes = Kaleidoscope.AffineKSIndexes(N=N,nu=1,sigma=nu,F=1,G=1,i=i,j=0)
            x = torch.logical_or(x,Kaleidoscope.ApplyKTTransform(data, mktindexes))
            x2 = torch.logical_or(x2,Kaleidoscope.pseudoInvKTransform(data, mktindexes))
            print(f"After: {nu} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")


    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)




    # plt.figure(3)
    # plt.imshow(np.abs(x[0,0,:,:]))
    # plt.title(f"Output")
    plt.imsave(f'fractals/output-{i}.png',np.abs(x[0,0,:,:].numpy()), cmap='gray')

    # plt.figure(4)
    # plt.imshow(np.abs(x2[0,0,:,:]))
    # plt.title(f"Output")
    plt.imsave(f'fractals/inv-output-{i}.png',np.abs(x2[0,0,:,:].numpy()))
    x = torch.zeros(N)
    x2 = torch.zeros(N)
    

    # plt.show()