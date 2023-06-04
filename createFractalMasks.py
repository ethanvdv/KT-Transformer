import numpy as np
import matplotlib.pyplot as plt
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fromMask = True

N = 256
data = np.zeros((N,N))

if fromMask: 
    #Load the Mask: 
    R = 1
    sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))

    plt.figure(1)
    plt.imshow(np.abs(sampling_mask[:,:]))
    plt.title(f"Original Fractal")


    a = 8
    data[128-a:129+a, 128-a:129+a] = sampling_mask[111-a:112+a, 111-a:112+a]
    data[data > 100] = 255
    data[data < 100] = 0

else:
    a = 3
    data[129-a:129+a, 129-a:129+a] = 255

x = torch.tensor(data.copy(), dtype=torch.float)
x2 = torch.tensor(data.copy(), dtype=torch.float)
data = torch.tensor(data.copy(), dtype=torch.float)
data = rearrange(data,'h w -> 1 1 h w')

x = rearrange(x,'h w -> 1 1 h w')
x2 = rearrange(x2,'h w -> 1 1 h w')

plt.figure(2)
plt.imshow(np.abs(x[0,0,:,:]))
plt.title(f"Central Pattern")

print(f"The percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

shifts = np.arange(N)

count = 0
for shift in shifts:
    # for g in shifts2:
        count += 1
        mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(N=N,shift=shift)
        x = torch.logical_or(x,Kaleidoscope.ApplyKTTransform(data, mktindexes))
        x2 = torch.logical_or(x2,Kaleidoscope.pseudoInvKTransform(data, mktindexes))
        print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")


print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
print(count)




plt.figure(3)
plt.imshow(np.abs(x[0,0,:,:]))
plt.title(f"Output")

plt.figure(4)
plt.imshow(np.abs(x2[0,0,:,:]))
plt.title(f"Output")


plt.show()