import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
from Kaleidoscope import *

#R8 = 22
#R6 = 29
#R4 = 44

N = 208
data = np.zeros((N,N))
h, w =data.shape

R = 4

if R == 8:
    value = 0
if R == 6:
    value = 7
if R == 4:
    value = 26
if R == 7:
    value = 1

indexs =[]
newmasks = np.zeros((N,N))
while len(np.unique(indexs)) < value:
    temp = random.gauss(88, 44)
    temp = round(temp)
    # print(round(temp))
    if (temp > 0) and (temp < 208):
        if (temp > (104-13)) and (temp < (104+13)):
            pass
        else:
            indexs.append(temp)
            newmasks[:,temp] = 255
        


# newmasks[:,88-11:88+11] = 255
newmasks[:,104-13:104+13] = 255
# print(np.count_nonzero(newmasks[:,:]))
plt.imsave(f'mask_R{R}.png', newmasks, cmap = 'gray')
print(f"The percentage is {np.count_nonzero(newmasks)}, {(np.count_nonzero(newmasks)/(208*208))}")


# ph = np.array(ImageOps.grayscale(Image.open("KT-Transformer/KT-And-Fractal/image.png")))
# newer = torch.tensor(ph.copy(), dtype=torch.float)
# newer = rearrange(newer, 'h w -> 1 1 h w')
    
# indexes = Kaleidoscope.MKTkaleidoscopeIndexes(15, N)
# output = Kaleidoscope.ApplyMKTransform(newer, indexes)
# plt.imsave(f'ktthing{R}.png', np.abs(output[0,0,:,:]), cmap = 'gray')

# input = Kaleidoscope.pseudoInvMKTransform(newer, indexes)
# plt.imsave(f'ktinvthing{R}.png', np.abs(input[0,0,:,:]), cmap = 'gray')
# # # #binary image
# # plt.imsave('mask_R12345.png', output,cmap = 'gray')

# output2 = np.fft.ifftshift(output) / np.max(np.abs(output))

# # plt.imsave('mask_R123456.png', output2)

# output3 = np.fft.ifftshift(output2) / np.max(np.abs(output2))

# plt.imsave('mask_R6.png', output)
# print(np.count_nonzero(output))
# print(1/(np.count_nonzero(output)/(176*176)))

# plt.show()
