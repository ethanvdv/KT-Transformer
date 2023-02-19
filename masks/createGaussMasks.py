import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps

#R8 = 22
#R6 = 29
#R4 = 44

N = 176
data = np.zeros((N,N))
h, w =data.shape

R = 7
# sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/masks/mask_R" + str(R) + ".png")))

# values = []

# for i in range(320):
#     if sampling_mask[0,i] > 0:
#         values.append(i)


if R == 8:
    value = 0
if R == 6:
    value = 7
if R == 4:
    value = 22
if R == 7:
    value = 1

indexs =[]
newmasks = np.zeros((N,N))
while len(np.unique(indexs)) < value:
    temp = random.gauss(88, 44)
    temp = round(temp)
    print(round(temp))
    if (temp > 0) and (temp < 176):
        if (temp > 77) and (temp < 99):
            pass
        else:
            indexs.append(temp)
            newmasks[:,temp] = 255
        


newmasks[:,88-11:88+11] = 255
print(np.count_nonzero(newmasks[:,:]))
plt.imsave(f'mask_R{R}.png', newmasks, cmap = 'gray')

# # # #binary image
# # plt.imsave('mask_R12345.png', output,cmap = 'gray')

# output2 = np.fft.ifftshift(output) / np.max(np.abs(output))

# # plt.imsave('mask_R123456.png', output2)

# output3 = np.fft.ifftshift(output2) / np.max(np.abs(output2))

# plt.imsave('mask_R6.png', output)
# print(np.count_nonzero(output))
# print(1/(np.count_nonzero(output)/(176*176)))

# plt.show()
