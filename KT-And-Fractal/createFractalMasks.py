import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps

N = 208
R = 9
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
# sampling_mask2 = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/masks/mask_R" + str(R) + ".png")))

data = np.zeros((N,N))

# a = 10
# data[88-a:89+a, 88-a:89+a] = sampling_mask[88-a:89+a, 88-a:89+a]
# data[88-a:88+a, 88-a:88+a] = sampling_mask[88-a:88+a, 88-a:88+a]

# data[data > 100] = 255
# data[data < 100] = 0


a = 20
data[88-a:89+a, 88-a:89+a] = 255


# data[data > 100] = 255
# data[data < 100] = 0


x = torch.tensor(data.copy(), dtype=torch.float)
data = torch.tensor(data.copy(), dtype=torch.float)
data = rearrange(data,'h w -> 1 1 h w')

x = rearrange(x,'h w -> 1 1 h w')
plt.imsave(f'mask_Rasdfdsf1.png', np.abs(x[0,0,:,:]))

# shifts = [3, 59, 5, 7, 25, 35, 29, 58, 87, 89, 89, 173, 179, 43]
# 175 = 5 7 25 35 
# 177 = 3 59
# 174 = 2, 3, 6, 29, 58, 87, 174 
# 178 = 1, 2, 89, 178
# 179 = 1, 179
# 173 = 1, 173
# 172 = 1, 2, 4, 43, 86, 172
# 180 = 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90 and 180.

# shifts = [3, 5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]
# shifts = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 67, 71, 73, 79, 89, 97, 101, 103, 107, 109, 127, 131, 137, 139, 149, 151, 157]
shifts = [3, 5, 7, 59]

for shift in shifts:
    # if np.mod(shift,2) == 1:
        mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
        x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
        # x = Kaleidoscope.ApplyMKTransform(data, mktindexes)
        # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:]))?
        print(f"After {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(176*176)}")
    

# x[0,0,88-a:89+a, 88-a:89+a] = 255
plt.imsave(f'mask_Rasf.png', np.abs(x[0,0,:,:]), cmap='gray')
print(f"After {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(176*176)}")
# data = np.zeros((N,N))
# finalprint = np.zeros((N,N))
# finalprint[0:88,0:88] = np.abs(x[0,0,1:89,1:89])
# finalprint[89:,0:88] = np.abs(x[0,0,89:,1:89])
# finalprint[89:,89:] = np.abs(x[0,0,89:,89:])
# finalprint[0:88,89:] = np.abs(x[0,0,1:89,89:])
# finalprint[:,88] = finalprint[:,87]
# finalprint[88,:] = finalprint[87,:]
# plt.imsave('output.png', np.abs(finalprint[:,:]), cmap='gray')
# print(f"After {shift} the percentage is {np.count_nonzero(finalprint)}, {np.count_nonzero(finalprint)/(176*176)}")
# print(f"After  the percentage is {np.count_nonzero(finalprint)}, {np.count_nonzero(finalprint)/(176*176)}")
