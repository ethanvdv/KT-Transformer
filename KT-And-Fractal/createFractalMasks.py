import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps

N = 208
R = 211
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
# sampling_mask2 = np.array(ImageOps.grayscale(Image.open("KT-Transformer/KT-And-Fractal/mask_R" + str(R) + ".png")))

data = np.zeros((N,N))

# a = 69
a = 13
# a = 16
# data[88-a:89+a, 88-a:89+a] = sampling_mask[88-a:89+a, 88-a:89+a]
# data[88-a:88+a, 88-a:88+a] = sampling_mask[88-a:88+a, 88-a:88+a]
# data[104-a:105+a, 104-a:105+a] = sampling_mask[88-a:89+a, 88-a:89+a]
# data[88-a:88+a, 88-a:88+a] = sampling_mask[88-a:88+a, 88-a:88+a]
# data[104-a:105+a, 104-a:105+a] = sampling_mask2[104-a:105+a, 104-a:105+a]
data[104-a:104+a, 104-a:104+a] = sampling_mask[105-a:105+a, 105-a:105+a]
# data[data > 100] = 255
# data[data < 100] = 0


# a = 30  
# data[104-a:105+a, 104-a:105+a] = 255


data[data > 100] = 255
data[data < 100] = 0


x = torch.tensor(data.copy(), dtype=torch.float)
data = torch.tensor(data.copy(), dtype=torch.float)
data = rearrange(data,'h w -> 1 1 h w')

x = rearrange(x,'h w -> 1 1 h w')
plt.imsave(f'mask_Rasdfdsf1.png', np.abs(x[0,0,:,:]))

shifts = [3, 9, 23, 69, 11, 19]
# shifts = [3]
# shifts = [3, 23, 69, 11, 19, 103, 5, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105]
shifts = range(104)

for shift in shifts:
    if np.mod(shift,2) == 1:
        if np.mod(104, shift) != 0:
            mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
            x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
            # x = Kaleidoscope.ApplyMKTransform(data, mktindexes)
            # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:]))
            print(f"After {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    

# x[0,0,88-a:89+a, 88-a:89+a] = 255
plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
print(f"After {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
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
