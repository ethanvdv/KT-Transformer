import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
from einops import rearrange

N = 176
R = 5
# sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/KT-And-Fractal/mask_R" + str(R) + ".png")))
print(sampling_mask.shape)
print(f"The percentage is {np.count_nonzero(sampling_mask)}, {np.count_nonzero(sampling_mask)/(176*176)}")
# data = np.zeros((N,N))
print(f"The percentage is {np.count_nonzero(sampling_mask)}, {1/(np.count_nonzero(sampling_mask)/(176*176))}")



# values = [0, 1, 172, 87, 86, 2, 171, 58, 115, 3, 170, 116, 57, 88, 85, 130, 43, 4, 169, 44, 129, 59, 114, 104, 69, 5, 168, 173]

# for value in values:
#     print(np.mod(value, 88))


# data[0:87,0:87] = sampling_mask[0:87,0:87]
# data[0:87,90:] = sampling_mask[0:87,87:]
# data[90:,0:87] = sampling_mask[87:,0:87]
# data[90:,90:] = sampling_mask[87:,87:]

# print(f"The percentage is {np.count_nonzero(data)}, {np.count_nonzero(data)/(176*176)}")

# data[87:90,:] = 255
# data[:,87:90] = 255

# print(f"The percentage is {np.count_nonzero(data)}, {np.count_nonzero(data)/(176*176)}")
# # data[87,:] = data[86]
# # data[88,:] = data[86]
# # data[87,:] = data[86]
# # data[87:89,]
# plt.imsave('output.png', np.abs(data[:,:]), cmap='gray')
# # data[65:112, 65:112] = sampling_mask[65:112, 65:112]
# a = 16
# data[88-a:89+a, 88-a:89+a] = sampling_mask[88-a:89+a, 88-a:89+a]
# # data[88-a:89+a, 88-a:89+a] = 255
# data[data > 100] = 255
# data[data < 100] = 0


# a = 11
# data[88-a:89+a, 88-a:89+a] = 255


# data[data > 100] = 255
# data[data < 100] = 0



# x = torch.tensor(data.copy(), dtype=torch.float)
# data = torch.tensor(data.copy(), dtype=torch.float)
# data = rearrange(data,'h w -> 1 1 h w')

# x = rearrange(x,'h w -> 1 1 h w')
# plt.imsave(f'mask_Rasdfdsf.png', np.abs(x[0,0,:,:]))

# # shifts = [3, 5, 7, 25, 35, 59]
# # shifts = [3, 5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
# shifts = [3, 5, 6, 7, 25, 35, 59, 9, 10, 15, 18, 29, 30, 45]
# # shifts = [3, 5, 7, 25, 29, 35, 58, 59, 87,  89]
# # shifts = range(59)

# for shift in shifts:
#     # if np.mod(shift,2) == 1:
#         mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
#         x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
#         print(f"After {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(176*176)}")

# plt.imsave('mask_R125.png', np.abs(x[0,0,:,:]), cmap='gray')
# # x[0,0,88-a:89+a, 88-a:89+a] = 255

# data = np.zeros((N,N))
# finalprint = np.zeros((N,N))
# finalprint[0:88,0:88] = np.abs(x[0,0,1:89,1:89])
# finalprint[89:,0:88] = np.abs(x[0,0,89:,1:89])
# print(finalprint[:,0:88].shape)
# finalprint[89:,89:] = np.abs(x[0,0,89:,89:])
# finalprint[0:88,89:] = np.abs(x[0,0,1:89,89:])
# finalprint[:,88] = finalprint[:,87]
# finalprint[88,:] = finalprint[87,:]
# plt.imsave('output.png', np.abs(finalprint[:,:]), cmap='gray')
# # print(f"After {shift} the percentage is {np.count_nonzero(finalprint)}, {np.count_nonzero(finalprint)/(176*176)}")
# print(f"After  the percentage is {np.count_nonzero(finalprint)}, {np.count_nonzero(finalprint)/(176*176)}")
