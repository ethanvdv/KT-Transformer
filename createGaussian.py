import numpy as np
import matplotlib.pyplot as plt
# from CreatePhantom import *
# from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

N = 256
r8 = 0
print(r8)
R = 4

if R == 8:
    value = N//8
if R == 6:
    value = N//6 - r8
if R == 4:
    value = N//4 - r8

indexs =[]
newmasks = np.zeros((N,N))
while len(np.unique(indexs)) < value:
    # temp = random.gauss(N//2, (N - N//2)/3)
    s = np.random.normal(N//2, (N - N//2)/3, 1)
    print(s)
    temp = round(s[0])

    if (temp > 0) and (temp < N):
        if (temp not in indexs):
            indexs.append(temp)
            newmasks[:,temp] = 255
print(f'Number of Lines = {len(indexs)}')

# newmasks[:,N//2 - r8//2:N//2 + r8//2] = 255 # add middle 8th
# print(np.count_nonzero(newmasks[:,:]))
plt.imsave(f'mask_R{R}.png', newmasks, cmap = 'gray')
print(f"The percentage is {np.count_nonzero(newmasks)}, {(np.count_nonzero(newmasks)/(N*N))}")


# N = 256
# data = np.zeros((N,N))

# # Gaussian Mask

# h, w =data.shape


# R = 6
# value = N**2 // R
# # value = (N // R) * N

# while np.count_nonzero(data) < value:
#     s = np.random.normal(N//2, (N - N//2)/3, 2)
#     x = round(s[0])
#     y = round(s[1])
#     if (x > 0) and (x < N):
#         if (y > 0) and (y < N):
#             data[:,y] = 255

# print(np.count_nonzero(data[:,:]))
# plt.imsave(f'mask_R{R}.png', data, cmap = 'gray')
# print(f"The percentage is {np.count_nonzero(data)}, {(np.count_nonzero(data)/(N*N))}")