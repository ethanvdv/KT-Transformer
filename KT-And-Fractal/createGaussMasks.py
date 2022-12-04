import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps

#R8 = 22
#R6 = 29
#R4 = 44
def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def kaleidoscope(img, nu, sigma):
    '''
    Perform a nu,sigma-Kaleidoscope transform on img
    '''

    img = img // np.abs(sigma) # Normalise image

    # Initialise new image
    imgNew = np.zeros_like(img, dtype = int)

    # Perform kaleidoscope transform
    # h, w = img.shape
    # h = img.shape

    rows = np.arange(176)
    # cols = np.arange(w)

    for r in rows:
        # for c in cols:

        m1 = np.mod(kal_round(h / nu, sigma) * np.mod(r, nu) + 
                    sigma * (r // nu), h)
            # m2 = np.mod(kal_round(w / nu, sigma) * np.mod(c, nu) + 
            #             sigma * (c // nu), w)

        imgNew[:,m1] = img[:,r]

    return imgNew


N = 176
data = np.zeros((N,N))
h, w =data.shape

R = 4
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


# indexs =[]
newmasks = np.zeros((1,176))
# while len(np.unique(indexs)) < value:
#     temp = random.gauss(88, 44)
#     temp = round(temp)
#     # print(round(temp))
#     if (temp > 0) and (temp < 176):
#         if (temp > 77) and (temp < 99):
#             pass
#         else:
#             indexs.append(temp)
#             newmasks[:,temp] = 255
# newmasks[:,88-11:88+11] = 255       
newmasks[:,0:value] = 255     
print(np.count_nonzero(newmasks)) 
# newmasks = kaleidoscope(newmasks, 3, 1)
# newmasks = kaleidoscope(newmasks, 5, 1)
# newmasks = kaleidoscope(newmasks, 7, 1)
values = [3, 5, 7]

for i in range(10):
    index = np.random.randint(0, 3)
    print(values[index])
    newmasks = kaleidoscope(newmasks, values[index], 1)


# newmasks = kaleidoscope(newmasks, 59, 1)
newmasks[:,88-11:88+11] = 255
newmasks2 = np.zeros((176,176))
newmasks2[:,:] = newmasks[:,:]

# newmasks2[:,88-11:88+11] = 255
print(np.count_nonzero(newmasks2))
print(newmasks2.shape)
plt.imsave(f'mask_R{R}.png', newmasks2, cmap = 'gray')

# # # #binary image
# # plt.imsave('mask_R12345.png', output,cmap = 'gray')

# output2 = np.fft.ifftshift(output) / np.max(np.abs(output))

# # plt.imsave('mask_R123456.png', output2)

# output3 = np.fft.ifftshift(output2) / np.max(np.abs(output2))

# plt.imsave('mask_R6.png', output)
# print(np.count_nonzero(output))
# print(1/(np.count_nonzero(output)/(176*176)))

# plt.show()
