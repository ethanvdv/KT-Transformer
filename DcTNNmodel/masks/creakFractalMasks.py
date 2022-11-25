import numpy as np
import matplotlib.pyplot as plt
# from phantominator import shepp_logan
# from einops import rearrange
from PIL import Image, ImageOps
def multiKT(image, indices):
    h, w =image.shape
    
    rows = np.arange(h)
    cols = np.arange(w)
    
    newimage = np.zeros_like(image)
    for row in rows:
        for col in cols:
            x = np.mod(indices*row, h)
            y = np.mod(indices*col, w)
            newimage[x,y] += image[row,col] * indices
            
    return newimage


middleshape = False


if middleshape == True:
    N = 176
    data = np.zeros((N,N))
    h, w =data.shape
        
    rows = np.arange(h)
    cols = np.arange(w)
    a = 88 -15
    b = 88 +15
    data[a:b, a:b] = 1

    data[a:b,a] = 0
    data[a, a:b] = 0
    values = np.arange(a,b)
    for i in values:
        data[i,i] = 0
        data[i,-i] = 0       
else: 
    N = 176
    # ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    # data = ph

R = 9
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/masks/mask_R" + str(R) + ".png")))
# plt.figure(1)
# plt.imshow(sampling_mask[70:107, 70:107])

print(np.count_nonzero(sampling_mask// np.max(np.abs(sampling_mask)) ))

sampling = sampling_mask// np.max(np.abs(sampling_mask))
# plt.imsave('AAAA.png', sampling,cmap = 'gray')
N = 176
data = np.zeros((N,N))
h, w =data.shape

data[50:127, 50:127] = sampling_mask[50:127, 50:127]


data = data// np.max(np.abs(data)) 
# data[data > 0] = 255

# plt.figure(2)
# plt.imshow(data)
# plt.show()

# '''
# factors of 175 = [5, 7, 25, 35, -5, -7, -25, -35]
# factors of 177 = [3, 59, -3, -59]
# '''

# factorslist = [5, 7, 25, 35, 59, -5, -7, -25, -35, 3,  -3, -59]
# factorslist = [5, 7, 25, 35, 59, 3]
factorslist = [7, 25, 59]

# factorslist = [3]
# plt.figure(1)
# plt.imshow(data)


output = np.zeros_like(data)


for a in factorslist:
    print(f'Value = {a}')
    prev = np.sum(output)
    
    step = multiKT(data, a)
    # output += step//np.max(step)
    
    output += step
    # output = output // np.max(np.abs(output)) 
    

output = (255*(output)/np.max(np.abs(output)))



# output = output // np.max(np.max(output))

# #greyscale
plt.imsave('mask_R121.png', output)
print(np.count_nonzero(output))
plt.imsave('data.png', data, cmap = 'gray')

# # plt.imshow(output)

# output[output < 128] +=128


# # #binary image
plt.imsave('mask_R120.png', output,cmap = 'gray')

output2 = np.fft.ifftshift(output) / np.max(np.abs(output))

plt.imsave('mak.png', output2)

output3 = np.fft.ifftshift(output2) / np.max(np.abs(output2))

plt.imsave('mak2.png', output3)

plt.show()
