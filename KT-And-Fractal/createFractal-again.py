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
            newimage[x,y] += image[row,col]
            
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
sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))
# plt.figure(1)
# plt.imshow(sampling_mask[70:107, 70:107])


N = 176
data = np.zeros((N,N))
h, w =data.shape

data[70:107, 70:107] = sampling_mask[70:107, 70:107]

# data[data > 0] = 255

# plt.figure(2)
# plt.imshow(data)
# plt.show()

# '''
# factors of 175 = [5, 7, 25, 35, -5, -7, -25, -35]
# factors of 177 = [3, 59, -3, -59]
# '''

factorslist = [5, 7, 25, 35, -5, -7, -25, -35, 3, 59, -3, -59]

# factorslist = [3]
# plt.figure(1)
# plt.imshow(data)


output = np.zeros_like(data)


for a in factorslist:
    print(f'Value = {a}')
    prev = np.sum(output)
    
    step = multiKT(data, a)
    output += step//np.max(step)



# #greyscale
plt.imsave('mask_R111.png', output, cmap = 'gray')

# plt.imshow(output)

output[output > 0] = 255

print(np.count_nonzero(output))
# #binary image
plt.imsave('mask_R100.png', output,cmap = 'gray')

plt.show()
