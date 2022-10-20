import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from einops import rearrange

def multiKT(image, indices):
    h, w =image.shape
    
    rows = np.arange(h)
    cols = np.arange(w)
    
    newimage = np.zeros_like(image)
    for row in rows:
        for col in cols:
            x = np.mod(indices*row, h)
            y = np.mod(indices*col, w)
            newimage[x,y] = image[row,col]
            
    return newimage


middleshape = True


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
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    data = ph


'''
factors of 175 = [5, 7, 25, 35, -5, -7, -25, -35]
factors of 177 = [3, 59, -3, -59]
'''

factorslist = [5, 7, 25, 35, -5, -7, -25, -35, 3, 59, -3, -59]


plt.figure(1)
plt.imshow(data)


output = np.zeros_like(data)


for a in factorslist:
    print(f'Value = {a}')
    prev = np.sum(output)
    
    step = multiKT(data, a)
    output += step



#greyscale
plt.imsave('mask_R210.png', output, cmap = 'gray')

output[output > 0] = 255

#binary image
plt.imsave('mask_R20.png', output)

plt.show()
