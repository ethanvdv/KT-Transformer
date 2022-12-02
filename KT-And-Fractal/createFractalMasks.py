import numpy as np
import matplotlib.pyplot as plt
# from phantominator import shepp_logan
# from einops import rearrange
from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps




# x = torch.arange(176)


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

# print(b-a)
# from PIL import Image, ImageDraw
# image = Image.new('1', ((b-a), (b-a))) #create new image, 10x10 pixels, 1 bit per pixel
# draw = ImageDraw.Draw(image)
# draw.ellipse((0, 0, 29, 29), outline ='white')
# # print list(image.getdata())

# plt.imshow(image)
# plt.imsave(f'ahhhh.png', image)

x = torch.tensor(data.copy(), dtype=torch.float)

# sampling_mask = np.array(ImageOps.grayscale(Image.open("ahhhh.png")))

# data[a:b, a:b] = sampling_mask

x = rearrange(x,'h w -> 1 1 h w')
# N = 176
# # x = CreatePhantom(N)
# R = 101
# sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
# # plt.figure(1)
# # plt.imshow(sampling_mask[70:107, 70:107])
# data = np.zeros((N,N))
# data[70:107, 70:107] = sampling_mask[70:107, 70:107]

# data = torch.tensor(data.copy(), dtype=torch.float)
# x = rearrange(data,'h w -> 1 1 h w')
# # x = sampling_mask
# plt.imsave(f'mask_Rasdfdsf.png', np.abs(x[0,0,:,:]))
# N = 176
# # shifts = [5, 7, 59, 3]
shifts = [5, 3, 7, 59]

# # plt.imsave('mask_R12345.png', np.abs(x[0,0,:,:]))

for shift in shifts:
    mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
    # x = x + Kaleidoscope.ApplyMKTransform(x, mktindexes)
    x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(x, mktindexes))
    plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:]))

# x[x == True] = 255
# x[x == False] = 0

# x3 = x
plt.imsave('mask_R125.png', np.abs(x[0,0,:,:]), cmap='gray')
print(f"After {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}")

# # shifts = [5, 7, 59, 3]
# # shifts2 = [3, 59, 7, 5]
# # shifts2 = [7]
# sigma = 1
# # shift2 = int((N - sigma)/7)
# # shift3 = (N - sigma)/7
# shift2 = 7

# shifts2 = [shift2, 7.0]
# for shift in shifts2:
#     x2 = x.clone()
#     mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
#     # x = Kaleidoscope.pseudoInvMKTransform(x, mktindexes)
#     # x = x2 - x
#     x = Kaleidoscope.newPseudoKT(x, mktindexes)
#     # x = torch.logical_and(x, Kaleidoscope.newPseudoKT(x, mktindexes))

#     plt.imsave(f'invmask_R{shift}.png', np.abs(x[0,0,:,:]))


# plt.imsave('mask_Rin.png', np.abs(x[0,0,:,:]))

# x2 = torch.arange(176)
# x2 = repeat(x2,'h -> 1 1 c h', c = N)

# print(np.abs(x[0,0,:,:] - x2[0,0,:,:]))


# def multiKT(image, indices):
#     h, w =image.shape
    
#     rows = np.arange(h)
#     cols = np.arange(w)
    
#     newimage = np.zeros_like(image)
#     for row in rows:
#         for col in cols:
#             x = np.mod(indices*row, h)
#             y = np.mod(indices*col, w)
#             newimage[x,y] += image[row,col] 
            
#     return newimage


# middleshape = False


# if middleshape == True:
#     N = 176
#     data = np.zeros((N,N))
#     h, w =data.shape
        
#     rows = np.arange(h)
#     cols = np.arange(w)
#     a = 88 -15
#     b = 88 +15
#     data[a:b, a:b] = 1

#     data[a:b,a] = 0
#     data[a, a:b] = 0
#     values = np.arange(a,b)
#     for i in values:
#         data[i,i] = 0
#         data[i,-i] = 0       
# else: 
#     N = 176
#     # ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
#     # data = ph

# R = 9
# sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
# # plt.figure(1)
# # plt.imshow(sampling_mask[70:107, 70:107])

# print(np.count_nonzero(sampling_mask// np.max(np.abs(sampling_mask)))/(176*176))





# # sampling = sampling_mask// np.max(np.abs(sampling_mask))
# # # plt.imsave('AAAA.png', sampling,cmap = 'gray')
# N = 176
# data = np.zeros((N,N))
# h, w =data.shape

# data[50:127, 50:127] = sampling_mask[50:127, 50:127]
# plt.imsave('data.png', data)

# data = data// np.max(np.abs(data)) 
# # data[data > 0] = 255

# # plt.figure(2)
# # plt.imshow(data)
# # plt.show()

# # '''
# # factors of 175 = [5, 7, 25, 35, -5, -7, -25, -35]
# # factors of 177 = [3, 59, -3, -59]
# # '''

# # factorslist = [5, 7, -5, -7, 25, 35, -25, -35, 3,  -3, 59, -59]
# factorslist = [5, 7, 25, 35, 59, 3, 16, 11]

# # R8 approx: 0.11805914256198347
# # factorslist = [7, 25]

# #R6 approx: 0.16118930785123967
# # factorslist = [5, 7, 25]

# #R4 approx: 0.24150955578512398
# # factorslist = [5, 7, 25, 35, 59]

# # factorslist = [3]
# # plt.figure(1)
# # plt.imshow(data)


# output = np.zeros_like(data)


# for a in factorslist:
#     print(f'Value = {a}')
#     prev = np.sum(output)
    
#     step = multiKT(data, a)
#     # output += step//np.max(step)
    
#     output += step
#     print(f"After {a} the percentage is {np.count_nonzero(output)/(176*176)}")
#     # output = output // np.max(np.abs(output)) 
    
# # output[:,88:89] = 0
# # output[88:89,:] = 0
# # output = (255*(output)/np.max(np.abs(output)))

# # output = output // np.max(np.max(output))

# # #greyscale
# # plt.imsave('mask_R1234.png', output)



# output[output > 0] = 255
# # # #binary image
# plt.imsave('mask_R12345.png', output,cmap = 'gray')

# output2 = np.fft.ifftshift(output) / np.max(np.abs(output))

# # plt.imsave('mask_R123456.png', output2)

# output3 = np.fft.ifftshift(output2) / np.max(np.abs(output2))

# plt.imsave('mask_R6.png', output2)
# print(1/(np.count_nonzero(output3)/(176*176)))

# plt.show()
