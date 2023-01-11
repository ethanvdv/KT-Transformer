import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random
from dc import *
from torchmetrics import StructuralSimilarityIndexMeasure

fractal = True
fromMask = True

N = 208
data = np.zeros((N,N))

if fractal == True:
    if fromMask: 
        #Load the Mask: 
        R = 81
        sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
        # sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R" + str(R) + ".png")))
        a = 17
        data[104-a:105+a, 104-a:105+a] = sampling_mask[104-a:105+a, 104-a:105+a]
        # data[104-a:104+a, 104-a:104+a] = sampling_mask[:2*a, :2*a]
        # data[104-a:104+a, 104-a:104+a] = sampling_mask[104-a:104+a, 104-a:104+a]
        data[data > 100] = 255
        data[data < 100] = 0

    else:
        # data = sampling_mask
        a = 50
        data[104-a:105+a, 104-a:105+a] = 255

    x = torch.tensor(data.copy(), dtype=torch.float)
    data = torch.tensor(data.copy(), dtype=torch.float)
    data = rearrange(data,'h w -> 1 1 h w')

    x = rearrange(x,'h w -> 1 1 h w')
    plt.imsave(f'mask_Rasdfdsf1.png', np.abs(x[0,0,:,:]))
    print(f"the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

    """
    Factors of 207 and 209:

    shifts = [3, 9, 23, 69, 11, 19]

    """
    #Removing unused values
    # shifts = [9, 23, 11, 19, 2, 4, 8, 13, 16, 26, 52, 104]
    # shifts = [3, 9, 23, 69, 11, 19]
    # shifts = [23, 11, 19]
    # shifts = [11]
    shifts = [3, 9, 23, 69, 11, 19, 103, 5, 7]
    # shifts = [3, 9, 23, 11, 19, 103, 5, 7]
    # shifts = range(N//2)
    # shifts = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    # shifts = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
    count = 0
    for shift in shifts:
            # if np.mod(shift,2) != 0:
                # if shift not in [3, 9, 23, 69, 11, 19]:
        # if (np.mod(N,shift) == 0):
        #     pass
        #     # if np.mod(shift,2) != 0:
        #         # if shift not in [3, 9, 23, 69, 11, 19]:
        # else: 
            if shift != 0:
                count += 1
                mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift, N=N)
                x = torch.logical_or(x,Kaleidoscope.ApplyMKTransform(data, mktindexes))
                # plt.imsave(f'mask_R{shift}.png', np.abs(x[0,0,:,:])) # save the image on each shift
                print(f"After: {shift} the percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")



    # print(f"The final percentage is {np.count_nonzero(value)}, {np.count_nonzero(value)/(N*N)}")

    plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:]), cmap='gray')
    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)
else: 

    # Gaussian Mask

    h, w =data.shape

    R = 6

    if R == 8:
        value = 0
    if R == 6:
        value = 8
    if R == 4:
        value = 26

    indexs =[]
    newmasks = np.zeros((N,N))
    while len(np.unique(indexs)) < value:
        temp = random.gauss(104, 52)
        temp = round(temp)

        if (temp > 0) and (temp < 208):
            if (temp > (104-13)) and (temp < (104+13)):
                #make sure it's not going over previous lines
                pass
            else:
                indexs.append(temp)
                newmasks[:,temp] = 255
            
    newmasks[:,104-13:104+13] = 255 # add middle 8th
    # print(np.count_nonzero(newmasks[:,:]))
    plt.imsave(f'mask_R{R}.png', newmasks, cmap = 'gray')
    print(f"The percentage is {np.count_nonzero(newmasks)}, {(np.count_nonzero(newmasks)/(208*208))}")



### Lets test the SSIM:
def undersample(ph):
    # Undersample the image
    y = fft_2d(ph.to('cuda')) * sampling_mask.to('cuda')
    zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :].to('cuda')
    return y, zf_image


# # shifts = [3, 23, 69, 11, 19]
changes23 = Kaleidoscope.MKTkaleidoscopeIndexes(23,N).to('cuda')
changes11 = Kaleidoscope.MKTkaleidoscopeIndexes(11,N).to('cuda')
changes19 = Kaleidoscope.MKTkaleidoscopeIndexes(19,N).to('cuda')


def unFractalise(image):
    image = torch.fft.ifftshift(image)
    out23 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes23).to('cuda')
    out11 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes11).to('cuda')
    # out11 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes11).to('cuda')
    out19 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes19).to('cuda')
    image = ((out23 + out11 + out19)).to('cuda')
    # image = out11
    image = torch.fft.ifftshift(image)
    return image

def Fractalise(image):
    image = torch.fft.ifftshift(image)
    out23 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes23).to('cuda')
    out11 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes11).to('cuda')
    # out11 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes11).to('cuda')
    out19 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes19).to('cuda')
    image = ((out23 + out11 + out19)).to('cuda')
    # image = out11
    image = torch.fft.ifftshift(image)
    return image


norm = 'ortho'
numCh=1

sampling_mask = np.array(ImageOps.grayscale(Image.open("mask_R.png")))
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)

ph = CreatePhantom(N)
# ph2 = unFractalise(ph)
y, zf = undersample(ph)
# y, zf2 = undersample(ph2)
plt.imsave(f'zf1.png', np.abs(zf[0,0,:,:].cpu()))
# plt.imsave(f'zf1.png', np.abs(ph[0,0,:,:].cpu()))

ssim = StructuralSimilarityIndexMeasure().to('cuda')

print(ssim(ph.to('cuda'),zf.to('cuda')))

# ph = unFractalise(ph)
# zf = unFractalise(zf)
# zf2 = Kaleidoscope.ApplyMKTransform(zf2.to('cuda'), changes11).to('cuda')
# zf2 = Fractalise(zf2)

# zf = unFractalise(zf)
# plt.imsave(f'zf2.png', np.abs(zf[0,0,:,:].cpu()))

# print(ssim(ph.to('cuda'),zf.to('cuda')))

# zf2 = Fractalise(zf2)

# # zf = unFractalise(zf)
# plt.imsave(f'zf23.png', np.abs(zf2[0,0,:,:].cpu()))

# print(ssim(ph.to('cuda'),zf2.to('cuda')))