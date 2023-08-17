import numpy as np
from skimage import data
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from Kaleidoscope import Kaleidoscope
import torch
from einops import rearrange
from phantominator import shepp_logan
import math
from functools import reduce
from skimage.metrics import structural_similarity as ssim
def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


'''
Attempt at Reducing Gaussian Noise through a Kaleidoscope Fractal.
Denoising code inspired by code from 3710
'''


def getCircle(image, radius):
    r, c = image.shape
    print(image.shape)
    N = r
    print("radius:", radius)
    centerX = r/2 
    centerY = c/2 
    print(centerX)
    print(centerY)
    #create pass filters
    lowPassFilter = np.zeros((r,c))
    highPassFilter = np.zeros((r,c))
    for indexX, row in enumerate(image):
        for indexY, column in enumerate(image):
            xValue = int(indexX-centerX) #center the index
            yValue = int(indexY-centerY) #center the index
            if ((xValue)**2 + yValue**2) <= radius**2: #within certain distance
                lowPassFilter[indexX, indexY] = 1.0 #keep low freqs
            # else:
            elif ((xValue)**2 + yValue**2) <= (radius*10)**2: #within certain distance
                highPassFilter[indexX, indexY] = 1.0 #else keep high freqs

    return lowPassFilter, highPassFilter


def fractalise(data, remove, hpfilter, affine):
    r, c = data.shape
    N = r
    x = torch.tensor(data.copy(), dtype=torch.float)
    hpfilter = torch.tensor(hpfilter.copy(), dtype=torch.float)
    # x2 = torch.tensor(data.copy(), dtype=torch.float)
    data = torch.tensor(data.copy(), dtype=torch.float)
    data = rearrange(data,'h w -> 1 1 h w')

    x = rearrange(x,'h w -> 1 1 h w')
    hpfilter = rearrange(hpfilter,'h w -> 1 1 h w')
    print(f"The percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")

    # shifts = np.arange(r//2)
    # shifts = [3, 107, 11, 29,  2, 3, 6, 53, 106, 159, 7, 14,23, 46]

    limit = 10
    factorslist = set([])
    for k in range(-limit, limit+1):
        iter = N + k
        if iter != N:
            # print(f"Value {iter} ({i})")
            factorslist = factorslist | factors(iter)
    # print(sorted(factors(iter)))
        
    # print(sorted(factorslist))
    shifts = list(sorted(factorslist))
    shifts = [value for value in shifts if value < N//2]


    count = 0
    for shift in shifts:
        # for g in shifts2:
            count += 1
            # mktindexes = Kaleidoscope.KalIndexes(N=N,nu=-1, sigma=shift)
            # mktindexes = Kaleidoscope.AffineKSIndexes(N=N,)
            mktindexes = Kaleidoscope.AffineKSIndexes(N,nu=1,sigma=shift,F=1,G=1,i=0,j=0)
            x = torch.logical_or(x,Kaleidoscope.ApplyKTTransform(data, mktindexes))
            # x = torch.logical_or(x,Kaleidoscope.pseudoInvKTransform(data, mktindexes))

    if remove: 
        x = torch.logical_xor(x,data)
    print(f"The final percentage is {np.count_nonzero(x[0,0,:,:])}, {np.count_nonzero(x[0,0,:,:])/(N*N)}")
    print(count)
    
    hpfilterout = torch.logical_xor(x,hpfilter)
    
    return x[0,0,:,:].numpy(), hpfilterout[0,0,:,:].numpy()

#compute quantitiative measure of denoising effectiveness
def immse(img1, img2):
    '''
    Compute the MSE of two images
    '''
    mse = ((img1 - img2) ** 2).mean(axis=None)

    return mse

def impsnr(img1, img2, maxPixel=255):
    '''
    Compute the MSE of two images
    '''
    mse = immse(img1,img2)
    psnr_out = 20 * math.log(maxPixel / math.sqrt(mse), 10)

    return psnr_out


# image = data.camera()
N = 512

image = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
maxValue = image.max()
mean = 0
var = maxValue/2.0
sigma = var**0.5
noise = np.random.normal(mean,sigma,image.shape)
# print(noise)

# imageNoisy = image
imageNoisy = image + noise


radius = 10
remove = False

lpfilter, hpfilter = getCircle(image, radius)

lowPassFilter, hpfrac = fractalise(lpfilter, remove, hpfilter, affine=-1)

# hpfrac = [lowPassFilter[x, y] for x, y in hpfilter if hpfilter[x,y] == 1]


# plt.gray()
plt.figure(3)
plt.imshow(np.abs(lowPassFilter))
plt.title(f"Output")


# plt.gray()
# plt.figure(1)
# plt.imshow(hpfrac)
# plt.title(f"Output")


#compute the FFT
fftImage = fftpack.fft2(imageNoisy) #2D FFT
fftImage = fftpack.fftshift(fftImage) #shift to the center of space for viewing
powerSpect = np.abs(fftImage) #compute absolute magnitude





#apply low pass
# lowFFTImage = fftImage * fftpack.fftshift(lowPassFilter)
lowFFTImage = fftImage * lowPassFilter
# lowFFTImage = fftImage * hpfrac
powerSpectLow = np.abs(lowFFTImage) #compute absolute magnitude



#inverse FFT to reconstruction image from filtered Fourier space
#compute the iFFT of original unfiltered image
ifftImage = fftpack.ifft2(fftImage) #2D FFT
ifftImage = fftpack.ifftshift(ifftImage) #shift to the center of space for viewing
#compute the iFFT of low pass filtering
lowFFTImage = fftpack.fftshift(lowFFTImage) #shift to the center of space for viewing
ifftLowImage = fftpack.ifft2(lowFFTImage) #2D FFT






#plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))

# plt.gray()
# plt.tight_layout()


maxValue = image.max()
print(ifftLowImage.max())
noisePSNR = impsnr(image, np.real(imageNoisy), maxPixel=maxValue)
deNoisedPSNR = impsnr(image, np.real(ifftLowImage), maxPixel=maxValue)

im1ssim = ssim(image, np.real(imageNoisy))
im2ssim = ssim(image, np.real(ifftLowImage))
print("Noisy PSNR", noisePSNR)
print("deNoised PSNR",deNoisedPSNR )


# ax[0].imshow(np.log10(powerSpectLow))
# ax[0].axis('off')
# ax[0].set_title('Fourer Space')
ax[0].imshow(np.real(ifftLowImage))
ax[0].axis('off')
ax[0].set_title(f'PSNR = {round(deNoisedPSNR,2)}, SSIM = {round(im2ssim,2)}')
ax[1].imshow(np.real(imageNoisy))
ax[1].axis('off')
ax[1].set_title(f'PSNR = {noisePSNR}, SSIM = {round(im1ssim,2)}')




plt.show()