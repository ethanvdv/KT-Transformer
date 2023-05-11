import numpy as np
import matplotlib.pyplot as plt
# from CreatePhantom import *
from Kaleidoscope import *  
from PIL import Image, ImageOps
import random

fromMask = True

N = 224
data = np.zeros((N,N))

if fromMask: 
    #Load the Mask: 
    R = 11
    sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/fractalmasks/mask_R" + str(R) + ".png")))
    a = 5

    data[N//2-a:N//2+1+a, N//2-a:N//2+1+a] = sampling_mask[N//2-a:N//2+1+a, N//2-a:N//2+1+a]
    
    data[data > 100] = 255
    data[data < 100] = 0

x = torch.tensor(data.copy(), dtype=torch.float)
data = torch.tensor(data.copy(), dtype=torch.float)
data = rearrange(data,'h w -> 1 1 h w').to('cuda')

x = rearrange(x,'h w -> 1 1 h w').to('cuda')
plt.imsave(f'mask_Rasdfdsf1.png', np.abs(x[0,0,:,:].cpu()))

nus = [4, 7, 8, 14, 16, 28, 32, 56, -4, -7, -8, -14, -16, -28, -32, -56]
# nus = [20, 16]
# sigmas = [-4, 167]
count = 0
# for shift in shifts:
for nu in nus:
    count += 1
    mktindexes = Kaleidoscope.KalIndexes(nu=nu, sigma=1, N=N).to('cuda')
    x = torch.logical_or(x,Kaleidoscope.pseudoInvMKTransform(data.to('cuda'), mktindexes).to('cuda')).to('cuda')
    print(f"The {nu} percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")
    
plt.imsave(f'mask_R.png', np.abs(x[0,0,:,:].cpu()), cmap='gray')
print(f"The final percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")
print(count)

for nu in nus:
    count += 1
    mktindexes = Kaleidoscope.KalIndexes(nu=nu, sigma=-1, N=N).to('cuda')
    x = torch.logical_or(x,Kaleidoscope.pseudoInvMKTransform(data.to('cuda'), mktindexes).to('cuda')).to('cuda')
    print(f"The {nu} percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")

plt.imsave(f'mask_1R.png', np.abs(x[0,0,:,:].cpu()), cmap='gray')
print(f"The final percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")
print(count)
 
for nu in nus:
    count += 1
    mktindexes = Kaleidoscope.KalIndexes(nu=1, sigma=nu, N=N).to('cuda')
    x = torch.logical_or(x,Kaleidoscope.pseudoInvMKTransform(data.to('cuda'), mktindexes).to('cuda')).to('cuda')
    print(f"The {nu} percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")

plt.imsave(f'mask_2R.png', np.abs(x[0,0,:,:].cpu()), cmap='gray')
print(f"The final percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")
print(count)


for nu in nus:
    count += 1
    mktindexes = Kaleidoscope.KalIndexes(nu=-1, sigma=nu, N=N).to('cuda')
    x = torch.logical_or(x,Kaleidoscope.pseudoInvMKTransform(data.to('cuda'), mktindexes).to('cuda')).to('cuda')
    print(f"The {nu} percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")

plt.imsave(f'mask_3R.png', np.abs(x[0,0,:,:].cpu()), cmap='gray')
print(f"The final percentage is {np.count_nonzero(x[0,0,:,:].cpu())}, {np.count_nonzero(x[0,0,:,:].cpu())/(N*N)}")
print(count)
