import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image, ImageOps
# def CreatePhantom(N):
    
#     #From Marlon's Code for loading Shepp_Logan
#     ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
#     plt.imsave("originalimage2.png", ph, cmap='inferno', vmin=0, vmax=np.max(ph))
#     ph = torch.tensor(ph.copy(), dtype=torch.float)

#     ph = rearrange(ph, 'h w -> 1 1 h w')

#     return ph
 

torch.cuda.set_device('cuda:0')
psnr = PeakSignalNoiseRatio().to('cuda')

N = 223
forwardarray = np.zeros([2*N,2*N])
backwardarray = np.zeros([2*N,2*N])

good = np.zeros([2*N,2*N])
good2 = np.zeros([2*N,2*N])
good3 = np.zeros([2*N,2*N])

# ph = CreatePhantom(N).to('cuda')
np.random.seed(1102)
ph = np.random.randint(0, N, (N, N))
R = 5
inputph = np.array(ImageOps.grayscale(Image.open("NC/2 (" + str(R) + ").jpeg")))
ph = inputph[8:-9,16:-17]

print(ph.shape)
ph = torch.tensor(ph.copy(), dtype=torch.float)

# ph = repeat(ph,'h -> h c', c = N)

ph = rearrange(ph, 'h w -> 1 1 h w').to('cuda')

plt.imsave("originalimage.png", np.abs(ph[0,0,:,:].cpu()))

nus = np.arange(-N,N)
nus = nus[nus != 0]
sigmas = nus[nus != 0]


for nu in nus:
    for sigma in sigmas:
        print(f"Sigma - nu, {(sigma,nu)}")
        mktindexes = Kaleidoscope.KalIndexes(nu=nu,sigma=sigma, N=N)

        #find forward error
        x = Kaleidoscope.ApplyKTTransform(ph, mktindexes)
        ph2 = Kaleidoscope.pseudoInvMKTransform(x, mktindexes)

        mse = torch.sum(torch.nonzero(ph-ph2))
        forwardarray[nu+N,sigma+N] = mse
        
        if mse == 0:
            print("forward zeero ----------------")
            good[nu+N,sigma+N] = 255
            
            
        #backward transform error
        x = Kaleidoscope.pseudoInvMKTransform(ph, mktindexes)
        ph2 = Kaleidoscope.ApplyKTTransform(x, mktindexes)
        mse1 = torch.sum(torch.nonzero(ph-ph2))
        
        backwardarray[nu+N,sigma+N] = mse1
        
        if mse1 == 0:
            print("backward zero ----------------")
            good2[nu+N,sigma+N] = 255

        if (mse1 == 0) and (mse == 0):
            good3[nu+N,sigma+N] = 255


np.save('forward.npy', forwardarray)
    
plt.imsave("forward-heatmap.png", forwardarray, cmap='inferno', vmin=0, vmax=np.max(forwardarray))
plt.imsave("forward-good.png", good, cmap='inferno', vmin=0, vmax=np.max(good))

plt.imsave("backwawrd-heatmap.png", backwardarray, cmap='inferno', vmin=0, vmax=np.max(backwardarray))
plt.imsave("backward-good.png", good2, cmap='inferno', vmin=0, vmax=np.max(good2))

np.save('backward.npy', backwardarray)

plt.imsave("good.png", good3, cmap='inferno', vmin=0, vmax=np.max(good3))

print("Done")

