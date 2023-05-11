import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from torchmetrics import PeakSignalNoiseRatio

def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    plt.imsave("originalimage2.png", ph, cmap='inferno', vmin=0, vmax=np.max(ph))
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph
 

torch.cuda.set_device('cuda:0')
psnr = PeakSignalNoiseRatio().to('cuda')

N = 224
forwardpsnrarray = np.zeros([2*N,2*N])
backwardpsnrarray = np.zeros([2*N,2*N])

good = np.zeros([2*N,2*N])
good2 = np.zeros([2*N,2*N])
good3 = np.zeros([2*N,2*N])

np.random.seed(1102)
# ph = CreatePhantom(N).to('cuda')
ph = np.random.randint(0, N, (N, N))

ph = torch.tensor(ph.copy(), dtype=torch.float)

# ph = repeat(ph,'h -> h c', c = N)

ph = rearrange(ph, 'h w -> 1 1 h w').to('cuda')

plt.imsave("originalimage1.png", np.abs(ph[0,0,:,:].cpu()))

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

       
        psnrval = psnr(ph2,ph)
        
        if torch.isinf(psnrval):
            print("forward zeero ----------------")
            good[nu+N,sigma+N] = 255
            forwardpsnrarray[nu+N,sigma+N] = 0
        else:
            forwardpsnrarray[nu+N,sigma+N] = psnrval
        
            
            
        #backward transform error
        x = Kaleidoscope.pseudoInvMKTransform(ph, mktindexes)
        ph2 = Kaleidoscope.ApplyKTTransform(x, mktindexes)
            
        psnrval2 = psnr(ph2,ph)
        
        if torch.isinf(psnrval2):
            print("backward zeero ----------------")
            good2[nu+N,sigma+N] = 255
            backwardpsnrarray[nu+N,sigma+N] = 0
        else:
            backwardpsnrarray[nu+N,sigma+N] = psnrval2
        

        if (torch.isinf(psnrval2)) and (torch.isinf(psnrval)):
            good3[nu+N,sigma+N] = 255


np.save('forward-psnr.npy', forwardpsnrarray)
    
plt.imsave("psnr-forward-heatmap.png", forwardpsnrarray, cmap='inferno', vmin=0, vmax=np.max(forwardpsnrarray))
plt.imsave("psnr-forward-good.png", good, cmap='inferno', vmin=0, vmax=np.max(good))

plt.imsave("psnr-backwawrd-heatmap.png", backwardpsnrarray, cmap='inferno', vmin=0, vmax=np.max(backwardpsnrarray))
plt.imsave("psnr-backward-good.png", good2, cmap='inferno', vmin=0, vmax=np.max(good2))

np.save('backward-psnr.npy', backwardpsnrarray)

plt.imsave("good-psnr.png", good3, cmap='inferno', vmin=0, vmax=np.max(good3))

print("Done")

