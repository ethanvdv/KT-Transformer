import time
import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from loadingim import GetDatasetFolder
import torch
from einops import rearrange, repeat
import csv


import numpy as np

def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    plt.imsave("originalimage2.png", ph, cmap='inferno', vmin=0, vmax=np.max(ph))
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph

device = 'cuda'


torch.cuda.set_device('cuda:0')

N = 224
totaltime = 0
start = time.time()
aarray = np.arange(4*N*N)
aarrayes = np.reshape(aarray, [2*N,2*N])
aarrayes2 = np.reshape(aarray, [2*N,2*N])
aarrayes22 = np.reshape(aarray, [2*N,2*N])
good = np.zeros([2*N,2*N])
good2 = np.zeros([2*N,2*N])
good3 = np.zeros([2*N,2*N])
ph = CreatePhantom(N).to('cuda')
# ph = np.random.randint(0, N, N)

# ph = torch.tensor(ph.copy(), dtype=torch.float)

# ph = repeat(ph,'h -> h c', c = N)

# ph = rearrange(ph, 'h w -> 1 1 h w').to('cuda')

# plt.imsave("originalimage.png", np.abs(ph[0,0,:,:].cpu()))

# ph = np.array(ImageOps.grayscale(Image.open("image.png")))
# ph = torch.tensor(ph.copy(), dtype=torch.float)
# ph = rearrange(ph, 'h w -> 1 1 h w').to('cuda')
nus = np.arange(-N,N)
nus = nus[nus != 0]
sigmas = nus[nus != 0]
answers = []
for nu in nus:
    for sigma in sigmas:
        print(f"Sigma - nu, {(sigma,nu)}")
        mktindexes = Kaleidoscope.KalIndexes(nu=nu,sigma=sigma, N=N)

        x = Kaleidoscope.ApplyKTTransform(ph, mktindexes)
        ph2 = Kaleidoscope.pseudoInvMKTransform(x, mktindexes)

        hhh = torch.sum(torch.nonzero(ph-ph2))
        aarrayes2[nu+N,sigma+N] = hhh
        if hhh == 0:
            good[nu+N,sigma+N] = 255
            # row = []
            # row.append(nu)
            # row.append(sigma)
            # answers.append(row)
            
            
        x = Kaleidoscope.pseudoInvMKTransform(ph, mktindexes)
        ph2 = Kaleidoscope.ApplyKTTransform(x, mktindexes)
        hhh1 = torch.sum(torch.nonzero(ph-ph2))
        aarrayes22[nu+N,sigma+N] = hhh1
        if hhh1 == 0:
            good2[nu+N,sigma+N] = 255
            
        if (hhh == 0) and (hhh1 == 0):
            good3[nu+N,sigma+N] = 255
            x = Kaleidoscope.ApplyKTTransform(ph, mktindexes)
            plt.imsave(f"Transforms/{(nu,sigma)}.png", np.abs(x[0,0,:,:].cpu()))
            x = Kaleidoscope.pseudoInvMKTransform(ph, mktindexes)
            plt.imsave(f"Transforms/{(nu,sigma)}-p.png", np.abs(x[0,0,:,:].cpu()))
            # plt.imsave("good.png", good, cmap='inferno', vmin=0, vmax=np.max(good))
            row = []
            row.append(nu)
            row.append(sigma)
            answers.append(row)



with open('transforms.csv', 'w', encoding="UTF-8", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(answers)
    
plt.imsave("heatmap.png", aarrayes, cmap='inferno', vmin=0, vmax=np.max(aarrayes))
plt.imsave("good.png", good, cmap='inferno', vmin=0, vmax=np.max(good))

plt.imsave("heatmap-full2.png", aarrayes2, cmap='inferno', vmin=0, vmax=np.max(aarrayes2))
plt.imsave("good2.png", good2, cmap='inferno', vmin=0, vmax=np.max(good2))

plt.imsave("good3.png", good3, cmap='inferno', vmin=0, vmax=np.max(good3))

# print(answers)
print("Done")

