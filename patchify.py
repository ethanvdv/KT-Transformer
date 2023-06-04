import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from PIL import Image, ImageOps
import torch.nn as nn
import math
def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph

import math
import torch.nn.functional as F

def do_patchify(img, N, nu, sigma, f, g):
    assert N//2 > abs(nu),  "Cannot Patchify size for |nu| > N/2"
    assert N//2 > abs(sigma),  "Cannot Patchify size for |sigma| > N/2"
    # if sigma > 0:
    b,c,h,w = img.shape

    if (N % sigma) != 0:
        N += sigma
        assert N % nu == 0
        xsize = int(f*N/(nu))
        ysize = int(g*N/(nu))            
        print((w % xsize) ,(h % ysize) )
        if (w % xsize) == 0:
            pad_h = 0
        else:
            pad_h = xsize - (w % xsize)
            
        if (h % ysize) == 0:
            pad_w = 0
        else:
            pad_w = ysize - (h % ysize) 
        print(pad_w, pad_h)
    else:
        xsize = int(f*N/(nu))
        ysize = int(g*N/(nu))   
        print((w % xsize) ,(h % ysize) )
        if (w % xsize) == 0:
            pad_h = 0
        else:
            pad_h = xsize - (w % xsize)
            
        if (h % ysize) == 0:
            pad_w = 0
        else:
            pad_w = ysize - (h % ysize) 

    out = F.pad(input=img, pad=(0, pad_w, 0, pad_h))
    print(out.shape)
    _, _, n1, n2 = out.shape
    out = rearrange(out, 'b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=xsize, p2=ysize)
    return out, n1, xsize, ysize
# if sigma < 0:
    #     b,c,h,w = img.shape

    #     if (N % sigma) != 0:
    #         N += sigma
    #         assert N % nu == 0
    #         xsize = int(f*N/(nu))
    #         ysize = int(g*N/(nu))            
    #         print((w % xsize) ,(h % ysize) )
    #         if (w % xsize) == 0:
    #             pad_h = 0
    #         else:
    #             pad_h = xsize - (w % xsize)
                
    #         if (h % ysize) == 0:
    #             pad_w = 0
    #         else:
    #             pad_w = ysize - (h % ysize) 
    #         print(pad_w, pad_h)
    #     else:
    #         xsize = int(f*N/(nu))
    #         ysize = int(g*N/(nu))   
    #         print((w % xsize) ,(h % ysize) )
    #         if (w % xsize) == 0:
    #             pad_h = 0
    #         else:
    #             pad_h = xsize - (w % xsize)
                
    #         if (h % ysize) == 0:
    #             pad_w = 0
    #         else:
    #             pad_w = ysize - (h % ysize) 
                
    #         print(pad_w, pad_h)

    #     out = F.pad(input=img, pad=(0, pad_w, 0, pad_h))
    #     print(out.shape)
    #     _, _, n1, n2 = out.shape
    #     out = rearrange(out, 'b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=xsize, p2=ysize)
    #     return out, n1, xsize, ysize

      
def undo_patchify(img, N1, xsize, ysize):
    b,c,h,w = img.shape
    valuey = int(N1/h)
    out = rearrange(img,'b (h w) (p1) (p2 c)-> b c (h p1) (w p2)', c=1, h=valuey, p1=xsize, p2=ysize)
    print(out.shape)
    return out


"""
Factors Pairs of 224:

(1, 224)
(2, 112)
(4, 56)
(7, 32)
(8, 28)
(14,16)

"""

N = 225
ph = CreatePhantom(N).to('cpu')

nu = 13
sigma = -4

f = 86
g = 124
# if sigma == 1:
print(f"For a {(nu,sigma,f,g)} Transform") 
print(f"There will be {int((nu**2) /(f * g))} KS Tokens with size:")

print(f"{int(nu/g)} x {int(nu/f)} grid, Tokens are {int(g*N/(nu))} pixels wide - {int(f*N/(nu))} pixels tall")

print(f"There will be {int((f * g * N**2)/(nu**2))} Inverse KS Tokens with be size:")

print(f"{int(N*g/(nu))} x {int(N*f/(nu))} grid, Tokens are {int(nu/g)} pixels wide  - {int(nu/f)} pixels tall")

mktindexes, check1, check2 = Kaleidoscope.KaleidoscopeShuffleIndexes(nu=nu,sigma=sigma, N=N, F=f, G=g)

print(f"{(check1, check2)}")
if check1 == N:
    print("Row Mapping is one-to-one")
else:
    print("Row Mapping is not one-to-one")

if check2 == N:
    print("Column Mapping is one-to-one")
else:
    print("Column Mapping is not one-to-one")

if check1 == check2 == N:
    print("Transform is invertible")


x = Kaleidoscope.ApplyKTTransform(ph, mktindexes)


out, n1, xsize, ysize = do_patchify(x, N, nu, sigma, f, g)
print(out.shape)
inpt = undo_patchify(out, n1, xsize, ysize)
print(inpt.shape)

inverseimage = Kaleidoscope.pseudoInvKTransform(inpt, mktindexes)
print(inverseimage.shape)

plt.figure(0)
plt.imshow(np.abs(ph[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) KS")
# plt.imsave('output3.png',np.abs(inverseimage[0,0,:,:].numpy()))
plt.imsave("step1.png", np.abs(ph[0,0,:,:].numpy()))

plt.figure(1)
plt.imshow(np.abs(x[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) KS")
plt.imsave("step2.png", np.abs(x[0,0,:,:].numpy()))

plt.figure(2)
plt.imshow(np.abs(out[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) KS")
plt.imsave("step3.png", np.abs(out[0,0,:,:].numpy()))

plt.figure(3)
plt.imshow(np.abs(inpt[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) KS")
plt.imsave("step4.png", np.abs(inpt[0,0,:,:].numpy()))


plt.figure(4)
plt.imshow(np.abs(inverseimage[0,0,:,:].numpy()))
plt.title(f"Inverse of ({nu}-{sigma})-({f}-{g}) KS")
plt.imsave('step5.png',np.abs(inverseimage[0,0,:,:].numpy()))

plt.show()


