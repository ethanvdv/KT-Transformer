import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps

SIZE_INCREASE = 10

def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)

    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = repeat(ph, f'h w-> ({SIZE_INCREASE} h) ({SIZE_INCREASE} w)')
    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph

def CreatePhantom2(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph
  

N = 72
ph = CreatePhantom(N).to('cpu')

N = 72*SIZE_INCREASE
nu = SIZE_INCREASE
sigma = 1
f = 1
g = 1

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
inverseimage = Kaleidoscope.pseudoInvKTransform(x, mktindexes)

inversetransformimage = Kaleidoscope.pseudoInvKTransform(ph, mktindexes)
invertedinverseimage = Kaleidoscope.ApplyKTTransform(inversetransformimage, mktindexes)


plt.close()
plt.figure(1)
plt.imshow(np.abs(ph[0,0,:,:].numpy()))
plt.title("Original Image")

# plt.figure(2)
# plt.imshow(np.abs(x[0,0,:,:].numpy()))
# plt.title(f"({nu}-{sigma})-({f}-{g}) KS")


plt.figure(3)
plt.imshow(np.abs(inversetransformimage[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) iKS")  

ph = CreatePhantom2(N).to('cpu')
plt.figure(4)
plt.imshow(np.abs(ph[0,0,:,:].numpy()))
plt.title("Original Image - Expected Image")

ph = CreatePhantom2(72).to('cpu')
plt.figure(5)
plt.imshow(np.abs(ph[0,0,:,:].numpy()))
plt.title("Original Image - Small Image")

plt.show()


