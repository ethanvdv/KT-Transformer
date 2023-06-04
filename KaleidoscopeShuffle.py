import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from PIL import Image, ImageOps

def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph


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

# -359,28
nu = -225
sigma = -225

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
# plt.figure(1)
# plt.imshow(np.abs(ph[0,0,:,:].numpy()))
# plt.title("Original Image")

plt.figure(2)
plt.imshow(np.abs(x[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) KS")
plt.imsave('output.png',np.abs(x[0,0,:,:].numpy()))

plt.figure(3)
plt.imshow(np.abs(inversetransformimage[0,0,:,:].numpy()))
plt.title(f"({nu}-{sigma})-({f}-{g}) iKS")  
plt.imsave('output2.png',inversetransformimage[0,0,:18,:18].numpy())

plt.figure(4)
plt.imshow(np.abs(inverseimage[0,0,:,:].numpy()))
plt.title(f"Inverse of ({nu}-{sigma})-({f}-{g}) KS")
plt.imsave('output3.png',np.abs(inverseimage[0,0,:,:].numpy()))


plt.figure(5)
plt.imshow(np.abs(invertedinverseimage[0,0,:,:].numpy()))
plt.title(f"Inverse of ({nu}-{sigma})-({f}-{g}) iKS")


plt.figure(6)
plt.imshow(np.abs(mktindexes[0,0,:,:].numpy())/N)
plt.title(f"{nu}-{sigma} Kaleidoscope Transform Indexes")
plt.imsave('index.png',np.abs(mktindexes[0,0,:,:].numpy())/N)

plt.show()


