import torch
from einops import rearrange
from PIL import Image, ImageOps
import numpy as np

from Kaleidoscope import *
from torchmetrics import StructuralSimilarityIndexMeasure
import math


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

"""
This is used to test the pixel count of a sampling mask:
"""
N = 208
R = 81
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
print(sampling_mask.shape)
print(f"The percentage is {np.count_nonzero(sampling_mask)}, {np.count_nonzero(sampling_mask)/(N*N)}")
print(f"The percentage is {np.count_nonzero(sampling_mask)}, {1/(np.count_nonzero(sampling_mask)/(N*N))}")


reconarray = np.array(ImageOps.grayscale(Image.open("recon.png")))
recon = torch.tensor(reconarray.copy(), dtype=torch.float)
recon = rearrange(recon, 'h w -> 1 1 h w')

originalarray = np.array(ImageOps.grayscale(Image.open("original.png")))
original = torch.tensor(originalarray.copy(), dtype=torch.float)
original = rearrange(original, 'h w -> 1 1 h w')

# Test SSIM
ssim = StructuralSimilarityIndexMeasure()
estSSim = ssim(recon,original)
print(estSSim)

#Test PSNR
print(psnr(originalarray, reconarray))

#Lets see the RGB copy of the image: 
plt.imsave(f'orig.png', np.abs(original[0,0,:,:]))
plt.imsave(f'reco.png', np.abs(recon[0,0,:,:]))