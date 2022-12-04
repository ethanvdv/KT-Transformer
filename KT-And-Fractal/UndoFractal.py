import torch
from einops import rearrange
from PIL import Image, ImageOps
import numpy as np
from CreatePhantom import * 
from Kaleidoscope import *

def fft_2d(input, norm='ortho', dim=(-2, -1)):
    x = input
    x = rearrange(x, 'b c h w -> b h w c').contiguous()
    if x.shape[3] == 1:
        x = torch.cat([x, torch.zeros_like(x)], 3)
    x = torch.view_as_complex(x)
    x = torch.fft.fft2(x, norm=norm, dim=dim)
    x = torch.view_as_real(x)
    x = rearrange(x, 'b h w c -> b c h w').contiguous()
    return x

def ifft_2d(input, norm='ortho', dim=(-2, -1)):
    x = input
    x = rearrange(x, 'b c h w -> b h w c').contiguous()
    x = torch.view_as_complex(x)
    x = torch.fft.ifft2(x, dim=dim, norm=norm)
    x = torch.view_as_real(x)
    x = rearrange(x, 'b h w c -> b c h w').contiguous()
    return x

def undersample(ph):
    # Undersample the image
    y = fft_2d(ph) * sampling_mask
    zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :]
    return y, zf_image

R = 59
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
# sampling_mask = sampling_mask // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)
# plt.imsave(f'sm.png',np.abs(sampling_mask[:, :]))


ph = np.array(ImageOps.grayscale(Image.open("KT-Transformer/KT-And-Fractal/image.png")))
ph = torch.tensor(ph.copy(), dtype=torch.float)
ph = rearrange(ph, 'h w -> 1 1 h w')
norm = 'ortho'
numCh = 1
N = 176

# # ph = CreatePhantom(N)

# y2, z2 = undersample(ph)

# plt.imsave(f'y.png',np.abs(y2[0, 0, :, :]))
# plt.imsave(f'z.png',np.abs(z2[0, 0, :, :]))

# # y = y.numpy()
# # y = np.fft.ifftshift(y) // np.max(np.abs(y))
# # y = torch.tensor(y.copy(), dtype=torch.float)

changes5 = Kaleidoscope.MKTkaleidoscopeIndexes(5,N)
changes3 = Kaleidoscope.MKTkaleidoscopeIndexes(3,N)
changes59 = Kaleidoscope.MKTkaleidoscopeIndexes(59,N)
changes7 = Kaleidoscope.MKTkaleidoscopeIndexes(37,N)

out5 = Kaleidoscope.ApplyMKTransform(ph, changes3)
print(out5.shape)

# out3 = Kaleidoscope.pseudoInvMKTransform(z2, changes3)
# out59 = Kaleidoscope.pseudoInvMKTransform(z2, changes59)
# out7 = Kaleidoscope.pseudoInvMKTransform(z2, changes7)

plt.imsave(f'5.png',np.abs(out5[0, 0, :, :]))
# plt.imsave(f'3.png',np.abs(out3[0, 0, :, :]))
# plt.imsave(f'59.png',np.abs(out59[0, 0, :, :]))
# plt.imsave(f'7.png',np.abs(out7[0, 0, :, :]))

# output = (out5 + out3 + out59 + out7)
# plt.imsave(f'out.png',np.fft.ifftshift(np.abs(output[0, 0, :, :])))