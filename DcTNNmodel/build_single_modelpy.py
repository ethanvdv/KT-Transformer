import time
import numpy as np
from DcTNN.tnn import * 
from dc.dc import *
from phantominator import shepp_logan
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
from torchmetrics import StructuralSimilarityIndexMeasure
from DcTNN.KalEncoder import *

norm = 'ortho'
N = 176
R = 9
numCh = 1
lamb = True

# Generate phantom for testing
ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
# As pytorch tensor
ph = torch.tensor(ph.copy(), dtype=torch.float)
# Create channel dim (with dummy data for the channel dimension)
ph = torch.unsqueeze(ph, 0)
ph = torch.cat([ph]*numCh, 0)
# Create batch dim
ph = torch.unsqueeze(ph, 0)

# Load a random sampling mask and inverse fourier shift
sampling_mask = np.array(ImageOps.grayscale(Image.open("masks/mask_R" + str(R) + ".png")))
print(np.sum(sampling_mask))
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)

# Undersample the image
y = fft_2d(ph) * sampling_mask
zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :]

# Define transformer encoder parameters
nhead_patch = 5
layerNo = 1
d_model_patch = None
num_encoder_layers = 2
numCh = numCh
dim_feedforward = None
sigma = 1

# Define the dictionaries of parameter values
kdArgs = {"nu": 25, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

kd2Args = {"nu": 35, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 7, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

kd3Args = {"nu": 7, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 7, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

# Define the array of encoders
encList = [patch2VIT, patch2VIT, patch2VIT]
# Define the array of dictionaries
encArgs = [kdArgs, kd2Args, kd3Args]

# Define the model
dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)

# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in dcenc.parameters() if p.requires_grad)
print("Number of trainable params: " + str(pytorch_total_params))

# Perform model prediction
with torch.no_grad():
    start = time.time()
    phRecon = dcenc(zf_image, y, sampling_mask)
    end = time.time()
    elapsed = end - start
    print("Execution took " + str(elapsed) + " seconds")

# Illustrate operations
# plt.close()
plt.figure(1)
plt.imshow(np.abs(ph[0, 0, :, :]))
plt.title("Phantom")

plt.figure(2)
plt.imshow(np.abs(zf_image[0, 0, :, :]))
plt.title("Zero Fill Image")

plt.figure(3)
plt.imshow(np.abs(phRecon[0, 0, :, :]))
plt.title("Reconstruction")

plt.figure(4)
plt.imshow(sampling_mask)
plt.title("Sampling Mask")

# print(ssim((np.array(phRecon[0, 0, :, :], dtype=np.float32)),(np.array(ph[0, 0, :, :],dtype=np.float32))))
# ssim = StructuralSimilarityIndexMeasure(gaussian_kernel=False,reduction='none')
ssim = StructuralSimilarityIndexMeasure()
print(ssim((phRecon),(ph)))

plt.show()

