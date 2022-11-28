import time
import numpy as np
from DcTNN.tnn import * 
from dc.dc import *
from DcTNN.KalEncoder import *
# from phantominator import shepp_logan
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from loadingimages import GetDatasetFolder
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure
import os
print("Updated Encoding")
randomnumber = np.random.randint(1,1000)
print(f"Random number to deal with namespace stuff: {randomnumber}")
print(torch.cuda.is_available)
norm = 'ortho'
N = 176
R = 4
numCh = 1
lamb = True
device = 'cuda'
#OASIS Dataset version

torch.cuda.set_device('cuda:0')

path = os.path.join('/home/Student/s4532907', str(randomnumber))

os.mkdir(path)

# Load a random sampling mask and inverse fourier shift
# sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/masks/mask_R" + str(R) + ".png")))
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
# sampling_mask = sampling_mask // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)
plt.imsave(f'{randomnumber}/samplingmask{randomnumber}.png',np.abs(sampling_mask))
sampling_mask.to('cuda')

def undersample(ph):
    # Undersample the image
    y = fft_2d(ph.to('cuda')) * sampling_mask.to('cuda')
    zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :].to('cuda')
    return y, zf_image

    
def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:].to('cuda')-img[:,:,:-1,:].to('cuda'), 2).sum()
    tv_w = torch.pow(img[:,:,:,1:].to('cuda')-img[:,:,:,:-1].to('cuda'), 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

# Define transformer encoder parameters
patchSize = 16
nhead_patch = 8
nhead_axial = 8
layerNo = 1
d_model_axial = None
d_model_patch = None
num_encoder_layers = 2
numCh = numCh
dim_feedforward = None


# # Define the dictionaries of parameter values
# patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
# kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
# axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}

kd3Args = {"nu": 5, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
kd2Args = {"nu": 3, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 3, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
kd1Args = {"nu": 7, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 7, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
kd4Args = {"nu": 25, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
# kd1Args = {"nu": 3, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 3, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
# kd2Args = {"nu": 5, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

# # Define the array of encoders
# encList = [axVIT, patchVIT, patchVIT]
# # Define the array of dictionaries
# encArgs = [axArgs, kdArgs, patchArgs]




encList = [patch2VIT, patch2VIT, patch2VIT, patch2VIT]
encArgs = [kd1Args, kd3Args,kd2Args, kd4Args]
# # 7 25 5

# Define the model
dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)
dcenc = dcenc.to(device)
# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in dcenc.parameters() if p.requires_grad)
print("Number of trainable params: " + str(pytorch_total_params))


lr = 3e-4
weighting = 10e-7
print(f'lr {lr} weighting {weighting}')
MAE_loss = torch.nn.L1Loss().to('cuda')

optimizer = optim.Adam(dcenc.parameters(),lr)
epochs = 100
step = 0

# BASE_PATH = '/home/groups/deep-compute/OASIS/'
BASE_PATH = '/home/Student/s4532907/'
BATCH_SIZE = 10
# batching = 15
offset = 1700

ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)


val_ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=1)

print(f"Sampling pattern used {R}")
print(encArgs)
print("Number of trainable params: " + str(pytorch_total_params))
print(f'Number of Epochs {epochs}')
print(f"Batch Size: {BATCH_SIZE}")
print(f"Value Offset: {offset}")


finalrunning_loss = 1
running_loss2 = 1
totaliterstrain = 0
totalitersval = 0
totaltime = 0
for epoch in range(epochs):
    start = time.time()
    # Train
    print(f"Epoch: {epoch}/{epochs}")
    running_loss = 0
    finalrunning_loss = 1
    for batch_idx, ph in enumerate(dl):
        ph = rearrange(ph, 'b h w -> b 1 h w').contiguous()
        ph.to(device)
        singleimage = ph[:, :, :, :].to('cuda')
        singleimage.to('cuda')
        y, zf_image = undersample(singleimage)
        y.to(device), zf_image.to(device)
        optimizer.zero_grad()
        phRecon = dcenc(zf_image, y, sampling_mask)
        phRecon.to('cuda')
        loss = (MAE_loss(phRecon.to('cuda'), singleimage.to('cuda')) + total_variation_loss(phRecon.to('cuda'),weighting))
        # loss = MAE_loss(phRecon.to('cuda'), singleimage.to('cuda'))
        running_loss += loss.item()
        if loss.item() < finalrunning_loss:
            finalrunning_loss = loss.item()
            # print(f'New Best: {finalrunning_loss}')
        loss.backward()
        optimizer.step()
        totaliterstrain += 1
        step += 1
    
    
    #Attempt at Testing Dataset
    for batch_idx, X in enumerate(val_dl):
        X = rearrange(X, 'b h w -> b 1 h w').contiguous()
        X = X.to(device)
        singleimage1 = X[:, :, :, :].to('cuda')
        singleimage1.to(device)
        y, zf_image = undersample(singleimage1)
        y.to(device), zf_image.to(device) 
        with torch.no_grad():
            phRecon1 = dcenc(zf_image, y, sampling_mask)
            phRecon1.to('cuda')
            loss = (MAE_loss(phRecon1.to('cuda'), singleimage1.to('cuda'))+ total_variation_loss(phRecon1,weighting))
            # loss = MAE_loss(phRecon1.to('cuda'), singleimage1.to('cuda'))
            loss.to('cuda') 
            if loss.item() < running_loss2:
                running_loss2 = loss.item() 
                print("_________________")
                print("New Best Iterate: " + str(running_loss2) )
                print("_________________")
            finalrunning_loss2 = loss.item()
            totalitersval +=1
    

    #Print a final reconstruction
    if (np.mod(epoch,25) == 0) or (epoch == (epochs-1)):
        plt.imsave(f'{randomnumber}/newensemble-{randomnumber}-{epoch}-recon.png',np.abs(phRecon1[-1, 0, :, :].cpu()))
        if epoch == 0:
            plt.imsave(f'{randomnumber}/newensemble-{randomnumber}-{epoch}-image.png',np.abs(singleimage1[-1, 0, :, :].cpu()))
            plt.imsave(f'{randomnumber}/newensemble-{randomnumber}-{epoch}-zf.png',np.abs(zf_image[-1, 0, :, :].cpu()))
        # plt.imsave(f'newensemble-{randomnumber}-{epoch}-zfimage.png',np.abs(zf_image[0, 0, :, :].cpu()))
        ssim = StructuralSimilarityIndexMeasure().to('cuda')
        print(f'SSIM at epoch {epoch}:')
        print(ssim(phRecon1[-2:-1, :, :, :].to('cuda'),singleimage1[-2:-1, :, :, :].to('cuda')))
        
    print("___________________________________")
    print(f"Epoch {epoch} Train loss: {running_loss}")
    print(f"Epoch {epoch} Best Train loss: {finalrunning_loss}")
    print(f"Epoch {epoch} Best Val loss: {running_loss2}")
    print("___________________________________")

    end = time.time()
    elapsed = end - start
    print("Execution took " + str(elapsed) + " seconds")

    totaltime += elapsed
    print("Total Time (Mins): " + str((1 / 60) * totaltime))
    print(f"Estimated Time Left (Mins): {(1 / 60) * (epochs - epoch +1) * (totaltime / (epoch + 1))}")


    print("___________________________________")
    print("___________________________________")
    print("___________________________________")

print(f"Total training iterations {totaliterstrain}")
print(f"Total validation iterations {totalitersval}")
print("Done")



