import time
import numpy as np
from DcTNN.tnn import * 
from dc.dc import *
from DcTNN.FractalEncoder import *
from DcTNN.KTEncoder import *
# from phantominator import shepp_logan
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from loadingimages import GetDatasetFolder
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure
import wandb

norm = 'ortho'
N = 211
R = 211
fractal = True
original = False
numCh = 1
lamb = True

device = 'cuda'
# Define transformer encoder parameters
patchSize = 16
nhead_patch = 8
nhead_axial = 8

d_model_axial = None
d_model_patch = None
num_encoder_layers = 2
numCh = numCh
dim_feedforward = None
# lr = 1e-4
lr = 5e-4
weighting = 10e-7
# weighting = 5e-7
MAE_loss = torch.nn.L1Loss().to('cuda')

firstepoch = 0
epochs = 400
step = 0

BASE_PATH = '/home/groups/deep-compute/OASIS/'
BATCH_SIZE = 15
offset = 1100

torch.cuda.set_device('cuda:0')


"""
Helper Functions
"""
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


if original == True:
    # layerNo = 3
    # patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    # kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    # axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}

    # encList = [axVIT, patchVIT, patchVIT]
    # encArgs = [axArgs, kdArgs, patchArgs]
    layerNo = 1

    kd15Args = {"nu": 15, 'sigma': 1,  "layerNo": layerNo, "numCh": numCh, "nhead": 14, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    kd14Args = {"nu": 14, 'sigma': 1,  "layerNo": layerNo, "numCh": numCh, "nhead": 15, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    kd10Args = {"nu": 10, 'sigma': 1,  "layerNo": layerNo, "numCh": numCh, "nhead": 21, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

    encList = [AntiVIT, AntiVIT, AntiVIT]
    encArgs = [kd15Args, kd14Args, kd10Args]
    

else:

    layerNo = 1

    kd15Args = {"nu": 15, 'sigma': 1,  "layerNo": layerNo, "numCh": numCh, "nhead": 14, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    kd14Args = {"nu": 14, 'sigma': 1,  "layerNo": layerNo, "numCh": numCh, "nhead": 15, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    kd10Args = {"nu": 10, 'sigma': 1,  "layerNo": layerNo, "numCh": numCh, "nhead": 21, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

    encList = [ShuffleVIT, ShuffleVIT, ShuffleVIT]
    encArgs = [kd10Args, kd10Args, kd10Args]
    

if fractal:
    sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/Prime/fractalmasks/mask_R" + str(R) + ".png")))
else:
    sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/Prime/masks/mask_R" + str(R) + ".png")))


sm = rearrange(sampling_mask, 'h w -> 1 1 h w')
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)
sampling_mask.to('cuda')


if original:
# Define the model
    dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)
else:
    dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)
dcenc = dcenc.to(device)
# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in dcenc.parameters() if p.requires_grad)

optimizer = optim.Adam(dcenc.parameters(),lr)

ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)


val_ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=1)

ssim = StructuralSimilarityIndexMeasure().to('cuda')

print(f'Original model?: {original}')
print(f'Fractal {fractal}')
print(f"Sampling pattern used {R}")
print(encList)
print(encArgs)
print("Number of trainable params: " + str(pytorch_total_params))
print(f'Number of Epochs {epochs}')
print(f"Batch Size: {BATCH_SIZE}")
print(f"Value Offset: {offset}")
print(f"Lambda {lamb}")

wandb.init(project="Prime Transformer - 2", config={"Original":original, "Fractal": fractal, "Sampling Pattern": R, "epochs": epochs, "lambda": lamb, "Batch Size": BATCH_SIZE})
# wandb.run.log_code(".")
wandb.config.update({"Value Offset": offset, "Enclist": encList, "encArgs": encArgs})
samplingmask = wandb.Image(np.abs(sm[0,0,:,:]), caption="Sampling Mask")
wandb.log({'mask': samplingmask})


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
        # print(ph.shape)
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
        wandb.log({'trainlos': loss.item()}, commit=False) 
        if loss.item() < finalrunning_loss:
            finalrunning_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        totaliterstrain += 1
        step += 1
    # print(totaliterstrain)
    
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
            if (epoch == 0) and (batch_idx == 0):
                bestSSim = ssim(phRecon1[0:1, :, :, :].to('cuda'),singleimage1[0:1, :, :, :].to('cuda'))
                originalimages = wandb.Image(np.abs(singleimage1[0, 0, :, :].cpu()), caption="Original Image")
                images = wandb.Image(np.abs(phRecon1[0, 0, :, :].cpu()), caption="Image Recon")
                wandb.log({"Original Image": originalimages, "recon": images}, commit=False)
            
            valuessim = ssim(phRecon1[0:1, :, :, :].to('cuda'),singleimage1[0:1, :, :, :].to('cuda'))
            if (epoch != 0):
                if valuessim > bestSSim:
                    bestSSim = valuessim
                    originalimages = wandb.Image(np.abs(singleimage1[0, 0, :, :].cpu()), caption="Original Image")
                    images = wandb.Image(np.abs(phRecon1[0, 0, :, :].cpu()), caption="Image Recon")
                    print(bestSSim)
                    wandb.log({'bestssim': bestSSim}, commit=False) 
                    if epoch > 5:
                        wandb.log({"Original Image": originalimages, "recon": images}, commit=False)
            if (np.mod(epoch, 5) == 0):
                originalimages = wandb.Image(np.abs(singleimage1[0, 0, :, :].cpu()), caption="Original Image")
                images = wandb.Image(np.abs(phRecon1[0, 0, :, :].cpu()), caption="Image Recon")
                wandb.log({"Original Image": originalimages, "recon": images}, commit=False)
            
                
            
            if loss.item() < running_loss2:
                running_loss2 = loss.item() 
                print("_________________")
                print("New Best Iterate: " + str(running_loss2) )
                print("_________________")
            finalrunning_loss2 = loss.item()
            totalitersval +=1
            wandb.log({'valloss': loss.item(), 'ssim': valuessim}, commit=False) 
            
    # print(totaliterstrain)
    if epoch == 0:
        zfimages = wandb.Image(np.abs(zf_image[-1, 0, :, :].cpu()), caption="Zero Fill Image")
        wandb.log({"Zero Fill": zfimages}, commit=False)        
        print(ssim(phRecon1[-2:-1, :, :, :].to('cuda'),singleimage1[-2:-1, :, :, :].to('cuda')))
       
    print("___________________________________")
    print(f"Epoch {epoch} Train loss: {running_loss}")
    print(f"Epoch {epoch} Best Train loss: {finalrunning_loss}")
    print(f"Epoch {epoch} Best Val loss: {running_loss2}")
    print("___________________________________")

    end = time.time()
    elapsed = end - start
    print("Execution took " + str(elapsed) + " seconds")
    wandb.log({'epoch': epoch, "time per epoch": elapsed}, commit=True) 

    totaltime += elapsed
    print("Total Time (Mins): " + str((1 / 60) * totaltime))
    print(f"Estimated Time Left (Mins): {(1 / 60) * (epochs - epoch +1) * (totaltime / (epoch + 1))}")


    print("___________________________________")
    print("___________________________________")
    print("___________________________________")

print(f"Total training iterations {totaliterstrain}")
print(f"Total validation iterations {totalitersval}")
print("Done")

