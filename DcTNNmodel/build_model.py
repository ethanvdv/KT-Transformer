import time
import numpy as np
from DcTNN.tnn import * 
from dc.dc import *
from DcTNN.FractalEncoder import *
# from phantominator import shepp_logan
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from loadingimages import GetDatasetFolder
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure
# import os
import wandb

# print("With Lambda")
# randomnumber = np.random.randint(1,1000)
# print(f"Random number to deal with namespace stuff: {randomnumber}")
# print(torch.cuda.is_available)
norm = 'ortho'
N = 208
R = 6
fractal = True
original = False
numCh = 1
lamb = True

device = 'cuda'
# Define transformer encoder parameters
patchSize = 16
nhead_patch = 8
nhead_axial = 8
layerNo = 3
d_model_axial = None
d_model_patch = None
num_encoder_layers = 2
numCh = numCh
dim_feedforward = None
lr = 3e-4
weighting = 20e-7
# weighting = 5e-7
MAE_loss = torch.nn.L1Loss().to('cuda')

firstepoch = 0
epochs = 400
step = 0

BASE_PATH = '/home/groups/deep-compute/OASIS/'
# /home/groups/deep-compute/OASIS/slices_pad
# BASE_PATH = '/home/Student/s4532907/'
# BATCH_SIZE = 30
BATCH_SIZE = 15
offset = 1100
# offset = 2200



torch.cuda.set_device('cuda:0')

# path = os.path.join('/home/Student/s4532907', str(randomnumber))

# os.mkdir(path)

"""
Helper Functions
"""
def undersample(ph):
    # Undersample the image
    y = fft_2d(ph.to('cuda')) * sampling_mask.to('cuda')
    
    # y = rearrange(sampling_mask, 'h w -> 1 1 h w')
    if original == True:
        y = y
        zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :].to('cuda')
        
    else:
        # y = unFractalise(y)
        zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :].to('cuda')
        zf_image = unFractalise(zf_image)
        # y = Fractalise(y)
 
    return y, zf_image

    
def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:].to('cuda')-img[:,:,:-1,:].to('cuda'), 2).sum()
    tv_w = torch.pow(img[:,:,:,1:].to('cuda')-img[:,:,:,:-1].to('cuda'), 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def FRAC_DC(x, y, mask, lamb, norm='ortho'):
    # x = unFractalise(x)
    # y = unFractalise(y)
    # Check if complex
    numCh = x.shape[1]

    # Get complex y view
    cy = y.permute(0, 2, 3, 1).contiguous()
    cy = torch.view_as_complex(cy)

    # By default, torch.view_as_complex uses last index as real,imag
    x = x.permute(0, 2, 3, 1).contiguous()

    # Perform operations depending on the number of dimensions
    if numCh == 1:

        # Populate imaginary axis with zeros
        x = torch.cat([x, torch.zeros_like(x)], 3)

    # get k-space of the input image
    z = torch.fft.fft2(torch.view_as_complex(x), norm=norm).to('cuda')

    y.to('cuda')
    x.to('cuda')
    mask.to('cuda')
    cy.to('cuda')

    # Perform data consistency 
    if lamb is None:
        # Replace Fourier measurements
        z = ((1 - mask.to('cuda')).to('cuda') * z.to('cuda')) + (mask.to('cuda') * cy.to('cuda'))
    else:
        # Weighted average of the collected and reconstructed points
        z = (((1 - mask).to('cuda')).to('cuda') * z.to('cuda')).to('cuda') + mask.to('cuda') * (z.to('cuda') + lamb.to('cuda') * cy.to('cuda')).to('cuda') / (1 + lamb)

    z.to('cuda')
    # Apply mask and invert (keep channels that we are working with)
    z = torch.view_as_real(torch.fft.ifft2(z, norm=norm))[:, :, :, 0:numCh].to('cuda')
    z = z.permute(0, 3, 1, 2)
    
    

    # Return masked image
    return z.to('cuda')


# shifts = [3, 23, 69, 11, 19]
changes3 = Kaleidoscope.MKTkaleidoscopeIndexes(3,N).to('cuda')
# changes9 = Kaleidoscope.MKTkaleidoscopeIndexes(9,N).to('cuda')
changes23 = Kaleidoscope.MKTkaleidoscopeIndexes(23,N).to('cuda')
changes69 = Kaleidoscope.MKTkaleidoscopeIndexes(69,N).to('cuda')
changes11 = Kaleidoscope.MKTkaleidoscopeIndexes(11,N).to('cuda')
changes19 = Kaleidoscope.MKTkaleidoscopeIndexes(19,N).to('cuda')


def unFractalise(image):
    # zf_image = rearrange(zf_image, 'h w -> 1 1 h w')
    image = torch.fft.ifftshift(image)
    out3 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes3).to('cuda')
    # out9 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes9).to('cuda')
    out23 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes23).to('cuda')
    out69 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes69).to('cuda')
    out11 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes11).to('cuda')
    out19 = Kaleidoscope.pseudoInvMKTransform(image.clone().to('cuda'), changes19).to('cuda')
    image = ((out3 + out23 + out69 + out11 + out19)/5).to('cuda')
    # image = (image + (out3 + out23 + out69 + out11 + out19)/5).to('cuda')
    # image = out3.to('cuda')
    # image = torch.fft.ifftshift(image) / torch.max(torch.abs(image))
    image = torch.fft.ifftshift(image)
    return image


def Fractalise(image):
    image = torch.fft.ifftshift(image)
    out3 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes3).to('cuda')
    # out9 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes9).to('cuda')
    out23 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes23).to('cuda')
    out69 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes69).to('cuda')
    out11 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes11).to('cuda')
    out19 = Kaleidoscope.ApplyMKTransform(image.clone().to('cuda'), changes19).to('cuda')
    image = ((out3 + out23 + out69 + out11 + out19)/5).to('cuda')
    # image = (image - (out3 + out23 + out69 + out11 + out19)/5).to('cuda')
    # image = ((out3 + out23 + out69 + out11 + out19 + out9)).to('cuda')
    # image = ((image + image) - (out3 + out23 + out69 + out11 + out19 + out9)).to('cuda')
    # image = out3.to('cuda')
    image = torch.fft.ifftshift(image)
    return image


if original == True:
#     #Marlons Network
    patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}

    encList = [axVIT, patchVIT, patchVIT]
    encArgs = [axArgs, kdArgs, patchArgs]
else:
    # patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    # kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    # axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}
    # encList = [axVIT]
    # encArgs = [axArgs]
    # encList = [axVIT, patchVIT, patchVIT]
    # encArgs = [axArgs, kdArgs, patchArgs]
    # kd1Args = {"nu": 9, "sigma": 1, "layerNo": layerNo, "numCh": numCh, "nhead": 23, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    # encList = [patchFracVIT]
    # encArgs = [kd1Args]
    patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
    axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}

    encList = [axVIT, patchVIT, patchVIT]
    encArgs = [axArgs, kdArgs, patchArgs]
    

if fractal:
    sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/fractalmasks/mask_R" + str(R) + ".png")))
else:
    sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/masks/mask_R" + str(R) + ".png")))



# plt.imsave(f'{randomnumber}/samplingmask{randomnumber}.png',np.abs(sampling_mask))
sm = rearrange(sampling_mask, 'h w -> 1 1 h w')
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)
sampling_mask.to('cuda')


if original:
# Define the model
    dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)
else:
    dcenc = cascadeNet(N, encList, encArgs, FRAC_DC, lamb)
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

wandb.init(project="Fractal Transformer", config={"Original":original, "Fractal": fractal, "Sampling Pattern": R, "epochs": epochs, "lambda": lamb, "Batch Size": BATCH_SIZE})
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
        wandb.log({'trainlos': loss.item()}) 
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
                wandb.log({"Original Image": originalimages, "recon": images})
            
            valuessim = ssim(phRecon1[0:1, :, :, :].to('cuda'),singleimage1[0:1, :, :, :].to('cuda'))
            if valuessim > bestSSim:
                bestSSim = valuessim
                originalimages = wandb.Image(np.abs(singleimage1[0, 0, :, :].cpu()), caption="Original Image")
                images = wandb.Image(np.abs(phRecon1[0, 0, :, :].cpu()), caption="Image Recon")
                print(bestSSim)
                wandb.log({'bestssim': bestSSim}) 
                if epoch > 8:
                    wandb.log({"Original Image": originalimages, "recon": images})
            
                
            
            if loss.item() < running_loss2:
                running_loss2 = loss.item() 
                print("_________________")
                print("New Best Iterate: " + str(running_loss2) )
                print("_________________")
            finalrunning_loss2 = loss.item()
            totalitersval +=1
            wandb.log({'valloss': loss.item(), 'ssim': valuessim}) 
            
    # print(totaliterstrain)
    if epoch == 0:
        zfimages = wandb.Image(np.abs(zf_image[-1, 0, :, :].cpu()), caption="Zero Fill Image")
        wandb.log({"Zero Fill": zfimages})        
        print(ssim(phRecon1[-2:-1, :, :, :].to('cuda'),singleimage1[-2:-1, :, :, :].to('cuda')))
       
    print("___________________________________")
    print(f"Epoch {epoch} Train loss: {running_loss}")
    print(f"Epoch {epoch} Best Train loss: {finalrunning_loss}")
    print(f"Epoch {epoch} Best Val loss: {running_loss2}")
    print("___________________________________")

    end = time.time()
    elapsed = end - start
    print("Execution took " + str(elapsed) + " seconds")
    wandb.log({'epoch': epoch, "time per epoch": elapsed}) 

    totaltime += elapsed
    print("Total Time (Mins): " + str((1 / 60) * totaltime))
    print(f"Estimated Time Left (Mins): {(1 / 60) * (epochs - epoch +1) * (totaltime / (epoch + 1))}")


    print("___________________________________")
    print("___________________________________")
    print("___________________________________")

print(f"Total training iterations {totaliterstrain}")
print(f"Total validation iterations {totalitersval}")
print("Done")

