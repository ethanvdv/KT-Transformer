import numpy as np
import time
from DcTNN.tnn import * 
from DcTNN.KalEncoder import * 
from dc.dc import *
from phantominator import shepp_logan
from PIL import Image, ImageOps
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from loadingimages import GetDatasetFolder
from torchmetrics import StructuralSimilarityIndexMeasure
# from torch.utils.tensorboard import SummaryWriter
print("25 and 7 - new batching system")
randomnumber = np.random.randint(1,100)
print(f"Random number to deal with namespace stuff: {randomnumber}")
def get_images(input):
    y = fft_2d(input) * sampling_mask
    zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :]

    return y, zf_image

def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


norm = 'ortho'
N = 176 # 320
R = 100
numCh = 1
lamb = True


# Define transformer encoder parameters
patchSize = 16
# patchSize2 = 11
# patchSize3 = 8

nhead_patch = 8
# nhead_patch2 = 11
nhead_axial = 8
layerNo = 1
d_model_axial = None
d_model_patch = None
num_encoder_layers = 2
numCh = numCh
dim_feedforward = None

# # Define the arrays of encoders and dictionaries
# patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
# kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
# axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}


# encList = [axVIT, patchVIT, patchVIT]
# encArgs = [axArgs,patchArgs, kdArgs]

'''

For Kaleidoscope stuff
kdArgs2 = {"patch_size": patchSize2, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch2, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
kdArgs3 = {"patch_size": patchSize3, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}



encList = [patchVIT, patchVIT, patchVIT]
encArgs = [kdArgs,kdArgs2, kdArgs3]
'''

sigma = 1

# # Define the dictionaries of parameter values
# # kd6Args = {"nu": 3, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 3, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

kd3Args = {"nu": 25, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

# kd2Args = {"nu": 35, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": 30}

# kdArgs = {"nu": 59, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": 30}

# kd5Args = {"nu": 5, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 5, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

kd4Args = {"nu": 7, "sigma": sigma, "layerNo": layerNo, "numCh": numCh, "nhead": 7, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}

# # Define the array of encoders
# # encList = [patch2VIT, patch2VIT, patch2VIT, patch2VIT,patch2VIT]
encList = [patch2VIT, patch2VIT]
# # Define the array of dictionaries
# # encArgs = [kdArgs, kd2Args, kd4Args, kd5Args,kd3Args]
encArgs = [kd3Args, kd4Args]

# Define the model
dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)

# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in dcenc.parameters() if p.requires_grad)

# device = 'cuda'
lr = 1e-4
weighting = 1e-7
dsc_loss = torch.nn.L1Loss()

optimizer = optim.Adam(dcenc.parameters(),lr)
epochs = 120
step = 0

BASE_PATH = '/home/groups/deep-compute/OASIS/'
BATCH_SIZE = 1
# batching = 15
offset = 100

ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)

val_ds = GetDatasetFolder(path=BASE_PATH, val_offset=offset, train=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=1)

# dcenc = dcenc.to(device)

print(f"Sampling pattern used {R}")

# Load a random sampling mask and inverse fourier shift
sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/masks/mask_R" + str(R) + ".png")))
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)
# sampling_mask = sampling_mask.to(device)

print(encArgs)
print(dsc_loss)
print(optimizer)
print("Number of trainable params: " + str(pytorch_total_params))
print(f'Number of Epochs {epochs}')
# print(f"Patch Size {patchSize}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Value Offset: {offset}")


totaliterstrain = 0
totalitersval = 0
for epoch in range(epochs):
    start = time.time()
    # Train
    print(f"Epoch: {epoch}/{epochs}")
    running_loss = 0
    for ph in dl:
        totaliterstrain += 1
        ph = rearrange(ph, 'b c h w -> c b h w').contiguous()
        # for iter in range(batching):
        index = np.random.randint(2,ph.size(dim=0))
        singleimage = ph[:index, :, :, :]
        y, zf_image = get_images(singleimage)
        optimizer.zero_grad()
        phRecon = dcenc(zf_image, y, sampling_mask)
        loss = dsc_loss(phRecon, singleimage) + total_variation_loss(phRecon,weighting)
        running_loss += loss.item()
        finalrunning_loss = loss.item()
        loss.backward()
        optimizer.step()
    
    
    #Attempt at Testing Dataset
    running_loss2 = 1
    for X in val_dl:
        # X = X.to(device)
        # y_true = y_true.to(device)
        totalitersval +=1
        ph = rearrange(X, 'b c h w -> c 1 b h w').contiguous()
        singleimage = ph[0, :, :, :, :]
        y, zf_image = get_images(singleimage)
        with torch.no_grad():
            phRecon = dcenc(zf_image, y, sampling_mask)
            loss = dsc_loss(phRecon, singleimage) + total_variation_loss(phRecon,weighting)
            if loss.item() < running_loss:
                running_loss2 = loss.item()
            finalrunning_loss2 = loss.item()
    
    #Print a final reconstruction
    if (np.mod(epoch,10) == 0) or (epoch == (epochs-1)):
        plt.imsave(f'{randomnumber}-257-{R}-{epoch}--recon.png',np.abs(phRecon[0, 0, :, :]))
        plt.imsave(f'{randomnumber}-257-{R}-{epoch}--image-.png',np.abs(singleimage[0, 0, :, :]))
        ssim = StructuralSimilarityIndexMeasure()
        print(f'SSIM at epoch {epoch}:')
        phRecon = rearrange(phRecon[0, 0, :, :], 'h w -> 1 1 h w')
        print(ssim(phRecon,singleimage))
        
    print("___________________________________")
    print(f"Epoch {epoch} Train loss: {running_loss}")
    print(f"Epoch {epoch} Final Train loss: {finalrunning_loss}")
    print(f"Best Epoch {epoch} Val loss: {running_loss2}")
    # print(f"Epoch {epoch} Final Val loss: {finalrunning_loss2}")
    print("___________________________________")
    end = time.time()
    elapsed = end - start
    print("Execution took " + str(elapsed) + " seconds")


    print(f"Total training iterations {totaliterstrain}")
    print(f"Total validation iterations {totalitersval}")
print("Done")



# dsc_loss = torch.nn.MSELoss()

# dsc_loss = DiceLoss()

# optimizer = optim.SGD(dcenc.parameters(), lr)
# writer_real = SummaryWriter(f"logs/real2")
# write_pred = SummaryWriter(f"logs/pred2")
# write_train = SummaryWriter(f"logs/train2")
# write_val = SummaryWriter(f"logs/val2")