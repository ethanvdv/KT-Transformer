import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import torch
from PIL import Image
from CreatePhantom import *
from Kaleidoscope import *
import wandb
from PIL import Image, ImageOps
# def kal_round(x, sigma):
#     '''
#     Rounds x up or down depending on whether sigma is positive or negative
#     '''
#     if sigma <= 0:
#         return int(np.floor(x))
#     else:
#         return int(np.ceil(x))

# def kaleidoscope(img, nu, sigma):
#     '''
#     Perform a nu,sigma-Kaleidoscope transform on img
#     Modified from Jacobs code
#     '''
    

#     # Perform kaleidoscope transform
#     h, w = img.shape
    
#     #Wraps so a 0 transform is actually a "N" transform
#     #Needed if we divide by zero
#     if nu == 0:
#         nu = h
        
#     if sigma == 0:
#         sigma = h
    


#     # img = img // np.abs(sigma) # Normalise image


#     # Perform kaleidoscope transform
#     h, w = img.shape

#     rows = np.arange(h)
#     cols = np.arange(w)

    
#     values1 = np.arange(h)
#     values2 = np.arange(w)
#     for r in rows:
#         for c in cols:

#             m1 = np.mod(kal_round(h / nu, sigma) * np.mod(r, nu) + sigma * (r // nu), h)
#             m2 = np.mod(kal_round(w / nu, sigma) * np.mod(c, nu) + sigma * (c // nu), w)
            
#             values1[r] = m1
#             values2[c] = m2
    
#     #Turn them into tensors
#     rowschange = torch.tensor(values1.copy(), dtype=torch.int64)
#     rowschange = repeat(rowschange,'h -> h c', c = h)

#     #Need to add an extra dimension since gather removes ones
#     rowschange = rearrange(rowschange, 'h w -> 1 1 h w 1')

#     colschange = torch.tensor(values2.copy(), dtype=torch.int64)
#     colschange = repeat(colschange,'h -> h c', c = w)
    
#     #Need to add an extra dimension since gather removes ones
#     colschange = rearrange(colschange, 'h w -> 1 1 w h 1')

#     #Use extra layer for RGB image
#     #Use extra layer for RGB image
#     zerosvector = torch.zeros_like(colschange)
    
#     changes = torch.cat((rowschange, colschange), 4)
#     changes = torch.cat((changes, zerosvector), 4)
    
#     #returns as (dummy) b h w c
#     return changes

# def tensorKaleidoscope(img, changes):
    
#     #Apply the change in rows
#     rows = torch.gather(img, 2, changes[:,:,:,:,0])

#     #Apply the change in columns
#     output = torch.gather(rows, 3, changes[:,:,:,:,1])
    
#     return output

# def pseudoInvKaleidoscope(source, changes2):
#     '''
#     Scatter is non-deterministic 
#     (from documentation)
#     It is "close" but not guranteed to be correct, if anything it adds more noise to it
#     '''
    
#     x = source.scatter(3,changes2[:,:,:,:,1],source)
    
#     output  = x.scatter(2,changes2[:,:,:,:,0], x)
    
#     return output

wandb.init(project="HeatMap")
N = 208

sampling_mask = np.array(ImageOps.grayscale(Image.open("KT-Transformer/DcTNNmodel/DcTNN/original.png")))

data = torch.tensor(sampling_mask.copy(), dtype=torch.float)
ph = rearrange(data,'h w -> 1 1 h w')
# ph = CreatePhantom(N)
ph.to('cuda')


# #Lets make the integer mask
# indexes = np.arange(N*N)

# #Reshape to square
# indexes = np.reshape(indexes, [N,N])

#Chose nu and sigma

nonLossyTransformsNu = []
nonLossyTransformsSigma = []

nonLossyTransformsNu1000 = []
nonLossyTransformsSigma1000 = []

aarray = np.arange(N*N)
aarrayes = np.reshape(aarray, [N,N])

i = 0
for nu in range(104):
    for sigma in range(208):               
        changes = Kaleidoscope.kaleidoscopeIndexes(N, nu, sigma)
        
        output = Kaleidoscope.ApplyMKTransform(ph,changes)
        output.to('cuda')
        outputs = wandb.Image(np.abs(output[0, 0, :, :].cpu()), caption=f"nu {nu} sigma {sigma}")
        
        input = Kaleidoscope.newPseudoKT(output, changes)
        inputs = wandb.Image(np.abs(input[0, 0, :, :].cpu()), caption=f"nu {nu} sigma {sigma}")
        input.to('cuda')
        
        
        # #Returns the indexes of the "rows" and the indexes of the "cols"
        # changes = kaleidoscope(indexes, nu, sigma)

        # output = tensorKaleidoscope(ph, changes)

        # input = pseudoInvKaleidoscope(output, changes)

        hhh = torch.sum(torch.nonzero(ph.to('cuda')-input.to('cuda')).to('cuda'))
        hhh2 = torch.count_nonzero(ph.to('cuda')-input.to('cuda')).to('cuda')
        if (hhh2 == 0):
            wandb.log({"GoodTransforms": [nu, sigma]})
            nonLossyTransformsNu.append([nu, sigma])    
            if (nu != 0):
                wandb.log({"KT": outputs, "Inverse": inputs})
        
        wandb.log({"Error": hhh, "nu": nu, "sigma": sigma})
        aarrayes[nu,sigma] = hhh

        print(i)
        
        i+=1
            

arrays = wandb.Image(aarrayes, caption=f"Full Heat Map")
wandb.log({"Heatmap": arrays})
print(nonLossyTransformsNu)
# plt.figure(100)
# plt.imshow(aarrayes, cmap='inferno', vmin=0, vmax=np.max(aarray))

# plt.title('Data Lost From Transform')
# plt.xlabel('nu')
# plt.ylabel('sigma')
# plt.colorbar()
plt.imsave(f'values3.png', aarrayes, cmap='inferno', vmin=0, vmax=np.max(aarray))


# plt.show()