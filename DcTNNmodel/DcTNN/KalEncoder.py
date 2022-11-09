"""
Creates generic vision transformers.

Author: Marlon Ernesto Bran Lorenzana
Date: February 15, 2022
"""

import torch
from dc.dc import *
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np
from einops import rearrange
from DcTNN.Kaleidoscope import *
# Helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class patch2VIT(nn.Module):
    """
    Defines a TNN that creates either Kaleidoscope or Patch tokens.
    Args:
        N (int)                     -       Image Size
        patch_size (int)            -       Size of patch or Kaleidoscope tokens
        kaleidoscope (bool)         -       If true the network will create Kaleidoscope tokens
        layerNo (int)               -       Number of cascaded TNN
        numCh (int)                 -       Number of input channels (real, imag)
        d_model (int)               -       Model dimension
        nhead (int)                 -       Number of heads to use in multi-head attention
        num_encoder_layers (int)    -       Number of encoder layers within each encoder
        dim_feedforward (int)       -       Feedforward size in the MLP
        dropout (float)             -       Dropout of various layers
        activation                  -       Defines activation function for transformer encoder
    """
    def __init__(self, N, nu=1, sigma=1, layerNo=2, numCh=1, d_model=None, 
                    nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1, activation='relu', 
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None):
        super(patch2VIT, self).__init__()
        # Define the number of iterations to go through the transformer
        self.layerNo = layerNo
        # Define the number of channels (2 means imaginary)
        self.numCh = numCh
        # Define the image size
        self.N = N
        
        # Define the patch size
        self.patch_size = nu
        self.nu = nu
        self.sigma = sigma
        # Determine d_model size
        if d_model is None:
            d_model = nu * nu * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))

        # For each layer, cascade an image transformer
        transformers = []
        for _ in range(layerNo):
            transformers.append(MKTEncoder(self.N, nu, sigma, numCh,
                                                            d_model, nhead, num_encoder_layers, dim_feedforward,
                                                            dropout, activation, layer_norm_eps, 
                                                            batch_first, device, dtype))
        self.transformers = nn.ModuleList(transformers)
    
    """
    xPrev should be [numBatch, numCh, ydim, xdim]
    """
    def forward(self, xPrev):

        im = xPrev

        # Go over the number of iterations to perform updating 
        for i in range(self.layerNo):           

            # Get a new image estimate based on previous estimate of x (xPrev) 
            im_denoise = self.transformers[i](im)
            im = im + im_denoise

        # Return the final output and the residual 
        return im


class cascadeNet(nn.Module):
    """
    Defines a TNN that cascades denoising networks and applies data consistency.
    Args:
        N (int)                     -       Image Size
        encList (array)             -       Should contain denoising network
        encArgs (array)             -       Contains dictionaries with args for encoders in encList
        dcFunc (function)           -       Contains the data consistency function to be used in recon
        lamb (bool)                 -       Whether or not to use a leanred data consistency parameter
    """
    def __init__(self, N, encList, encArgs, dcFunc=FFT_DC, lamb=True):
        super(cascadeNet, self).__init__()
        # Define lambda for data consistency
        if lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * 0.5)
        else:
            self.lamb = False
        # Define image size
        self.N = N
        # Define the data consistency function
        self.dcFunc = dcFunc

        # Cascade the transformers
        transformers = []
        for i, enc in enumerate(encList):
            transformers.append(enc(N, **encArgs[i]))

        self.transformers = nn.ModuleList(transformers)

    """
    xPrev should be [numBatch, numCh, ydim, xdim]
    y should be [numBatch, kCh, ydim, xdim]
    sampleMask should be [ydim, xdim]
    """
    def forward(self, xPrev, y, sampleMask):

        im = xPrev

        # Go over the number of iterations to perform updating 
        for i, transformer in enumerate(self.transformers):        

            # Denoise the image
            im = transformer(im)

            # Update the residual
            if self.lamb is False:
                im = self.dcFunc(im, y, sampleMask, None)
            else:
                im = self.dcFunc(im, y, sampleMask, self.lamb[i])

        # Return the final output
        return im


class MKTEncoder(nn.Module):
    """
    Here we initialize a standard Encoder that utilizes image patches or kaleidoscope tokens.

    Args are the same as that defined by the normal Encoder class
    """
    def __init__(self, image_size, nu=1, sigma=1, numCh=1, d_model=512, nhead=8, 
                num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05,
                batch_first=True, device=None, dtype=None, norm=None):
        super().__init__()
        # Define the transformer
        self.encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, layer_norm_eps, batch_first, device, dtype)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers, norm=norm)
        # Define size of the transformer 
        self.d_model = d_model
        patch_dim = nu * nu * numCh
        self.nu = nu
        self.N = image_size

        if np.mod(image_size - sigma,nu) == 0:
            self.case = 1
            self.shift = (image_size - sigma)/nu

        else:
            self.case = 2
            self.shift = (image_size + sigma)/nu
        
        num_patches = int(self.shift * self.shift)
            
        self.mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=self.shift,N=image_size)

        # Define positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        if self.case == 1:
            # Embed the image in patches
            self.to_patch_embedding = nn.Sequential(
                # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=nu, p2=nu),
                # Rearrange('c (h k1) (w k2) -> (h w) c k1 k2', k1=self.nu, k2=self.nu),
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.nu, p2=self.nu),
                nn.Linear(patch_dim, d_model),
            )
            # Define layer normalisation and linear transformation. As well-as de-patching the image.
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, patch_dim),
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=numCh, h=self.N // self.nu, p1=self.nu, p2=self.nu)
            )
            
        if self.case == 2: 
            # Embed the image in patches
            self.to_patch_embedding = nn.Sequential(
                # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=nu, p2=nu),
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.nu, p2=self.nu),
                nn.Linear(patch_dim, d_model),
            )
            # Define layer normalisation and linear transformation. As well-as de-patching the image.
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, patch_dim),
                # Rearrange('(h w) c k1 k2-> c (h k1) (w k2)', h= (self.N + 1) // self.nu, k1=self.nu, k2=self.nu),
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=numCh, h=(self.N + 1) // self.nu, p1=self.nu, p2=self.nu),
            )
                

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)



    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None):

        x = img

        
        x = Kaleidoscope.ApplyMKTransform(x, self.mktindexes)

        if self.case == 1:
            # Get the patch representation
            x = self.to_patch_embedding(x[:, :, :-1, :-1])
        
        if self.case == 2:
            #Build Extra rows    
            newtensor = torch.zeros(1, 177, 177)
            newtensor[:,:-1,:-1] = x[0,:,:,:]
            newtensor[:,:-1,-2:-1] = x[0, 0, :, -2:-1]
            newtensor[:,-2:-1,:-1] = x[0, 0, -2:-1, :]

            x = self.to_patch_embedding(newtensor)
            

        # Get the positional embedding
        x = x + self.pos_embedding
        
        # Perform dropout
        x = self.dropout(x)
        
        # Get the output of the transformer
        x = self.encoder(x, src_mask)
        # Pass-through multi-layer perceptron and un-patch
        
        if self.case == 1:
            
            undotensor = self.mlp_head(x)
            
            newtensor = torch.zeros(1, 176, 176)
            newtensor[:,:-1,:-1] = undotensor[0,:,:,:]
            newtensor[:,:-1,-2:-1] = undotensor[0, 0, :, -2:-1]
            newtensor[:,-2:-1,:-1] = undotensor[0, 0, -2:-1, :]

            x = newtensor
            
        if self.case == 2:
            
            undotensor = self.mlp_head(x)
            x = undotensor[:, :-1, :-1]

        x = rearrange(x,'c h w -> 1 c h w')
        x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        
        # Return the output
        return x


