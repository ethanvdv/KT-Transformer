"""
Creates generic vision transformers.

Author: Ethan van der Vegt
Date: December 20, 2022
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

class KTVIT(nn.Module):
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
    def __init__(self, N, patch_size=16, nu=1, kaleidoscope=False, layerNo=2, numCh=1, d_model=None, 
                    nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1, activation='relu', 
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None):
        super(KTVIT, self).__init__()
        # Define the number of iterations to go through the transformer
        self.layerNo = layerNo
        # Define the number of channels (2 means imaginary)
        self.numCh = numCh
        # Define the image size
        self.N = N
        # Define the patch size
        self.patch_size = patch_size
        
        # Determine d_model size
        if d_model is None:
            d_model = patch_size * patch_size * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        # Whether or not kaleidoscope
        self.kaleidoscope = kaleidoscope

        # For each layer, cascade an image transformer
        transformers = []
        for _ in range(layerNo):
            transformers.append(KTEncoder(self.N, self.patch_size, nu, numCh, kaleidoscope,
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
            im = im_denoise

        # Return the final output and the residual 
        return im



class KTEncoder(nn.Module):
    """
    This method applies the pseudo inverse before it is processed by the KT encoder.
    """
    def __init__(self, image_size, patch_size, nu=1, numCh=1, kaleidoscope=False, d_model=512, nhead=8, 
                num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05,
                batch_first=True, device=None, dtype=None, norm=None):
        super().__init__()
        # Define the transformer
        self.encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, layer_norm_eps, batch_first, device, dtype)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers, norm=norm)
        # Define size of the transformer 
        self.d_model = d_model

        ## Define the image size params
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Get the patch dimensionality
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * numCh

        # Define Kaleidoscope transform
        if kaleidoscope:
            self.to_embedding = nn.Sequential(
                Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
                nn.Linear(patch_dim, d_model)
            )
            self.from_embedding = Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)', k1=patch_height, k2=patch_width, h=image_height // patch_height, c=numCh)
        else:
            # Embed the image in patches
            self.to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=numCh, h=image_height // patch_height, p1=patch_height, p2=patch_width)

        # Define positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Define layer normalisation and linear transformation. As well-as de-patching the image.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding,
        )

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        self.mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(nu,image_height)


    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None):

        x = img

        x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        
        # Get the patch representation
        x = self.to_embedding(x)
        
        # Get the positional embedding
        x = x + self.pos_embedding

        # Perform dropout
        x = self.dropout(x)

        # Get the output of the transformer
        x = self.encoder(x, src_mask)

        # Pass-through multi-layer perceptron and un-patch
        x = self.mlp_head(x)     
        
        x = Kaleidoscope.ApplyMKTransform(x, self.mktindexes)  
        
        
        # Return the output
        return x
