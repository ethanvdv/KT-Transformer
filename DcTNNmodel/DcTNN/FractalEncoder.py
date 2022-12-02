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


class patchFracVIT(nn.Module):
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
        super(patchFracVIT, self).__init__()
        # Define the number of iterations to go through the transformer
        self.layerNo = layerNo
        # Define the number of channels (2 means imaginary)
        self.numCh = numCh
        # Define the image size
        self.N = N
        
        # Define the patch size
        # self.patch_size = nu
        self.nu = nu
        self.sigma = sigma
        
        if np.mod(N - sigma,nu) == 0:
            shift = int((N - sigma)/nu)

        else:
            shift = int((N + sigma)/nu)
        
        # Determine d_model size
        if d_model is None:
            d_model = shift * nu * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))

        # For each layer, cascade an image transformer
        transformers = []
        for _ in range(layerNo):
            transformers.append(fracEncoder(self.N, nu, sigma, numCh,
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



class fracEncoder(nn.Module):
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
        
        self.nu = nu
        self.N = image_size
        device = 'cuda'
        # print(f'{image_size} {sigma} {nu} {np.mod(image_size - sigma,nu)} ')
        if np.mod(image_size - sigma,nu) == 0:
            # print(f'{image_size} {sigma} {nu} {np.mod(image_size - sigma,nu)} ')
            self.case = 1
            self.shift = int((image_size - sigma)/nu)

        else:
        # if np.mod(image_size - sigma,nu) == 0
            self.case = 2
            self.shift = int((image_size + sigma)/nu)
            
        print(self.case)
        patch_dim = int(self.shift) * int(self.shift) * numCh
        
        num_patches = int(self.nu * self.nu)
        
        # print(f'{nu} and {self.shift} , {num_patches}, {d_model}')
        

        # if nu == 7:
        #     value = (image_size - sigma)/25
        # else:
        #     value = nu
            

        self.mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=nu, N=image_size)
        self.mktindexes.to(device)
        # Define positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        if self.case == 1:
            # Embed the image in patches
            
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=self.shift, p2=self.shift),
                Rearrange('b h (p1) (p2 c) -> b h (p1 p2 c)', p1=self.shift, p2=self.shift),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = nn.Sequential(
                Rearrange('b h (p1 p2 c) -> b h (p1) (p2 c)', p1=self.shift, p2=self.shift),
                Rearrange('b (h w) (p1) (p2 c)-> b c (h p1) (w p2)', c=numCh, h=(self.N - 1) // self.shift, p1=self.shift, p2=self.shift)
            )
            
        if self.case == 2: 
            # Embed the image in patches
            self.to_patch_embedding = nn.Sequential(
                # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=nu, p2=nu),
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=self.shift, p2=self.shift),
                Rearrange('b h (p1) (p2 c)-> b h (p1 p2 c)', p1=self.shift, p2=self.shift),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = nn.Sequential(
                Rearrange('b h (p1 p2 c) -> b h (p1) (p2 c)', p1=self.shift, p2=self.shift),
                Rearrange('b (h w) (p1) (p2 c) -> b c (h p1) (w p2)', c=numCh, h=(self.N + 1) // self.shift, p1=self.shift, p2=self.shift)
            )
            
            
        # Define layer normalisation and linear transformation. As well-as de-patching the image.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding 
        )
                

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

        # N = 176
        # self.changes5 = Kaleidoscope.MKTkaleidoscopeIndexes(5,N)
        # self.changes3 = Kaleidoscope.MKTkaleidoscopeIndexes(3,N)
        # self.changes7 = Kaleidoscope.MKTkaleidoscopeIndexes(7,N)



    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None):

        x = img
        # x1 = img.detach().clone()
        # print(x.shape)
        # print('1')
        x.to('cuda')

        # print(x.shape)
        # out5 = Kaleidoscope.pseudoInvMKTransform(x.clone(), self.changes5)
        # out3 = Kaleidoscope.pseudoInvMKTransform(x.clone(), self.changes3)
        # out = torch.add(out5,out3)
        # # out59 = Kaleidoscope.pseudoInvMKTransform(x.detach().clone(), self.changes59)
        # out7 = Kaleidoscope.pseudoInvMKTransform(x.clone(), self.changes7)
        # out = torch.add(out,out7)
        # print(x.shape)
        # print(out.shape)
        # x = torch.subtract(x, out)

        x = Kaleidoscope.ApplyMKTransform(x, self.mktindexes)
        # x = Kaleidoscope.newPseudoKT(x, self.mktindexes)
        # x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        # print('2')
        # print(x.shape)
        if self.case == 1:
            # Get the patch representation
            # print(x.shape)
            # print(x[:, :, :-1, :-1].shape)
            x1 = x.clone()
            x = self.to_patch_embedding(x[:, :, :-1, :-1].to('cuda'))
        
        if self.case == 2:
            # print('3')
            # print(x.shape)
            x = torch.cat((x, x[:,:,[-1],:].to('cuda')), 2)
            # print('4')
            # print(x.shape)

            x = torch.cat((x, x[:,:,:,[-1]].to('cuda')), 3)     
            # print('5')
            # print(x.shape)
            x = self.to_patch_embedding(x)
            

        # Get the positional embedding
        x = x + self.pos_embedding
        
        # Perform dropout
        x = self.dropout(x)
        
        # Get the output of the transformer
        x = self.encoder(x, src_mask)
        # Pass-through multi-layer perceptron and un-patch
        
        if self.case == 1:
            x = self.mlp_head(x)
            x = torch.cat((x, x1[:,:,[-1],:-1].to('cuda')), 2)
            x = torch.cat((x, x1[:,:,:,[-1]].to('cuda')), 3)
            # 
            # x = torch.cat((x, x[:,:,[-1],:].to('cuda')), 2)
            # x = torch.cat((x,torch.cat((x[:,:,:,[-1]].to('cuda'), x[:,:,-1,[-1]].to('cuda')), 3).to('cuda')), 3)
        if self.case == 2:
            
            x = self.mlp_head(x)
            x = x[:,:, :-1, :-1].to('cuda')

        
        # x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        
        
        # Return the output
        return x


