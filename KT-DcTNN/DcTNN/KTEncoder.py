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
    Defines a TNN that creates Kaleidoscope Shuffle Tokens
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
        super(KTVIT, self).__init__()
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
        
        # if np.mod(N - sigma,nu) == 0:
        #     shift = int((N - sigma)/nu)
        # else:
        #     shift = int((N + sigma)/nu)
            
        shift = int((N)/nu)

        
        # Determine d_model size
        if d_model is None:
            d_model = shift * shift * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3/2))
        
        # For each layer, cascade an image transformer
        transformers = []
        for _ in range(layerNo):
            transformers.append(KTEncoder(self.N, nu, sigma, numCh,
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
    Here we initialize a standard Encoder that utilizes Kaleidoscope Shuffle tokens.

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
        
        self.nu = int(nu)
        self.N = image_size
        device = 'cuda'
    
        # if np.mod(image_size - sigma,nu) == 0:
            # self.case = 1
        self.shift = int((image_size)/nu)

        
        print(f'{image_size} {sigma} {nu} {np.mod(image_size - sigma,nu)}')

        patch_dim = int(self.shift) * int(self.shift) * numCh
        
        num_patches = int(self.nu * self.nu)
        print(f"{d_model} and {dim_feedforward}, {patch_dim}")
        #Define the Indexes of the Shuffle
        # self.mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(nu,image_size, case=self.case)
        self.mktindexes = Kaleidoscope.KalIndexes(self.N, nu, sigma)

        # Define positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # if self.case == 1:
        #     # Embed the image in patches
            
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=self.shift, p2=self.shift),
            Rearrange('b h (p1) (p2 c) -> b h (p1 p2 c)', p1=self.shift, p2=self.shift),
            nn.Linear(patch_dim, d_model),
        )
        self.from_embedding = nn.Sequential(
            Rearrange('b h (p1 p2 c) -> b h (p1) (p2 c)', p1=self.shift, p2=self.shift),
            Rearrange('b (h w) (p1) (p2 c)-> b c (h p1) (w p2)', c=numCh, h=self.nu, p1=self.shift, p2=self.shift)
        )
        
        # if self.case == 2: 
        #     # Embed the image in patches
        #     self.to_patch_embedding = nn.Sequential(
        #         Rearrange('b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=self.shift, p2=self.shift),
        #         Rearrange('b h (p1) (p2 c)-> b h (p1 p2 c)', p1=self.shift, p2=self.shift),
        #         nn.Linear(patch_dim, d_model),
        #     )
        #     self.from_embedding = nn.Sequential(
        #         Rearrange('b h (p1 p2 c) -> b h (p1) (p2 c)', p1=self.shift, p2=self.shift),
        #         Rearrange('b (h w) (p1) (p2 c) -> b c (h p1) (w p2)', c=numCh, h=self.nu, p1=self.shift, p2=self.shift)
        #     )
            
            
        # Define layer normalisation and linear transformation. As well-as de-patching the image.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding 
        )
                

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        

    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None):

        x = img

        x.to('cuda')

        x = Kaleidoscope.ApplyKTTransform(x, self.mktindexes)

        x = self.to_patch_embedding(x)
            

        # Get the positional embedding
        x = x + self.pos_embedding
        
        # Perform dropout
        x = self.dropout(x)
        
        # Get the output of the transformer
        x = self.encoder(x, src_mask)

        x = self.mlp_head(x)

        x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        
        
        # Return the output
        return x


class InvKTVIT(nn.Module):
    """
    Defines a TNN that creates Kaleidoscope Shuffle Tokens
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
        super(InvKTVIT, self).__init__()
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
        
        # if np.mod(N - sigma,nu) == 0:
        #     shift = int((N - sigma)/nu)
        # else:
        #     shift = int((N + sigma)/nu)

        shift = int((N)/nu)
        
        # Determine d_model size
        if d_model is None:
            d_model = shift * shift * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3/2))
        
        # For each layer, cascade an image transformer
        transformers = []
        for _ in range(layerNo):
            transformers.append(InvKTEncoder(self.N, nu, sigma, numCh,
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



class InvKTEncoder(nn.Module):
    """
    Here we initialize a standard Encoder that utilizes Kaleidoscope Shuffle tokens.

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
        
        self.nu = int(nu)
        self.N = image_size
        device = 'cuda'
    
        # if np.mod(image_size - sigma,nu) == 0:
            # self.case = 1
        self.shift = int((image_size)/nu)

        
        print(f'{image_size} {sigma} {nu} {np.mod(image_size - sigma,nu)}')

        patch_dim = int(self.shift) * int(self.shift) * numCh
        
        num_patches = int(self.nu * self.nu)
        print(f"{d_model} and {dim_feedforward}, {patch_dim}")
        #Define the Indexes of the Shuffle
        # self.mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(nu,image_size, case=self.case)
        self.mktindexes = Kaleidoscope.KalIndexes(self.N, nu, sigma)

        # Define positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # if self.case == 1:
            # Embed the image in patches
            
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=self.shift, p2=self.shift),
            Rearrange('b h (p1) (p2 c) -> b h (p1 p2 c)', p1=self.shift, p2=self.shift),
            nn.Linear(patch_dim, d_model),
        )
        self.from_embedding = nn.Sequential(
            Rearrange('b h (p1 p2 c) -> b h (p1) (p2 c)', p1=self.shift, p2=self.shift),
            Rearrange('b (h w) (p1) (p2 c)-> b c (h p1) (w p2)', c=numCh, h=self.nu, p1=self.shift, p2=self.shift)
        )
            
        # if self.case == 2: 
        #     # Embed the image in patches
        #     self.to_patch_embedding = nn.Sequential(
        #         Rearrange('b c (h p1) (w p2) -> b (h w) (p1) (p2 c)', p1=self.shift, p2=self.shift),
        #         Rearrange('b h (p1) (p2 c)-> b h (p1 p2 c)', p1=self.shift, p2=self.shift),
        #         nn.Linear(patch_dim, d_model),
        #     )
        #     self.from_embedding = nn.Sequential(
        #         Rearrange('b h (p1 p2 c) -> b h (p1) (p2 c)', p1=self.shift, p2=self.shift),
        #         Rearrange('b (h w) (p1) (p2 c) -> b c (h p1) (w p2)', c=numCh, h=self.nu, p1=self.shift, p2=self.shift)
        #     )
            
            
        # Define layer normalisation and linear transformation. As well-as de-patching the image.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding 
        )
                

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        

    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None):

        x = img

        x.to('cuda')

        x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        
        x = self.to_patch_embedding(x)
            
        # Get the positional embedding
        x = x + self.pos_embedding
        
        # Perform dropout
        x = self.dropout(x)
        
        # Get the output of the transformer
        x = self.encoder(x, src_mask)

        x = self.mlp_head(x)

        x = Kaleidoscope.ApplyKTTransform(x, self.mktindexes)
        
        # Return the output
        return x
    
    
    

class InvaxVIT(nn.Module):
    """
    Defines the transformer for MRI reconstruction using exclusively a Transformer Encoder and axial tokens
    Args:
            N (int)                     -       Image Size
            layerNo (int)               -       Number of cascaded TNN
            numCh (int)                 -       Number of input channels (real, imag)
            d_model (int)               -       Model dimension
            nhead (int)                 -       Number of heads to use in multi-head attention
            num_encoder_layers (int)    -       Number of encoder layers within each encoder
            dim_feedforward (int)       -       Feedforward size in the MLP
            dropout (float)             -       Dropout of various layers
            activation                  -       Defines activation function for transformer encoder
    """
    def __init__(self, N, nu, sigma, layerNo=2, numCh=1, d_model=None, nhead=8, num_encoder_layers=2,
                    dim_feedforward=None, dropout=0.1, activation='relu',
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None):
        super(InvaxVIT, self).__init__()
        # Define the number of iterations to go through the transformer - this could be projections I suppose
        self.layerNo = layerNo
        # Define the number of channels (2 means imaginary)
        self.numCh = numCh
        # Define the image size
        self.N = N
        
        # Determine d_model size, can't be a prime number if we want MH Attention
        if d_model is None:
            d_model = N * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        

        # Cascade the denoising transformers
        transformers = []
        for _ in range(layerNo):
            transformers.append(InvaxialEncoder(self.N, nu, sigma, numCh, d_model, nhead, num_encoder_layers, dim_feedforward,
                                        dropout, activation, layer_norm_eps, batch_first, device, dtype))
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
    

class InvaxialEncoder(nn.Module):
    """
    Initializes a standard Encoder that utilizes axial attention.

    Args are the same as that defined by the normal Encoder class
    """
    def __init__(self, image_size, nu=1, sigma=1, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None, dropout=0.1, 
                    activation='relu', layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None, norm=None):
        # def __init__(self, image_size, nu=1, sigma=1, numCh=1, d_model=512, nhead=8, 
        #         num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05,
        #         batch_first=True, device=None, dtype=None, norm=None):
        super().__init__()
        # Define the transformer
        self.encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, layer_norm_eps, batch_first, device, dtype)
        # Because we cascade veritcal and horizontal transformers,
        # they must share the total number of encoder layers
        numLayers = num_layers // 2
        if numLayers < 1:
            numLayers = 1
        # Define each of the encoders
        self.horizontalEncoder = nn.TransformerEncoder(self.encoderLayer, numLayers, norm=norm)
        self.verticalEncoder = nn.TransformerEncoder(self.encoderLayer, numLayers, norm=norm)
        
        # Define size of the transformer 
        self.d_model = d_model

        ## Define image size
        image_height, image_width = pair(image_size)

        # Embed the slices horizontally
        self.to_horizontal_embedding = nn.Sequential(
            Rearrange('b c h w -> b h (w c)'),
            nn.Linear(image_width * numCh, d_model)
        )

        # Embed the slices vertically
        self.to_vertical_embedding = nn.Sequential(
            Rearrange('b c h w -> b w (h c)'),
            nn.Linear(image_height * numCh, d_model)
        )

        self.mktindexes = Kaleidoscope.KalIndexes(image_height, nu, sigma)
        
        # Define positional embedding
        self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, image_width, d_model))
        self.vertical_pos_embedding = nn.Parameter(torch.randn(1, image_height, d_model))

        # Define layer normalisation and linear transformation.
        self.horizontal_mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, image_width * numCh),
            Rearrange('b h (w c) -> b c h w', c=numCh)
        )
        self.vertical_mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, image_height * numCh),
            Rearrange('b w (h c) -> b c h w', c=numCh)
        )

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)



    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None, src_key_padding_mask=None):

        x = img

        x = Kaleidoscope.pseudoInvMKTransform(x, self.mktindexes)
        
        # Get the horizontal representation
        x = self.to_horizontal_embedding(x)
        
        # Get the positional embedding
        x = x + self.horizontal_pos_embedding

        # Perform dropout
        x = self.dropout(x)

        # Get the output of horizontal MHA
        x = self.horizontalEncoder(x, src_mask, src_key_padding_mask)

        # Pass-through multi-layer perceptron and un-token
        x = self.horizontal_mlp_head(x) 

        # Get the vertical representation
        x = self.to_vertical_embedding(x)
        
        # Get the positional embedding
        x = x + self.vertical_pos_embedding

        # Perform dropout
        x = self.dropout(x)

        # Get the output of horizontal MHA
        x = self.verticalEncoder(x, src_mask, src_key_padding_mask)

        # Pass-through multi-layer perceptron and un-token
        x = self.vertical_mlp_head(x)

        x = Kaleidoscope.ApplyKTTransform(x, self.mktindexes)
        # Return the output
        return x
