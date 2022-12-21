import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib
import random
import math
from skimage.transform import resize
import matplotlib.pyplot as plt

# Get data from OASIS Dataset

class GetDatasetFolder(Dataset):

    def __init__(self, path, val_offset=150,train=True):
        # Combine pattern with path     
        self.path = path
        # self.base_path = path + 'MPRAGE'
        # self.base_path = path + 'slices_seg_pad'
        self.base_path = path + 'slices_pad'
        
        self.all_samples = os.listdir(self.base_path)
        self.all_samples.sort()

        if train:
            self.all_samples = self.all_samples[:-val_offset]
        else:
            self.all_samples = self.all_samples[len(self.all_samples)-val_offset:]
        # self.all_samples = self.all_samples[:-val_offset]
        random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)
        
    
    def __getitem__(self, item):
        scan_id = self.all_samples[item]

        # Load T1 image
        
        volume_t1, __ = self._load_nii(path=f'{self.path}slices_pad/{scan_id}')
        # 
        volume_t1 = (volume_t1 - np.min(volume_t1)) / (np.max(volume_t1) - np.min(volume_t1))
        # volume_t1 = volume_t1[:][104][:]
        
        # try:
        #     volume_t1 = volume_t1[:][random_index][:]
        # except IndexError: 
        #     print(f"aaaa{random_index}")

        # volume_t1 = torch.from_numpy(volume_t1).unsqueeze(0)
        volume_t1 = torch.from_numpy(volume_t1)
        
        # idx = torch.randperm(volume_t1.shape[0])

        # volume_t1 = volume_t1[idx]
        
        
        return volume_t1

    @staticmethod
    def _load_nii(path: str, size: int = None, primary_axis: int = 1, dtype: str = "float32"):
        """Load a neuroimaging file with nibabel, [w, h, slices]
        https://nipy.org/nibabel/reference/nibabel.html
        Args:
            path (str): Path to nii file
            size (int): Optional. Output size for h and w. Only supports rectangles
            primary_axis (int): Primary axis (the one to slice along, usually 2)
            dtype (str): Numpy datatype
        Returns:
            volume (np.ndarray): Of shape [w, h, slices]
            affine (np.ndarray): Affine coordinates (rotation and translation),
                                shape [4, 4]
        """
        # Load file
        data = nib.load(path, keep_file_open=False)
        volume = data.get_fdata(caching='unchanged')  # [w, h, slices]
        # print(volume.shape)
        affine = data.affine

        # Squeeze optional 4th dimension
        if volume.ndim == 4:
            volume = volume.squeeze(-1)

        # Resize if size is given and if necessary
        if size is not None and (volume.shape[0] != size or volume.shape[1] != size):
            volume = resize(volume, [size, size, size])

        # Convert
        volume = volume.astype(np.dtype(dtype))

        # Move primary axis to first dimension
        volume = np.moveaxis(volume, primary_axis, 0)
        plt.imsave(f'mask_Rasdfdsf1.png', np.abs(volume[:,:]))
        print(volume.shape)

        # volume = volume[40:256-40,40:256-40]
        # volume = volume[32:256-32,32:256-32]
        volume = volume[24:256-24,24:256-24]

        # volume = volume[70:138,:,:]
        
        return volume, affine