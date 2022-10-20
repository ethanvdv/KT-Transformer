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

# Get data from OASIS Dataset

class GetDatasetFolder(Dataset):

    def __init__(self, path, val_offset=150,train=True):
        # Combine pattern with path     
        self.path = path
        self.base_path = path + 'MPRAGE'
        
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
        
        volume_t1, affine_t1 = self._load_nii(path=f'{self.path}MPRAGE/{scan_id}')
        # random_index = random.randint(85, 123)
        volume_t1 = (volume_t1 - np.min(volume_t1)) / (np.max(volume_t1) - np.min(volume_t1))
        # volume_t1 = volume_t1[:][104][:]
        
        # try:
        #     volume_t1 = volume_t1[:][random_index][:]
        # except IndexError: 
        #     print(f"aaaa{random_index}")

        

        volume_t1 = torch.from_numpy(volume_t1).unsqueeze(0)
        
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

        return volume, affine