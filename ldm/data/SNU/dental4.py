import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import pydicom
import cv2

import glob
import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths
import torch
# from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from sklearn.model_selection import train_test_split



import numpy as np
import cv2


class DentalBase(Dataset):
    def __init__(self, config=None, **kwargs):
        self.out_point_idx = [7,8,9,10,12,18,19,20,21,22, 31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        self.data_path = kwargs['data_path']
        self.data = glob.glob(os.path.join(self.data_path, '*.png'))
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.2, random_state = 42)
        self.size = (kwargs['size'], kwargs['size'])
        self.image = kwargs['image']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data = np.array(Image.open(data).resize(self.size)) / 255.0
        data = data * 2 - 1


        return {'image': data[:,:,:1]}
        
        
class DentalTrain(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.train_data
#         with open('dental_train.pickle', 'rb') as f:
#             self.landmark = pickle.load(f)

        
class DentalValidation(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.test_data
#         with open('dental_valid.pickle', 'rb') as f:
#             self.landmark = pickle.load(f)

class DentalBaseStage2(Dataset):
    def __init__(self, config=None, **kwargs):
        self.out_point_idx = [7,8,9,10,12,18,19,20,21,22, 31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        self.data_path = kwargs['data_path']
        self.data = glob.glob(os.path.join(self.data_path, '*A*.png')) + glob.glob(os.path.join(self.data_path, '*PRE*.png'))
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.2, random_state = 42)
        self.size = (kwargs['size'], kwargs['size'])
        self.image = kwargs['image']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        A_PRE = self.data[idx]
        B = A_PRE.replace('A', 'B') if 'A' in A_PRE else A_PRE.replace('PRE', 'B')
        
        landmark_A_PRE = np.load(A_PRE.replace('stylegan2_inversion_lat_png', 'stylegan2_inversion_lat_npy').replace('png', 'npy')).astype(np.float32)[self.out_point_idx]
        landmark_B = np.load(B.replace('stylegan2_inversion_lat_png', 'stylegan2_inversion_lat_npy').replace('png', 'npy')).astype(np.float32)[self.out_point_idx]

        A_PRE = np.array(Image.open(A_PRE)) / 255.0
        B = np.array(Image.open(B)) / 255.0
        
        data = np.concatenate([A_PRE[:,:,:1], B[:,:,:1]], 2)
        landmark = np.concatenate([landmark_B, landmark_A_PRE], 0)
        
        if not self.image:
            pre = np.zeros_like(data)
        else:
            pre = data

        return {'image': data, 'landmark': landmark}
        
        
class DentalTrainStage2(DentalBaseStage2):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.train_data
#         with open('dental_train.pickle', 'rb') as f:
#             self.landmark = pickle.load(f)

        
class DentalValidationStage2(DentalBaseStage2):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.test_data
#         with open('dental_valid.pickle', 'rb') as f:
#             self.landmark = pickle.load(f)
