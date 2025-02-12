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
import random

def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

        
class DentalBase(Dataset):
    def __init__(self, config=None, **kwargs):
        self.data_root = kwargs['data_path']
        self.data = glob.glob(os.path.join(self.data_root, '*.png'))
        self.size = (kwargs['size'], kwargs['size'])
        self.image = kwargs['image']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        image = np.array(Image.open(path).resize(self.size)) / 255.0
        image = image * 2 - 1
        data = {}
        data['image'] = image[:,:,:1]
        
        return data

        
class DentalTrain(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[:-500]
        
        
class DentalValidation(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[-500:]


class DentalTest(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)