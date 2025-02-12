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
        self.data = glob.glob(os.path.join(self.data_root, '*'))
        self.size = (kwargs['size'], kwargs['size'])
        self.image = kwargs['image']
        self.out_point_idx = [7,8,9,10,12,18,19,20,21,22,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        self.threshold = kwargs['threshold']
        self.conditioning = kwargs['conditioning']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            A_path = glob.glob(os.path.join(self.data[idx], '*A.png'))[0]
        except:
            A_path = glob.glob(os.path.join(self.data[idx], '*PRE.png'))[0]
        B_path = glob.glob(os.path.join(self.data[idx], '*B.png'))[0]
        A_image = np.array(Image.open(A_path).resize(self.size)) / 255.0
        A_image = A_image * 2 - 1

        landmarkwithpre = dict()
        data = dict()
        landmarkwithpre['path'] = [A_path, B_path]
            
        
        if random.random() >= self.threshold: # A_image
            image = A_image.copy()
            if not len(self.conditioning)==0:
                A_landmark = np.load(A_path.replace('png', 'npy')).astype(np.float32)[:45]
                movement = np.concatenate([A_landmark, A_landmark], 1)
        
        else:
            image = np.array(Image.open(B_path).resize(self.size)) / 255.0
            image = image * 2 - 1

            if not len(self.conditioning)==0:

                B_landmark = np.load(B_path.replace('png', 'npy')).astype(np.float32)[:45]
                A_landmark = np.load(A_path.replace('png', 'npy')).astype(np.float32)[:45]
                movement = np.concatenate([A_landmark, B_landmark], 1)
            
        data['image'] = image[:,:,:1]
        if 'movement' in self.conditioning:
            landmarkwithpre['movement'] = movement[self.out_point_idx]
        else:
            landmarkwithpre['movement'] = np.zeros_like(movement[self.out_point_idx])
            
        if 'landmark' in self.conditioning:
            landmarkwithpre['landmark'] = A_landmark
            
        if 'pre' in self.conditioning:
            landmarkwithpre['pre'] = A_image[:,:,:1]
        else:
            landmarkwithpre['pre'] = np.zeros_like(A_image[:,:,:1])
            
        if 'line' in self.conditioning:
            line_path = A_path.replace('.png', '_Line.png')
            line = np.array(Image.open(line_path).resize(self.size))[:,:,:1] / 255.0
            landmarkwithpre['line'] = line
        
        data['landmarkwithpre'] = landmarkwithpre
        
        return data

        
class DentalTrain(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        
class DentalValidation(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)

class DentalTest(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)