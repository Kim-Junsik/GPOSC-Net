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

def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

class DentalBase(Dataset):
    def __init__(self, config=None, **kwargs):
        self.data_root = '/mnt/dental/new_total_research/post_processing_image/total/SNU'
        self.data = glob.glob(os.path.join(self.data_root, '*'))
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.2, random_state = 42)
        self.size = (kwargs['size'], kwargs['size'])
        self.image = kwargs['image']
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx]
        number = data.split('/')[-1]
        if os.exists(os.path.join(data, f'{number}_B.png')):
            data_output = glob.glob(os.path.join(data, f'{number}_B.png'))[0]
        else:
            data_output = glob.glob(os.path.join(data, f'{number}_PRE.png'))[0]
            
        if os.exists(os.path.join(data, f'{number}_A.png')):
            data_input = glob.glob(os.path.join(data, f'{number}_A.png'))[0]
        else:
            data_input = glob.glob(os.path.join(data, f'{number}_PRE.png'))[0]
            
        input_sequence  = data_input.split('/')[-1].split('_')[-1].split('.')[0]
        output_sequence = data_output.split('/')[-1].split('_')[-1].split('.')[0]
        
        input_landmark = np.array(data_input.replace('.png', '.npy'))
        output_landmark = np.array(data_input.replace('.png', '') + f'_{output_sequence}.npy')
        
        
            
        landmark = np.load(landmark + '')
        landmark_x, landmark_y = landmark[:,0], landmark[:,1]-240
        landmark_y = np.where(landmark_y<2120, landmark_y, 2120)
        
        data = pydicom.dcmread(data).pixel_array
        shape_x, shape_y = data.shape[0], 1880
        landmark_x = (landmark_x / shape_x * 512).astype(np.uint8)
        landmark_y = (landmark_y / shape_y * 512).astype(np.uint8)
        landmark = np.stack([landmark_x, landmark_y], 1).reshape(-1)
        
        data = cv2.resize(data[:,2360//2-940:2360//2+940], self.size)
        data = data/255
        data = torch.from_numpy(data).unsqueeze(2)
        
        if not self.image:
            pre = torch.zeros_like(data)
        else:
            pre = data

        return {'output': data, 'landmarkwithpre': {'landmark_input_output': landmark, 'input': pre}}
        
        
class DentalTrain(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.train_data
        with open('dental_train.pickle', 'rb') as f:
            self.landmark = pickle.load(f)

        
class DentalValidation(DentalBase):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.data = self.test_data
        with open('dental_valid.pickle', 'rb') as f:
            self.landmark = pickle.load(f)
