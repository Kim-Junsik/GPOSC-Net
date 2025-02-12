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

def img_crop(pre_image_paths, post_image_paths, pre_points, post_points, train_size):
    pre_xs = pre_points[:,0]
    pre_ys = pre_points[:,1]
    min_x = int(np.min(pre_xs))
    min_y = int(np.min(pre_ys))
    max_x = int(np.max(pre_xs))
    max_y = int(np.max(pre_ys))
    #####################################################################
    #이미지, 계측점 크롭처리 부분
    #####################################################################
    
    if pre_image_paths.split('/')[-1].split('.')[-1] == 'png':
        pre_image = cv2.imread(pre_image_paths)
    elif pre_image_paths.split('/')[-1].split('.')[-1] == 'dcm':
        pre_image = pydicom.dcmread(pre_image_paths).pixel_array

    if post_image_paths.split('/')[-1].split('.')[-1] == 'png':
        post_image = cv2.imread(post_image_paths)
    elif post_image_paths.split('/')[-1].split('.')[-1] == 'dcm':
        post_image = pydicom.dcmread(post_image_paths).pixel_array
        
    pre_crop_image =  pre_image[min_y - int(image.shape[0]/10) :image.shape[0],  min_x-int(image.shape[1]/10):image.shape[1]]
    post_crop_image =  post_image[min_y - int(image.shape[0]/10) :image.shape[0],  min_x-int(image.shape[1]/10):image.shape[1]]

    pre_points[:,0] -= (min_x - int(image.shape[1]/10))
    pre_points[:,1] -= (min_y - int(image.shape[0]/10))
    post_points[:,0] -= (min_x - int(image.shape[1]/10))
    post_points[:,1] -= (min_y - int(image.shape[0]/10))
    crop_pre_points = pre_points
    crop_post_points = post_points
    #이미지 학습할때 크기로 변경
    if(crop_pre_points.shape[0] > crop_pre_points.shape[1]):
        bigger_axis = crop_pre_points.shape[0]
    else:
        bigger_axis = crop_pre_points.shape[1]
        
    resize_factor = train_size[0] / bigger_axis
    pre_image = cv2.resize(pre_crop_image, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    post_image = cv2.resize(post_crop_image, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    pre_points = crop_pre_points*resize_factor
    post_points = crop_post_points*resize_factor
    
    pre_pading_image = np.zeros((train_size[0],train_size[1],1),dtype=np.uint8)
    pre_pading_image[0:pre_image.shape[0],0:pre_image.shape[1]] = pre_image
    
    post_pading_image = np.zeros((train_size[0],train_size[1],1),dtype=np.uint8)
    post_pading_image[0:image.shape[0],0:image.shape[1]] = post_image
    
    return pre_pading_image, post_pading_image, pre_points, post_points


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

class DentalBase(Dataset):
    def __init__(self, config=None, **kwargs):
        self.data_path = kwargs['data_path']
        self.data = glob.glob(os.path.join(self.data_path, 'SNU/*'))
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
        

        
        
        
        
        data = self.data[idx]
        landmark = np.load(data.replace('raw', 'raw_data_landmark_result').replace('dcm', 'npy'))
        data, landmark = img_crop(data, landmark, self.size)
        
        if not self.image:
            pre = np.zeros_like(data)
        else:
            pre = data

        return {'image': data, 'landmarkwithpre': {'landmark_input_output': landmark, 'input': pre}}
        
        
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
