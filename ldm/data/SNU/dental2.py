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
def img_crop(image_paths, points, train_size):
    pre_xs = points[:,0]
    pre_ys = points[:,1]
    min_x = int(np.min(pre_xs))
    min_y = int(np.min(pre_ys))
    max_x = int(np.max(pre_xs))
    max_y = int(np.max(pre_ys))
    #####################################################################
    #이미지, 계측점 크롭처리 부분
    #####################################################################
    if image_paths.split('/')[-1].split('.')[-1] == 'png':
        image = cv2.imread(image_paths)
    elif image_paths.split('/')[-1].split('.')[-1] == 'dcm':
        image = pydicom.dcmread(image_paths).pixel_array
    Crop_image =  image[min_y - int(image.shape[0]/10) :image.shape[0],  min_x-int(image.shape[1]/10):image.shape[1]]
    points[:,0] -= (min_x - int(image.shape[1]/10))
    points[:,1] -= (min_y - int(image.shape[0]/10))

    crop_points = points
    #이미지 학습할때 크기로 변경
    if(Crop_image.shape[0] > Crop_image.shape[1]):
        bigger_axis = Crop_image.shape[0]
    else:
        bigger_axis = Crop_image.shape[1]
    resize_factor = train_size[0] / bigger_axis
    image = cv2.resize(Crop_image, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    points = crop_points*resize_factor
    
    pading_image = np.zeros((train_size[0],train_size[1],1),dtype=np.uint8)
    pading_image[0:image.shape[0],0:image.shape[1]] = image[:,:,None]
    return pading_image/255, points


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

class DentalBase(Dataset):
    def __init__(self, config=None, **kwargs):
        self.out_point_idx = [7,8,9,10,12,18,19,20,21,22, 31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        self.data_path = kwargs['data_path']
        self.data = glob.glob(os.path.join(self.data_path, '*', '*.png'))
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.2, random_state = 42)
        self.size = (kwargs['size'], kwargs['size'])
        self.image = kwargs['image']
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx]
        landmark = np.load(data.replace('raw', 'raw_data_landmark_result').replace('dcm', 'npy')).astype(np.float32)
        data, landmark = img_crop(data, landmark, self.size)
        
        if not self.image:
            pre = np.zeros_like(data)
        else:
            pre = data

        return {'image': data[:,:,:1], 'landmarkwithpre': {'landmark_input_output': landmark, 'input': pre}}
        
        
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
