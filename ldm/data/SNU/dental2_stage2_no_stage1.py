import random
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import math
import copy
import pickle
from sklearn.model_selection import train_test_split
import glob



class Dataset_Image_Point(Dataset):
    def __init__(self, mode = "train", transform=None, train_size = (512,512)): 
        self.ROOT = "/mnt/nas125/dental/new_total_research/post_processing_image/total"
        #                                       경조직                              |                   연조직
#         self.out_point_idx = [7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27, 31,32,33,34,35,36,37,38,39,40,41,42,43,44] <-- 전체

        #                                         경조직   |                   연조직
        self.out_point_idx = [7,8,10,11, 12,18,19,20,21,22, 31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        self.image_paths = []
        self.points = []
        
        paths = glob.glob('/mnt/nas125/dental/new_total_research/post_processing_image/SNU/*') + glob.glob('/mnt/nas125/dental/new_total_research/post_processing_image/KBU/*') + glob.glob('/mnt/nas125/dental/new_total_research/post_processing_image/KH/*')
        
        for path in paths:
            for p in glob.glob(os.path.join(path, '*.png')):                
                self.image_paths.append(p)
                self.points.append(p.replace('png', 'npy'))
                
        print(len(self.image_paths))

        self.mode = mode
        self.train_size = train_size
        
        self.transform = transform
        self.num_of_land = 33 #--> 연조직까지 포함이라 숫자 바뀜
        
    ##################################################################################################################
    #서브 모듈
    ##################################################################################################################
    def angle_trunc(self, a):
        while a < 0.0:
            a+= np.pi * 2
        return a

    def getAngleBetweenPoints(self,x_orig, y_orig, x_landmark, y_landmark):
        deltaY = y_landmark - y_orig
        deltaY = deltaY * -1
        deltaX = x_landmark - x_orig
        return self.angle_trunc(math.atan2(deltaY, deltaX))
    ##################################################################################################################

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        distance = []
        points = np.load(self.points[index])
#         patient_info = np.array(self.patient_infos[index])

        #경조직에서 최대 최소점 가져오기
        xs = points[:,0]
        ys = points[:,1]
        min_x = int(np.min(xs))
        min_y = int(np.min(ys))
        max_x = int(np.max(xs))
        max_y = int(np.max(ys))

        #####################################################################
        #이미지, 계측점 크롭처리 부분
        #####################################################################
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        
        crop_y = min_y - int(image.shape[0]/10) if min_y - int(image.shape[0]/10) > 0 else 0
        crop_x = min_x - int(image.shape[1]/10) if min_x - int(image.shape[1]/10) > 0 else 0
        
        Crop_image =  image[crop_y:image.shape[0],  crop_x:image.shape[1]]
        
        points[:,0] -= crop_x
        points[:,1] -= crop_y

        crop_points = points

        #이미지 학습할때 크기로 변경
        if(Crop_image.shape[0] > Crop_image.shape[1]):
            bigger_axis = Crop_image.shape[0]
        else:
            bigger_axis = Crop_image.shape[1]

        resize_factor = self.train_size[0] / bigger_axis

        image = cv2.resize(Crop_image, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
        points = crop_points*resize_factor

        pading_image = np.zeros((self.train_size[0],self.train_size[1],3),dtype=np.uint8)
        pading_image[0:image.shape[0],0:image.shape[1]] = image

        #process 를 편하게 하기 위해서 우선 agumentation 부터 적용
        sample = {'image': pading_image, 'landmarks': points}

        if self.transform:
            sample = self.transform(sample)

        image = sample['image'][:,:,:1]/255.
        points = sample['landmarks']
        points = points.astype("float").reshape(-1)
        
        return {'landmarks': points, 'image': image}



class Dataset_Landmark_Train(Dataset_Image_Point):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)

        self.image_paths, _, self.points, _ = train_test_split(self.image_paths, self.points, test_size = 0.2, random_state = 42)

class Dataset_Landmark_Valid(Dataset_Image_Point):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)

        _, self.image_paths, _, self.points = train_test_split(self.image_paths, self.points, test_size = 0.2, random_state = 42)