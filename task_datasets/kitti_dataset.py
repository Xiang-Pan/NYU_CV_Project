'''
Author: Xiang Pan
Date: 2021-11-10 09:01:19
LastEditTime: 2021-12-16 23:33:49
LastEditors: Xiang Pan
Description: 
FilePath: /project/task_datasets/kitti_dataset.py
@email: xiangpan@nyu.edu
'''
# from torch.utils import data
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class semantic_dataset(Dataset):
    def __init__(self, task = "KITTI", split = 'train', transform = None):
        self.split = split
        path = "/".join(["./cached_datasets", task, split])
        print(task)
        if task == "KITTI":
            self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
            self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
            self.ignore_index = 250
            self.class_map = dict(zip(self.valid_labels, range(19)))
            # 1242x375
            self.img_size = (376, 1242)
            self.img_resize_size = (1242, 376)
            self.transform_image_size = (3, 376, 1242)
            if self.split in ['train', 'val']:
                self.img_path = './cached_datasets/KITTI/training/image_2/'    
                self.mask_path = './cached_datasets/KITTI/training/semantic/'
            else:
                self.img_path = './cached_datasets/KITTI/testing/image_2/'
                self.mask_path = None
        elif task == "unity-streetview-low-res":
            # print(task)
            # 842x474
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.ignore_index = 250
            self.void_labels = [
                0
            ]
            self.valid_labels = [
                i for i in range(1, 35)
            ]
            self.class_map = dict(zip(self.valid_labels, range(34)))
            # print(self.class_map)
            self.img_path = './cached_datasets/unity-streetview-low-res/RGB'
            self.mask_path = './cached_datasets/unity-streetview-low-res/SemanticSegmentation_gray/'
        elif task == "unity-streetview-high-res":
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.ignore_index = 250
            self.void_labels = [
                0
            ]
            self.valid_labels = [
                i for i in range(1, 35)
            ]
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.class_map = dict(zip(self.valid_labels, range(34)))
            self.img_path = './cached_datasets/unity-streetview-high-res/RGB/'
            self.mask_path = './cached_datasets/unity-streetview-high-res/SemanticSegmentation_gray/'
        elif task == "unity-cameraview-low-res":
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.ignore_index = 250
            self.void_labels = [
                0
            ]
            self.valid_labels = [
                i for i in range(1, 35)
            ]
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.class_map = dict(zip(self.valid_labels, range(34)))
            self.img_path = './cached_datasets/unity-cameraview-low-res/RGB/'
            self.mask_path = './cached_datasets/unity-cameraview-low-res/SemanticSegmentation_gray/'
        elif task == "unity-cameraview-high-res":
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.ignore_index = 250
            self.void_labels = [
                0
            ]
            self.valid_labels = [
                i for i in range(1, 35)
            ]
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.class_map = dict(zip(self.valid_labels, range(34)))
            self.img_path = './cached_datasets/unity-cameraview-high-res/RGB/'
            self.mask_path = './cached_datasets/unity-cameraview-high-res/SemanticSegmentation_gray/'
        elif task == "unity-streetview-complex":
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.ignore_index = 250
            self.void_labels = [
                0
            ]
            self.valid_labels = [
                i for i in range(1, 35)
            ]
            self.img_size = (474, 842)
            self.img_resize_size = (842, 474)
            self.transform_image_size = (3, 474, 842)
            self.class_map = dict(zip(self.valid_labels, range(34)))
            self.img_path = './cached_datasets/unity-streetview-complex/RGB/'
            self.mask_path = './cached_datasets/unity-streetview-complex/SemanticSegmentation_gray/'


        # elif task == "unity-streetview-medium-res":
        #     self.img_size = (474, 842)
        #     self.img_resize_size = (842, 474)
        #     self.transform_image_size = (3, 474, 842)
        #     self.void_labels = [
        #         i for i in range(35,255)
        #     ]
        #     self.valid_labels = [
        #         i for i in range(0, 35)
        #     ]
        #     self.class_map = dict(zip(self.valid_labels, range(35)))

        #     self.img_path = './cached_datasets/unity-streetview-medium-res/RGB/'
        #     self.mask_path = './cached_datasets/unity-streetview-medium-res/SemanticSegmentation/'
        


        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
        ])
        print(self.img_path)
        self.img_list = self.get_filenames(self.img_path)
        if self.split == 'train':
            self.mask_list = self.get_filenames(self.mask_path)
        elif self.split == 'val':
            self.mask_list = self.get_filenames(self.mask_path)
        elif self.split == 'test':
            self.mask_list = None
        
        # Split between train and valid set (80/20)
        if split in ['train', 'val']:
            random_inst = random.Random(23333)  # for repeatability
            n_items = len(self.img_list)
            idxs = random_inst.sample(range(n_items), n_items // 5)
            if self.split == "train":
                idxs = [idx for idx in range(n_items) if idx not in idxs]
            self.img_list = [self.img_list[i] for i in idxs]
            self.mask_list = [self.mask_list[i] for i in idxs]
        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.resize(img, self.img_resize_size)
        mask = None
        if self.split in ['train', 'val']:
            mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_resize_size)
            mask = self.encode_segmap(mask)
            assert(mask.shape == self.img_size)
            # print(mask== save)
        if self.transform:
            img = self.transform(img)
            assert(img.shape == self.transform_image_size)
        # else :
        #     assert(img.shape == (376, 1242, 3))
        
        if self.split in ['train', 'val']:
            return img, mask
        else :
            return img
    
    def encode_segmap(self, mask):
        '''
        Sets void classes to zero so they won't be considered for training
        '''
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask
    
    def get_filenames(self, path):
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

def main():
    dataset = semantic_dataset(task = "unity-streetview-high-res", split = 'train')
    dataset = semantic_dataset(task = "unity-streetview-low-res", split = 'val')


if __name__ == "__main__":
    main()

