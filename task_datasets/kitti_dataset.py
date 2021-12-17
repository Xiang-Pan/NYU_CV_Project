'''
Author: Xiang Pan
Date: 2021-11-10 09:01:19
LastEditTime: 2021-12-15 03:49:27
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

# class semantic_dataset(Dataset):
#     def __init__(self, task = "KITTI", split = 'training', transform = None):
#         self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
#         self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
#         self.ignore_index = 250
#         self.class_map = dict(zip(self.valid_labels, range(19)))
#         self.split = split
#         self.img_size = (376, 1242)

#         path = "/".join(["./cached_datasets", task, split])
#         if task == "KITTI":
#             if self.split == 'train':
#                 self.img_path = './cached_datasets/KITTI/training/image_2/'    
#                 self.mask_path = './cached_datasets/KITTI/training/semantic/'
#             else:
#                 self.img_path = './cached_datasets/KITTI/testing/image_2/'
#                 self.mask_path = None
#         elif task == "unity-streetview-low-res":
#             self.img_path = './cached_datasets/unity-streetview-low-res/RGBdb54b8d2-abed-46d1-afe7-6f1bdfd57e30/'
#             self.mask_path = './cached_datasets/unity-streetview-low-res/SemanticSegmentation/'
#         elif task == "unity-streetview-medium-res":
#             self.img_path = './cached_datasets/unity-streetview-medium-res/RGBf11e1f2e-c37a-441e-92b4-aceee444e439/'
#             self.mask_path = './cached_datasets/unity-streetview-medium-res/SemanticSegmentation/'

#         self.transform = transform
        
#         self.img_list = self.get_filenames(self.img_path)
#         self.mask_list = None
#         if self.split == 'train':
#             self.mask_list = self.get_filenames(self.mask_path)
        
#     def __len__(self):
#         return(len(self.img_list))
    
#     def __getitem__(self, idx):
#         img = cv2.imread(self.img_list[idx])
#         img = cv2.resize(img, self.img_size)
#         mask = None
#         if self.split == 'train':
#             mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
#             mask = cv2.resize(mask, (1242, 376))
#             mask = self.encode_segmap(mask)
#             assert(mask.shape == (376, 1242))
        
#         if self.transform:
#             img = self.transform(img)
#             assert(img.shape == (3, 376, 1242))
#         else :
#             assert(img.shape == (376, 1242, 3))
        
#         if self.split == 'train':
#             return img, mask
#         else :
#             return img
    
#     def encode_segmap(self, mask):
#         '''
#         Sets void classes to zero so they won't be considered for training
#         '''
#         for voidc in self.void_labels :
#             mask[mask == voidc] = self.ignore_index
#         for validc in self.valid_labels :
#             mask[mask == validc] = self.class_map[validc]
#         return mask
    
#     def get_filenames(self, path):
#         files_list = list()
#         for filename in os.listdir(path):
#             files_list.append(os.path.join(path, filename))
#         return files_list



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
    
    # print(dataset[0])

    # dataset = semantic_dataset(task = "KITTI", split = 'train')
    # print(dataset[0])



if __name__ == "__main__":
    main()



# preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')


# DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
# DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)




# class KITTIDataset(Dataset):
#     """Class for KITTI Semantic Segmentation Benchmark dataset.
#     Dataset link - http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015
#     There are 34 classes in the given labels. However, not all of them are useful for training
#     (like railings on highways, road dividers, etc.).
#     So, these useless classes (the pixel values of these classes) are stored in the `void_labels`.
#     The useful classes are stored in the `valid_labels`.
#     The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
#     (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
#     `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
#     the loss function when comparing with the output.
#     The `get_filenames` function retrieves the filenames of all images in the given `path` and
#     saves the absolute path in a list.
#     In the `get_item` function, images and masks are resized to the given `img_size`, masks are
#     encoded using `encode_segmap`, and given `transform` (if any) are applied to the image only
#     (mask does not usually require transforms, but they can be implemented in a similar way).
#     >>> from pl_examples import _DATASETS_PATH
#     >>> dataset_path = os.path.join(_DATASETS_PATH, "Kitti")
#     >>> _create_synth_kitti_dataset(dataset_path, image_dims=(1024, 512))
#     >>> KITTI(dataset_path, 'train')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
#     <...semantic_segmentation.KITTI object at ...>
#     """

#     IMAGE_PATH = os.path.join("training", "image_2")
#     MASK_PATH = os.path.join("training", "semantic")

#     def __init__(
#         self,
#         data_path: str,
#         split: str,
#         img_size: tuple = (1242, 376),
#         void_labels: list = DEFAULT_VOID_LABELS,
#         valid_labels: list = DEFAULT_VALID_LABELS,
#         transform=None,
#     ):
#         self.img_size = img_size
#         self.void_labels = void_labels
#         self.valid_labels = valid_labels
#         self.ignore_index = 250
#         self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
#         self.transform = transform

#         self.split = split
#         self.data_path = data_path
#         self.img_path = os.path.join(self.data_path, self.IMAGE_PATH)
#         self.mask_path = os.path.join(self.data_path, self.MASK_PATH)
#         self.img_list = self.get_filenames(self.img_path)
#         self.mask_list = self.get_filenames(self.mask_path)

#         # Split between train and valid set (80/20)
#         random_inst = random.Random(12345)  # for repeatability
#         n_items = len(self.img_list)
#         idxs = random_inst.sample(range(n_items), n_items // 5)
#         if self.split == "train":
#             idxs = [idx for idx in range(n_items) if idx not in idxs]
#         self.img_list = [self.img_list[i] for i in idxs]
#         self.mask_list = [self.mask_list[i] for i in idxs]

#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, idx):
#         img = Image.open(self.img_list[idx])
#         img = img.resize(self.img_size)
#         img = np.array(img)

#         mask = Image.open(self.mask_list[idx]).convert("L")
#         mask = mask.resize(self.img_size)
#         mask = np.array(mask)
#         mask = self.encode_segmap(mask)

#         if self.transform:
#             img = self.transform(img)

#         return img, mask

#     def encode_segmap(self, mask):
#         """Sets void classes to zero so they won't be considered for training."""
#         for voidc in self.void_labels:
#             mask[mask == voidc] = self.ignore_index
#         for validc in self.valid_labels:
#             mask[mask == validc] = self.class_map[validc]
#         # remove extra idxs from updated dataset
#         mask[mask > 18] = self.ignore_index
#         return mask

#     def get_filenames(self, path):
#         """Returns a list of absolute paths to images inside given `path`"""
#         files_list = []
#         for filename in os.listdir(path):
#             files_list.append(os.path.join(path, filename))
#         return files_list


