'''
Author: Xiang Pan
Date: 2021-12-12 23:23:22
LastEditTime: 2021-12-12 23:24:24
LastEditors: Xiang Pan
Description: 
FilePath: /project/task_datasets/unity_dataset.py
@email: xiangpan@nyu.edu
'''
from torch.utils.data import Dataset

class UnityDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        return self.data_path[idx]


class Residential_Interiors_Dataset(UnityDataset):
    def __init__(self, data_path, transform=None):
        super(Residential_Interiors_Dataset, self).__init__(data_path, transform)



