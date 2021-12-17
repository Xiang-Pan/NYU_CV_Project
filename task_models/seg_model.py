'''
Author: Xiang Pan
Date: 2021-12-14 00:55:53
LastEditTime: 2021-12-14 19:32:54
LastEditors: Xiang Pan
Description: 
FilePath: /project/task_models/seg_model.py
@email: xiangpan@nyu.edu
'''
import os
# from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
import torchvision.transforms as transforms
# import torchvision
# import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
from task_models.enet.model import ENet
from task_datasets.kitti_dataset import semantic_dataset
from tools import mIOU

class SegModel(pl.LightningModule):    
    def __init__(self, task_name="KITTI", load_checkpoint_path=None):
        super(SegModel, self).__init__()
        self.task_name = task_name
        self.batch_size = 4
        self.learning_rate = 1e-3
        if self.task_name == "KITTI":
            self.num_classes = 19
        else:
            self.num_classes = 34
        self.net = ENet(num_classes = self.num_classes)

        self.trainset = semantic_dataset(task=self.task_name, split='train')
        self.valset = semantic_dataset(task=self.task_name, split='val')
        self.testset = semantic_dataset(task=self.task_name, split='test')
        self.loss_measure = nn.CrossEntropyLoss(ignore_index=250)

        if load_checkpoint_path is not None:
            if "Cityscapes" in load_checkpoint_path:
                from task_models.enet_citiscape import ENet_Cityscapes
                state_dict = torch.load("./cached_models/ENet_Cityscapes/ENet")['state_dict']
                del state_dict['transposed_conv.weight']
                self.net = ENet_Cityscapes(num_classes=19)
                self.net.load_state_dict(state_dict, strict=False)
            else:
                d = torch.load(load_checkpoint_path)['state_dict']
                del d["net.transposed_conv.weight"]
                del d["net.transposed_conv.bias"]
                self.load_state_dict(d, strict=False)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        # pred_mask = out.softmax(dim=1).argmax(dim=1)
        mIOU_score = mIOU(pred=out, label=mask)
        
        train_loss = self.loss_measure(out, mask)
        train_loss = F.cross_entropy(out, mask, ignore_index=250)
        
        self.log("train/loss", train_loss, on_step=True, on_epoch=True)
        self.log("train/mIOU", mIOU_score, on_step=True, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True)
        return {'loss' : train_loss}
    
    # def 
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.net(img)
        mIOU_score = mIOU(pred=out, label=mask)
        val_loss = F.cross_entropy(out, mask, ignore_index=250)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        self.log("val/mIOU", mIOU_score, on_step=True, on_epoch=True)
        return None
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        # scheduler = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size = 1, shuffle = True)