'''
Author: Xiang Pan
Date: 2021-11-10 19:57:30
LastEditTime: 2021-12-14 20:40:12
LastEditors: Xiang Pan
Description: 
FilePath: /project/main.py
@email: xiangpan@nyu.edu
'''

import os
import random
from typing import Any, List
from argparse import ArgumentParser, Namespace
from option import *

from tools import *
from metrics import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import wandb

from task_models.seg_model import SegModel
# from task_models.seg_model_back import SegModel


def main(hparams: Namespace):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(
        task_name=hparams.task_name,
        load_checkpoint_path=hparams.load_checkpoint_path,
        # data_path=hparams.data_path,
        # batch_size=hparams.batch_size,
        # lr=hparams.lr,
        # num_layers=hparams.num_layers,
        # features_start=hparams.features_start,
        # bilinear=hparams.bilinear,
    )

    # ------------------------
    # 2 SET LOGGER
    # ------------------------
    # run = wandb.init()
    # run.tags += tuple(hparams.task_name)
    if hparams.log_name is None:
        hparams.log_name = hparams.task_name
    wandb_logger = WandbLogger(project="NYU_CV_Project", 
                               name = hparams.log_name, 
                               notes=hparams.log_name,
                               save_dir="./outputs/wandb",
                               tags=[hparams.task_name])

    # optional: log model topology
    wandb_logger.watch(model.net)

    # save any arbitrary metrics like `val_loss`, etc. in name
    # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
    # save any arbitrary metrics like `val_loss`, etc. in name
    # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
    checkpoint_callback = ModelCheckpoint(
        dirpath='./outputs/'+hparams.log_name,
        every_n_epochs=1,
        save_top_k=-1,
        filename='{epoch}-{val/loss:.2f}-{val/mIOU:.2f}',
    )
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpu,
        weights_save_path="./outputs",
        callbacks=[checkpoint_callback]
    )
    # trainer = Trainer(logger=wandb_logger, gpus=option.gpu, max_epochs=option.max_epochs, callbacks=[checkpoint_callback])

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    # cli_lightning_logo()
    # parser = ArgumentParser(add_help=True)
    # parser = Trainer.add_argparse_args(parser)
    # parser = SegModel.add_model_specific_args(parser)
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    # parser = SegModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)




# def _create_synth_kitti_dataset(path_dir: str, image_dims: tuple = (1024, 512)):
#     """Create synthetic dataset with random images, just to simulate that the dataset have been already
#     downloaded."""
#     path_dir_images = os.path.join(path_dir, KITTI.IMAGE_PATH)
#     path_dir_masks = os.path.join(path_dir, KITTI.MASK_PATH)
#     for p_dir in (path_dir_images, path_dir_masks):
#         os.makedirs(p_dir, exist_ok=True)
#     for i in range(3):
#         path_img = os.path.join(path_dir_images, f"dummy_kitti_{i}.png")
#         Image.new("RGB", image_dims).save(path_img)
#         path_mask = os.path.join(path_dir_masks, f"dummy_kitti_{i}.png")
#         Image.new("L", image_dims).save(path_mask)