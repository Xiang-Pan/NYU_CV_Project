'''
Author: Xiang Pan
Date: 2021-12-13 00:35:59
LastEditTime: 2021-12-14 16:53:26
LastEditors: Xiang Pan
Description: 
FilePath: /project/task_models/seg_model_back.py
@email: xiangpan@nyu.edu
'''
import torch
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
import torchvision.transforms as transforms
from task_datasets.kitti_dataset import semantic_dataset
from task_models.unet import UNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from task_models.enet.model import ENet
from tools import mIOU
import torch.nn as nn


# class SenmanticSegmentationModel(pl.LightningModule):
    

class SegModel(pl.LightningModule):
    """Semantic Segmentation Module.
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    >>> from pl_examples import _DATASETS_PATH
    >>> dataset_path = os.path.join(_DATASETS_PATH, "Kitti")
    >>> _create_synth_kitti_dataset(dataset_path, image_dims=(1024, 512))
    >>> SegModel(dataset_path)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    SegModel(
      (net): UNet(
        (layers): ModuleList(
          (0): DoubleConv(...)
          (1): Down(...)
          (2): Down(...)
          (3): Up(...)
          (4): Up(...)
          (5): Conv2d(64, 19, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    """

    def __init__(
        self,
        task_name = None,
        load_checkpoint_path = None,
        # data_path: str = None,
        # batch_size: int = 4,
        # lr: float = 1e-3,
        # num_layers: int = 3,
        # features_start: int = 64,
        # bilinear: bool = False,
    ):
        super().__init__()
        num_classes = 19
        self.num_classes = 19
        # self.iou_train = JaccardIndex(num_classes)
        # self.iou_val = JaccardIndex(num_classes)
        # self.iou_test = JaccardIndex(num_classes)
        # self.iou_train = Iou(num_classes)
        # self.iou_val = Iou(num_classes)
        # self.iou_test = Iou(num_classes)
        

        # self.loss_measure = smp.utils.losses.DiceLoss(mode="multiclass")
        self.loss_measure = nn.CrossEntropyLoss(ignore_index=250)
        # self.loss_measure = smp.utils.losses.SoftCrossEntropyLoss()
        # self.iou_measure = smp.utils.metrics.IoU(num_classes)

        self.lr = 1e-3
        self.batch_size = 4
        # self.batch_size = batch_size
        # self.lr = lr
        # self.num_layers = num_layers
        # self.features_start = features_start
        # self.bilinear = bilinear

        # self.net = UNet(num_classes=19, num_layers=4, features_start=128, bilinear=self.bilinear)
        self.net = ENet(num_classes = 19)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.35675976, 0.37380189, 0.3764753], std=[0.32064945, 0.32098866, 0.32325324]
                ),
            ]
        )
        self.trainset = semantic_dataset(split = 'train', transform = self.transform)
        self.testset = semantic_dataset(split = 'test', transform = self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        outputs = self.forward(img)
        # train_loss = self.loss_measure(outputs, mask)
        train_loss = F.cross_entropy(outputs, mask, ignore_index=250)
        mIOU_score = mIOU(pred=outputs, label=mask)
        self.log("train/loss", train_loss, on_step=True, on_epoch=True)
        self.log("train/mIOU", mIOU_score, on_step=True, on_epoch=True)
        
        return None
    
    def training_epoch_end(self, outputs) -> None:
        self.log("train/lr", self.lr, on_step=False, on_epoch=True)
    #     metrics_avg = self.iou_train.compute()
    #     self.log("train/mIoU", metrics_avg.miou)
    #     self.iou_train.reset()

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        outputs = self.net(img)
        out = outputs.squeeze()
        mIOU_score = mIOU(pred=outputs, label=mask)
        val_loss = self.loss_measure(out, mask)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        self.log("val/mIOU", mIOU_score, on_step=True, on_epoch=True)
        return None

    # def validation_epoch_end(self, outputs): 
    #     metrics_avg = self.iou_val.compute()
    #     self.log("val/mIoU", metrics_avg.miou)
    #     self.iou_val.reset()
    #     loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     log_dict = {"val_loss": loss_val}
    #     return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    # def test_step(self, batch, batch_idx):
    #     inputs, labels = batch
    #     outputs = self.net(inputs)
    #     pred_mask = outputs
    #     mIOU = mIoU(pred=pred_mask, label=labels)

    #     test_loss = self.loss_measure(outputs, labels)
    #     self.log("test/mIOU", mIOU, on_step=True, on_epoch=True)
    #     self.log("test/loss", test_loss)
    #     return None

    # def test_epoch_end(self, outputs: List[Any]):
    #     # Compute and log metrics across epoch
    #     # metrics_avg = self.iou_test.compute()
    #     self.log("test/mIoU", metrics_avg.miou)
    #     self.log("test/accuracy", metrics_avg.accuracy.mean())
    #     self.log("test/precision", metrics_avg.precision.mean())
    #     self.log("test/recall", metrics_avg.recall.mean())

    #     # Save test results as a Table (WandB)
    #     self.log_results_table_wandb(metrics_avg)
    #     self.iou_test.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
        # return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)

    # @staticmethod
    # def add_model_specific_args(parent_parser):  # pragma: no-cover
    #     parser = parent_parser.add_argument_group("SegModel")
    #     parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    #     parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    #     parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    #     parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
    #     parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
    #     parser.add_argument(
    #         "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
    #     )
    #     return parent_parser
