# Code adapted from https://github.com/cloudsen12/models/blob/master/unet_mobilenetv2/cloudsen12_unet.py
import warnings
from pathlib import Path
from typing import Optional

import albumentations as A
import albumentations.pytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning
import pytorch_lightning as pl
import rasterio as rio
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import fbeta_score, precision_score, recall_score
from torchmetrics import Metric

# load the dataset
dataset_metadata = pd.read_csv("cloudSEN12/cloudsen12_metadata.csv")
dataset_metadata_high = dataset_metadata[dataset_metadata["label_type"] == "high"]

# train/val/test split
train_val_db = dataset_metadata_high[dataset_metadata_high["test"] == 0]
train_val_db.reset_index(drop=True, inplace=True)

# train dataset
train_db = train_val_db.sample(frac=0.9, random_state=42)
train_db.reset_index(drop=True, inplace=True)


# val dataset
val_db = train_val_db.drop(train_db.index)
val_db.reset_index(drop=True, inplace=True)

# test dataset
test_db = dataset_metadata_high[dataset_metadata_high["test"] == 1]
test_db.reset_index(drop=True, inplace=True)

DATASET = [train_db, val_db, test_db]


# Non destructive transformations - Dehidral group D4
nodestructive_pipe = A.OneOf(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
    ],
    p=1,
)

weak_augmentation = A.Compose(
    [
        A.PadIfNeeded(min_height=512, min_width=512, p=1, always_apply=True),
        nodestructive_pipe,
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

no_augmentation = A.Compose(
    [
        A.PadIfNeeded(min_height=512, min_width=512, p=1, always_apply=True),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

AUGMENTATION = weak_augmentation


# Create a DataLoader object.
class SEGDATALOADER(torch.utils.data.DataLoader):
    def __init__(self, dataset, augmentation=False):
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Select the S2 and ROI id
        roi_id = self.dataset.loc[index, "roi_id"]
        s2_id = self.dataset.loc[index, "s2_id_gee"]

        # Load the numpy file
        s2l1c = f"cloudSEN12/high/%s/%s/S2L2A.tif" % (roi_id, s2_id)
        with rio.open(s2l1c) as src:
            X = (
                src.read([2, 3, 4, 9]) / 10000
            )  # ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT', 'WVP', 'TCI_R', 'TCI_G', 'TCI_B')
            # X = X[1:4,...]
            X = np.moveaxis(X, 0, -1)
            # X = np.moveaxis(X, -1, 0)

        # Load target image.
        target = Path(
            f"cloudSEN12/manual_hq/{roi_id}__{s2_id}.tif"
        )  # f"cloudSEN12/high/%s/%s/labels/manual_hq.tif" % (roi_id, s2_id)
        if not target.is_file():
            target = f"cloudSEN12/high/%s/%s/labels/manual_hq.tif" % (roi_id, s2_id)
        with rio.open(target) as src:
            y = src.read(1)

        # Augmentation pipeline
        if self.augmentation:
            X, y = self.augmentation(image=X, mask=y).values()
            # X = np.moveaxis(X, -1, 0)

        # Check semantic_segmentation_pytorch model input shape requirements.
        if X.shape[0] > X.shape[2]:
            warnings.warn(
                "segmentation_models.pytorch expects channels first (B, C, H, W)"
            )
        return X, y, "%s__%s" % (roi_id, s2_id)


SEGMODEL = smp.Unet(
    encoder_name="mobilenet_v2", encoder_weights=None, classes=4, in_channels=4
)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # flatten label and prediction tensors
        # input = input.view(-1)
        target = target.type(torch.long)
        BCE = torch.nn.functional.cross_entropy(input, target)
        return BCE


CRITERION = CrossEntropyLoss()


class BF2score(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, thershold: float = 0.90):
        super().__init__()
        self.add_state("container", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thershold = thershold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score_container = list()
        for index in range(preds.shape[0]):
            score_container.append(
                fbeta_score(
                    target[index].flatten().detach().cpu(),
                    preds[index].flatten().detach().cpu(),
                    average="macro",
                    beta=2,
                    zero_division=1,
                )
            )
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container / self.total * 100


class BPAscore(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, thershold: float = 0.90):
        super().__init__()
        self.add_state("container", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thershold = thershold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score_container = list()
        for index in range(preds.shape[0]):
            score_container.append(
                recall_score(
                    target[index].flatten().detach().cpu(),
                    preds[index].flatten().detach().cpu(),
                    average="macro",
                    zero_division=1,
                )
            )
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container / self.total * 100


class BUAscore(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, thershold: float = 0.90):
        super().__init__()
        self.add_state("container", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thershold = thershold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score_container = list()
        for index in range(preds.shape[0]):
            score_container.append(
                precision_score(
                    target[index].flatten().detach().cpu(),
                    preds[index].flatten().detach().cpu(),
                    average="macro",
                    zero_division=1,
                )
            )
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container / self.total * 100


METRICS = {"f2_score": BF2score(), "pa_score": BPAscore(), "ua_score": BUAscore()}


class litSegModel(pl.LightningModule):
    """
    Lightning Class template to wrap segmentation models.
    Args:
      hparams (`DictConfig`) : A `DictConfig` that stores the configs for training .
    """

    def __init__(self, batch_size=2):
        super().__init__()
        # self.save_hyperparameters()  # Save the hyperparameters.
        self.model = SEGMODEL
        self.dataloader = SEGDATALOADER
        self.criterion = CRITERION
        self.metrics = METRICS
        self.dataset = DATASET
        self.augmentation = AUGMENTATION
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """
        Change the file utils/prepare_data.py to this function. It must return
        SegDataset and SegDataLoader.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # train/val/test split
        train, val, test = self.dataset
        if stage in (None, "fit"):
            self.dbtrain = self.dataloader(train, AUGMENTATION)
            self.dbval = self.dataloader(val, no_augmentation)

        if stage in (None, "test"):
            self.dbtest = self.dataloader(test, no_augmentation)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dbtrain,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dbval,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dbtest,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y, _ = batch
        y_hat = self.forward(X)
        # save_breakpoint([y_hat, y])
        loss = self.criterion(y_hat, y)
        self.log("loss_train", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, _ = batch
        y_hat = self.forward(X)
        # save_breakpoint([y_hat, y])
        loss = self.criterion(y_hat, y)
        self.log("loss_val", loss, prog_bar=True, logger=True, on_epoch=True)

        # Update metrics
        if self.metrics is not None:
            y_hat_class = y_hat.argmax(dim=1)
            y = y.type(torch.long)

            # Iterate for each metric
            for value in self.metrics.values():
                value.update(y_hat_class, y)

        return loss

    def validation_epoch_end(self, val_metrics_results):
        if self.metrics is not None:
            for key, value in self.metrics.items():
                metric_value = value.compute()
                logging_name = key.lower() + "_val"
                self.log(
                    name=logging_name,
                    value=metric_value,
                    prog_bar=False,
                    logger=True,
                    on_epoch=True,
                )
                value.reset()

    def test_step(self, batch, batch_idx):
        if self.metrics is not None:
            # Update metrics
            X, y, _ = batch
            y_hat = self.forward(X).squeeze()

            y_hat_class = y_hat.argmax(dim=1)
            y = y.type(torch.long)

            # Iterate for each metric
            for value in self.metrics.values():
                value.update(y_hat_class, y)

    def test_epoch_end(self, outputs):
        if self.metrics is not None:
            for key, value in self.metrics.items():
                metric_value = value.compute()
                logging_name = key.lower() + "_test"
                self.log(
                    name=logging_name,
                    value=metric_value,
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                )
                value.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimier to use   for training

        Returns:
            torch.optim.Optimier: the optimizer for updating the model's parameters
        """
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.001)

        # Set a scheduler
        self.sch = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.opt, mode="min", factor=0.1, patience=4, verbose=True
            ),
            "frequency": 1,  #
            "monitor": "loss_val",  # quantity to be monitored
        }
        return [self.opt], [self.sch]


def main():
    mymodel = litSegModel(batch_size=32)
    callbacks = [
        pytorch_lightning.callbacks.EarlyStopping(
            monitor="loss_val", patience=10, mode="min"
        ),
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor="loss_val", save_top_k=1, save_last=True, mode="min"
        ),
    ]
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="logs/", name="cloudmask_L2Argbnir"
    )
    trainer = Trainer(
        gpus=[
            0,
        ],
        max_epochs=100,
        precision=16,
        callbacks=callbacks,
        logger=tb_logger,
    )  # , logger=LOGGER)
    # start train
    trainer.fit(mymodel)

    # start test
    trainer.test(mymodel)


if __name__ == "__main__":
    main()
