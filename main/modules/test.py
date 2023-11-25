import time
from time import sleep
from typing import Any, Optional

import lightning.pytorch as pl
import torch.nn as nn
import torch
from .get_optm import *
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
import torch.distributed as dist
import os
class Test(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))
        self.loss = nn.CrossEntropyLoss()
        # self.ckpt = os.path.join(config['output_dir'], "sleep_seed1_from_", "version_0", "checkpoints", "epoch=9-step=2150.ckpt")
        # ckpt = torch.load(self.ckpt, map_location='cpu')
        # print(ckpt)

    def forward(self, x) -> Any:
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        tensor = torch.tensor(1).to(self.device)
        print("rank: ", dist.get_rank(), "step: ", self.global_step, "loss1: ", loss)

        if dist.is_initialized():
            # print("step")
            # print("rank: ", dist.get_rank(), "step: ", self.global_step, "time: ", time.time())
            # for name, parameters in self.named_parameters():
            #     print(name, parameters)
            if dist.get_rank() == 0:
                sleep(30)
            # print(tensor)
            self.log('**step-log: ', self.global_step, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
            self.log('**rank-log: ', self.global_rank, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
            # dist.all_reduce(tensor)
            # print(tensor)

        self.log('training_step_y_hat', y_hat.mean())

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('batch_idx', batch_idx, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "test:": "test"}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if dist.is_initialized():
            print("rank: ", dist.get_rank())
        print("on_train_batch_end: ", outputs)

    def on_train_epoch_end(self) -> None:
        print("on_train_epoch_end")
        print(self.lr_schedulers().get_lr(), self.lr_schedulers().last_epoch)

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)

    def configure_optimizers(self):
        return set_schedule(self)
