import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from lightning.pytorch.cli import LRSchedulerTypeUnion
from lightning.pytorch.utilities.types import STEP_OUTPUT
from . import get_optm
from main.utils import init_weights, set_metrics
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scipy import interpolate
from timm.models import create_model
from . import heads
from typing import Any, Optional
from . import objectives
from . import multiway_transformer
from lightning import LightningModule

import pynvml


class Mu_Std(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.mu_list = []
        self.mu_2_list = []

    def forward(self, batch: Any) -> Any:
        epochs = batch['epochs']
        mu_res = torch.mean(epochs)
        mu_2_res = torch.mean(np.square(epochs))
        return mu_res, mu_2_res

    def training_step(self, batch) -> STEP_OUTPUT:
        res = self(batch)
        self.mu_list.append(res[0])
        self.mu_2_list.append(res[1])
        return res

    def on_train_end(self) -> None:
        print(torch.mean(torch.tensor(self.mu_list).item()))
        print(torch.mean(torch.tensor(self.mu_2_list).item()))

    def configure_optimizers(self):
        pass
