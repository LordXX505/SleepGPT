import os
import sys

sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone import Model
import pytorch_lightning as pl
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
import torch
import pyarrow as pa
import pandas as pd
@ex.automain
def main(_config):
    # pre_train = Model_Pre(_config)

    pre_train = Model(_config)
    print(_config)
    pl.seed_everything(512)
    dm = MultiDataModule(_config, kfold=0)
    dm.setup(stage='train')
    pre_train.training = False
    pre_train.mask_same = True
    # pre_train.eval()
    for batch in dm.train_dataloader():
        pre_train.set_task()
        sp_batch = []
        for _batch in batch:
            sp_label = _batch['Spindle_label']
            print(f'sp_label.shape: {sp_label.shape}')
            if torch.sum(sp_label) >= 100*0.25:
                sp_batch.append(_batch)
                print(f"name: {_batch['name']}")
        if len(sp_batch) == 0:
            continue
        res = pre_train(sp_batch, stage='test')



