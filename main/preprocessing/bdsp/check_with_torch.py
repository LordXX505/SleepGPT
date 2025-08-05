import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

sys.path.append('/root/Sleep')

from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from main.modules.backbone import Model
import pytorch_lightning as pl
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
from main.modules.mixup import Mixup
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0',
             '#e67e22', '#B883D4', '#9E9E9E']
    return color[:nums]

def get_names():
    return ['C3', 'C4', 'EMG', 'EOG1', 'F3', 'Fpz', 'O1', 'Pz']

@ex.automain
def main(_config):
    print(_config)
    pl.seed_everything(512)
    dm = MultiDataModule(_config, kfold=0)
    dm.setup(stage='train')

    # Debug: Check the type and content of self.dms
    print(f"Type of self.dms: {type(dm.dms)}")
    for i, d in enumerate(dm.dms):
        print(f"Type of dm.dms[{i}]: {type(d)}")

    try:
        # Ensure all elements in self.dms are datasets
        test_datasets = [d.test_dataset for d in dm.dms if hasattr(d, 'test_dataset')]
        val_datasets = [d.val_dataset for d in dm.dms if hasattr(d, 'val_dataset')]
        train_datasets = [d.train_dataset for d in dm.dms if hasattr(d, 'train_dataset')]

        print(f"Number of test datasets: {len(test_datasets)}")
        print(f"Number of val datasets: {len(val_datasets)}")
        print(f"Number of train datasets: {len(train_datasets)}")

        dm.test_dataset = ConcatDataset(test_datasets)
        dm.val_dataset = ConcatDataset(val_datasets)
        dm.train_dataset = ConcatDataset(train_datasets)
    except TypeError as e:
        print(f"TypeError: {e}")
        sys.exit(1)

    num_workers = 128  # Set to the number of CPU cores
    batch_size = 1024  # Adjust based on your memory and requirement

    test_loader = DataLoader(dm.test_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dm.val_dataset, batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(dm.train_dataset, batch_size=batch_size, num_workers=num_workers)

    test_error_files = []
    val_error_files = []
    train_error_files = []

    for batch in tqdm(test_loader, desc="Processing test data"):
        for b in batch['err']:
            if b!='good':
                print(b)
            test_error_files.append(b)

    # np.save('./test_error', test_error_files)

    for batch in tqdm(val_loader, desc="Processing validation data"):
        for b in batch['err']:
            if b != 'good':
                print(b)
            val_error_files.append(b)

    # np.save('./val_error', val_error_files)

    for batch in tqdm(train_loader, desc="Processing train data"):
        for b in batch['err']:
            if b != 'good':
                print(b)
            train_error_files.append(b)

    # np.save('./train_error', train_error_files)