import os
from multiprocessing import Process, current_process
from multiprocessing import Pool
from tqdm import tqdm
import pyarrow as pa
import gc
import pandas as pd
import numpy as np
import mne
from threading import Thread
import multiprocessing
import time
import glob as glob
import torch
import torch.nn as nn
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
def get_mean_std_value(loader):
    channels_sum,channel_squared_sum,num_batches = 0,0,0

    for _data in tqdm(loader):
        data = _data['epochs'][0]
        if num_batches==1:
            print(channels_sum.shape)
        print(f"******num_batches: {num_batches}******")
        channels_sum += torch.mean(data,dim=[0,2])#shape [n_samples(batch),channels,seq]
        channel_squared_sum += torch.mean(data**2,dim=[0,2])#shape [n_samples(batch),channels,seq]
        num_batches +=1

    mean = channels_sum/num_batches

    std = (channel_squared_sum/num_batches - mean**2)**0.5
    return mean,std

@ex.automain
def main(_config):
    dm = MultiDataModule(_config)
    print(_config)
    dm.setup(stage='train')
    mean, std = get_mean_std_value(dm.train_dataloader())
    print(mean, std)
