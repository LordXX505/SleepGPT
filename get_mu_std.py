import sys


import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
sys.path.append('../../main')
from main import MultiDataModule
from main.config import ex
from torch.utils.data.distributed import DistributedSampler

@ex.automain
def main(_config):
    all_channels = ['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5'
        , 'CP6', 'Cz', 'ECG', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'F1', 'F2', 'F3', 'F4', 'F5'
        , 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fpz', 'Fz'
        , 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'POz'
        , 'Pz', 'T7', 'T8', 'TP7', 'TP8']
    choose = [4, 5, 15, 16, 18, 22, 23, 36, 38, 39, 52]
    print(np.array(all_channels)[choose])