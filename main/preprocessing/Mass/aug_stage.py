import mne
import numpy as np
import pyarrow as pa
import os
import glob
import pandas as pd
import gc
import torch
import os
import sys
import matplotlib.patches as patches

sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone import Model
from main.transforms import unnormalize
import pytorch_lightning as pl
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
import torch
import re


def get_epochs(data):
    try:
        x = np.array(data.as_py())
    except:
        x = np.array(data.to_pylist())
    x = x * 1e6
    x = torch.from_numpy(x).float()
    return {'x': x}


def get_stage(data):
    return {'Stage_label': torch.from_numpy(np.array(data)).long()}


def random_plot_regenerate(origin, new, cnt, mode, sub):
    if torch.rand(1) > 0.001:
        return
    fig = plt.figure(figsize=(20, 20))

    for c in range(8):
        axes = fig.add_subplot(8, 2, c * 2 + 1)
        axes.plot(origin[c])

    for c in range(8):
        axes = fig.add_subplot(8, 2, (c + 1) * 2)
        axes.plot(new[c])

    os.makedirs(f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{sub.split('/')[-1]}/MASS2/epoch_{cnt}",
                exist_ok=True)
    try:
        plt.savefig(
            f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{sub.split('/')[-1]}/MASS2/epoch_{cnt}/predict_{mode}.svg",
            format='svg')
    except:
        print('save figure failed')
    plt.show()
    plt.close('all')

def save_epoch(save_epochs, save_labels, filename, name, cnt):
    dataframe = pd.DataFrame(
        {'x': [save_epochs.tolist()], 'stage': save_labels}
    )
    table = pa.Table.from_pandas(dataframe)
    os.makedirs(f"{filename}/{name}/", exist_ok=True)
    print(f"save path: {filename}/{name}/{str(cnt).zfill(5)}.arrow, stage:{save_labels}")
    with pa.OSFile(
            f"{filename}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    del dataframe
    del table
def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        raise NotImplementedError


def clone_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.clone()
    elif isinstance(batch, dict):
        return {k: clone_batch(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [clone_batch(v) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(clone_batch(v) for v in batch)
    else:
        raise NotImplementedError

@ex.automain
def main(_config):
    pre_train = Model(_config)
    print(_config)
    pl.seed_everything(512)
    pre_train.set_task()
    pre_train.mask_same = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_train.to(device)
    dm = MultiDataModule(_config, kfold=0)
    dm.setup(stage='test')
    collate = dm.collate
    path = '/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2'
    batch_size = 256
    edf_items = sorted(glob.glob(os.path.join(path, '*')))
    partition = int(len(edf_items)/2) + 1
    partition_idx = 0
    resume_flag = False
    for edf_items_index, sub in enumerate(edf_items[(partition_idx*partition):((partition_idx+1)*partition)]):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            sub_arr_items = sorted(glob.glob(os.path.join(sub, '*')))
            for start_index in range(0, len(sub_arr_items), batch_size):
                batch_list = []
                label_list = []
                for sub_arr_index, sub_arr in enumerate(sub_arr_items[start_index:start_index + batch_size]):
                    print(f'===========begin: {base_name}===========cnt:{start_index+sub_arr_index}==========={sub_arr}=========')
                    tables = pa.ipc.RecordBatchFileReader(
                        pa.memory_map(sub_arr, "r")
                    ).read_all()
                    epoch = get_epochs(tables['x'][0])['x']
                    stage = get_stage(tables['stage'])['Stage_label']
                    label_list.append(stage)
                    batch_orig = {'stage': stage.detach().clone().unsqueeze(0),
                                  'x': (epoch, torch.tensor([4, 5, 16, 18, 22, 36, 38, 52])),
                                  'index': torch.tensor(1),
                                  'need_norm': True}  # index is no use
                    batch_list.append(batch_orig)
                batch_cpu = collate(batch_list)
                batch_cuda = move_to_device(batch_cpu, device)
                batch = clone_batch(batch_cuda)
                pre_train.first_log_gpu = True
                with torch.no_grad():
                    res = pre_train(batch, stage='test')
                generate_epoch = res['cls_feats'] * res['time_mask_patch'].unsqueeze(-1)
                generate_epoch = pre_train.unpatchify(generate_epoch).detach().clone()
                origin = batch['epochs'][0].detach().clone()
                for sub_arr_index, sub_arr in enumerate(sub_arr_items[start_index:start_index + batch_size]):
                    random_plot_regenerate(origin[sub_arr_index].detach().cpu().numpy(),
                                           generate_epoch[sub_arr_index].detach().cpu().numpy(),
                                           sub=base_name, cnt=sub_arr_index, mode='Random')
                    save_epoch(generate_epoch[sub_arr_index].detach().cpu().numpy()+origin[sub_arr_index].detach().cpu().numpy(),
                               label_list[sub_arr_index].detach().cpu().numpy()[0],
                               filename=os.path.join('/', *path.split('/')[:-1], 'Aug_Random'), name=base_name
                               , cnt=sub_arr_index+start_index)

    print('------------------all finished------------------')
