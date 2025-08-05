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

    os.makedirs(f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{sub.split('/')[-1]}/EDF/epoch_{cnt}",
                exist_ok=True)
    try:
        plt.savefig(
            f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{sub.split('/')[-1]}/EDF/epoch_{cnt}/predict_{mode}.svg",
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
    path = '/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed'
    batch_size = 256
    edf_items = sorted(glob.glob(os.path.join(path, '*')))
    partition = int(len(edf_items)/2) + 1
    partition_idx = 0
    resume_flag = False
    for edf_items_index, sub in enumerate(edf_items[(partition_idx*partition):((partition_idx+1)*partition)]):
        if os.path.isdir(sub):
            # if torch.rand(1)<0.5:
            #     print(f'edf_items skipping: {sub}')
            #     continue
            base_name = os.path.basename(sub)
            # if base_name == 'SC4191E0':
            #     resume_flag = True
            # if resume_flag is False:
            #     continue
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
                    # if stage[0] == 0:
                    #     continue
                    batch_orig = {'stage': stage.detach().clone().unsqueeze(0),
                                  'x': (epoch, torch.tensor([16, 18, 36, 52])),
                                  'index': torch.tensor(1)}  # index is no use
                    batch_list.append(batch_orig)
                batch_cpu = collate(batch_list)
                batch_cuda = move_to_device(batch_cpu, device)

                batch = clone_batch(batch_cuda)
                print('augment channels')

                pre_train.first_log_gpu = True
                # batch['mask'][0][:, 2] = 0
                batch['mask'][0][:, :2] = 1
                batch['mask'][0][:, 2] = 0
                batch['mask'][0][:, 4] = 1
                batch['mask'][0][:, 6] = 1
                batch['random_mask'][0][:] = torch.zeros(120)
                batch['random_mask'][0][:, 0:30] = torch.ones(30)
                batch['random_mask'][0][:, 60:75] = torch.ones(15)
                batch['random_mask'][0][:, 90:105] = torch.ones(15)

                with torch.no_grad():
                    res = pre_train(batch, stage='test')
                generate_epoch_f3 = res['cls_feats'] * res['time_mask_patch'].unsqueeze(-1)
                generate_epoch_f3 = pre_train.unpatchify(generate_epoch_f3).detach().clone()
                origin_f3 = batch['epochs'][0].detach().clone()

                # # c4, index=1, patch_position = 1*15:2*15
                # print('augment c4 channel')
                # batch = clone_batch(batch_cuda)

                # batch['mask'][0][:, 2] = 0
                # batch['mask'][0][:, 1] = 1
                # batch['random_mask'][0][:] = torch.zeros(120)
                # batch['random_mask'][0][:, 15:30] = torch.ones(15)
                # # batch['random_mask'][0][0][75:90] = torch.ones(15)
                # with torch.no_grad():
                #     res = pre_train(batch, stage='test')
                # generate_epoch_c4 = res['cls_feats'] * res['time_mask_patch'].unsqueeze(-1)
                # generate_epoch_c4 = pre_train.unpatchify(generate_epoch_c4).detach().clone()
                # origin_c4 = batch['epochs'][0].detach().clone()
                #
                # # c4 and f3
                # print('augment c4 and f3 channel')
                # batch = clone_batch(batch_cuda)
                #
                # batch['mask'][0][:, 2] = 0
                # batch['mask'][0][:, 1] = 1
                # batch['mask'][0][:, 4] = 1
                # batch['random_mask'][0][:] = torch.zeros(120)
                # batch['random_mask'][0][:, 15:30] = torch.ones(15)
                # batch['random_mask'][0][:, 60:75] = torch.ones(15)
                # with torch.no_grad():
                #     res = pre_train(batch, stage='test')
                # generate_epoch_f3_c4 = res['cls_feats'] * res['time_mask_patch'].unsqueeze(-1)
                # generate_epoch_f3_c4 = pre_train.unpatchify(generate_epoch_f3_c4).detach().clone()
                # origin_f3_c4 = batch['epochs'][0].detach().clone()

                for sub_arr_index, sub_arr in enumerate(sub_arr_items[start_index:start_index + batch_size]):
                    random_plot_regenerate(origin_f3[sub_arr_index].detach().cpu().numpy(), generate_epoch_f3[sub_arr_index].detach().cpu().numpy(), sub=base_name, cnt=sub_arr_index, mode='Aug_All')
                    save_epoch(generate_epoch_f3[sub_arr_index].detach().cpu().numpy()+origin_f3[sub_arr_index].detach().cpu().numpy(), label_list[sub_arr_index].detach().cpu().numpy()[0],
                               filename=os.path.join('/', *path.split('/')[:-1], 'Aug_All'), name=base_name
                               , cnt=sub_arr_index+start_index)

                    # random_plot_regenerate(origin_c4[sub_arr_index].detach().cpu().numpy(), generate_epoch_c4[sub_arr_index].detach().cpu().numpy(), sub=base_name, cnt=sub_arr_index,
                    #                        mode='C4')
                    # save_epoch(generate_epoch_c4[sub_arr_index].detach().cpu().numpy()+origin_c4[sub_arr_index].detach().cpu().numpy(), label_list[sub_arr_index].detach().numpy()[0],
                    #            filename=os.path.join('/', *path.split('/')[:-1], 'Aug_C4'), name=base_name
                    #            , cnt=sub_arr_index + start_index)
                    #
                    # random_plot_regenerate(origin_f3_c4[sub_arr_index].detach().cpu().numpy(), generate_epoch_f3_c4[sub_arr_index].detach().cpu().numpy(), sub=base_name, cnt=sub_arr_index,
                    #                        mode='C4andF3')
                    # save_epoch(generate_epoch_f3_c4[sub_arr_index].detach().cpu().numpy()+origin_f3_c4[sub_arr_index].cpu().detach().numpy(), label_list[sub_arr_index].detach().numpy()[0],
                    #            filename=os.path.join('/', *path.split('/')[:-1], 'Aug_F3_C4'), name=base_name
                    #            , cnt=sub_arr_index + start_index)

    print('------------------all finished------------------')
