import sys

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
#
# path_root = "/home/cuizaixu_lab/huangweixuan/data/data/SS2"
# # path_root = '/Volumes/T7 Shield/SS2'
# ana_spindle = os.path.join(path_root, 'SS2_ana')
# spindle_path_E1 = glob.glob(ana_spindle + '/*Spindles_E1*')
# spindle_path_E2 = glob.glob(ana_spindle + '/*Spindles_E2*')
# spindle_path_E1 = sorted(spindle_path_E1)
# spindle_path_E2 = sorted(spindle_path_E2)
# epoch = os.path.join(path_root, 'SS2_bio')
# def plot_generate(generate_epoch, res, batch_epoch, aug_time, i, save_label, inverse):
#     fig, Axes = plt.subplots(nrows=8, ncols=2, sharex='all', figsize=(30, 32))
#     for c in range(8):
#         Axes[c][0].plot(generate_epoch[c].detach().numpy())
#         Axes[c][0].plot(save_label*max(generate_epoch[c].detach().numpy()))
#         Axes[c][0].set_xticks(np.arange(0, 2000, 200))
#         for i in range(c * 15, (c + 1) * 15):
#             if res["time_mask_patch"][0][i] == 1:
#                 Axes[c][0].add_patch(
#                     patches.Rectangle(
#                         ((i % 15) * 200, 0),
#                         200,
#                         0.0001,
#                         edgecolor='red',
#                         fill=False
#                     )
#                 )
#         Axes[c][1].plot(batch_epoch[c].detach().numpy() / 1e6)
#     # plt.show()
#     os.makedirs(f'/home/cuizaixu_lab/huangweixuan/Sleep/result/{aug_time}/aug_{i}/', exist_ok=True)
#     plt.savefig(f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{aug_time}/aug_{i}/predict_{inverse}.svg", format='svg')
#
#
# @ex.automain
# def main(_config):
#     pre_train = Model(_config)
#     pre_train.mask_same = True
#     print(_config)
#     pl.seed_everything(512)
#     pre_train.set_task()
#     dm = MultiDataModule(_config, kfold=0)
#     dm.setup(stage='test')
#     collate = dm.collate
#     count_label = 0
#     count_stage2 = 0
#     aug_time = 2
#     MASS_aug_name = f"MASS_aug_new_{aug_time}"
#     print(f'MASS_aug_name: {MASS_aug_name}')
#     for _, path in enumerate([spindle_path_E1, spindle_path_E2]):
#         plot_cnt = 0
#         expert = 'E1'
#         if _ == 1:
#             expert = 'E2'
#         for items in path:
#             name = os.path.split(items)[1].split(' ')[0]
#             print(f"--------------{name}---------------")
#             # if name != '01-02-0019':
#             #     continue
#             epochs = mne.io.read_raw_edf(glob.glob(epoch + f"/{name}*PSG*")[0])
#             Base = mne.read_annotations(glob.glob(epoch + f"/{name}*Base*")[0])
#             print(f'base.info: {Base.description}')
#             anno = mne.read_annotations(items)
#             print(f"epochs: {epochs.info}")
#             epochs.load_data()
#             epochs = epochs.resample(100)
#             print(f"resample epochs: {epochs.info}")
#             epochs = epochs.filter(l_freq=0.3, h_freq=35, n_jobs='cuda', method='fir')
#
#             bads = epochs.info['bads']
#             badsidx = [epochs[_] for _ in bads]
#             badsidx = sorted(badsidx)
#             print(f'{epochs.info["bads"]}, idx: {badsidx}')
#             epochs.rename_channels(
#                 {'EEG C3-CLE': 'C3', 'EEG C4-CLE': 'C4', 'EEG F3-CLE': 'F3', 'EEG O1-CLE': 'O1', 'EEG Fpz-CLE': 'Fpz',
#                  'EMG Chin': 'EMG', 'EEG Pz-CLE': 'Pz', 'EOG Left Horiz': 'EOG'})
#             epochs.pick(['C3', 'C4', 'EMG', 'EOG', 'F3', 'Fpz', 'O1', 'Pz'])
#             labels = np.zeros(len(epochs))
#             choose_idx = {}
#             n_epochs = len(epochs) // 2000
#             for i in range(n_epochs):
#                 choose_idx[i] = 0
#             bucket_num = 0
#             for times in anno:
#                 # print(f'times: {times["onset"]}')
#                 begin_ind = epochs.time_as_index(times=times['onset'])[0]
#                 # print(f'times after: {begin_ind}')
#                 # print(f'end: {times["onset"] + times["duration"]}')
#                 end_ind = epochs.time_as_index(times=times['onset'] + times['duration'])[0]
#                 labels[begin_ind:end_ind] = 1
#             select_idx = []
#             for index, item in enumerate(Base):
#                 if item['description'] == 'Sleep stage 2':
#                     select_idx.append(index)
#             select_idx = np.array(select_idx)
#             count_stage2 += len(select_idx)
#             print(type(labels))
#             epochs = epochs[:, :n_epochs * 2000][0]
#             print(epochs.shape)
#             labels = labels[:n_epochs * 2000]
#
#             ## train
#             epochs = np.array(np.split(epochs, n_epochs, axis=1))
#             print(f'max: {np.max(epochs)}, min:{np.min(epochs)}')
#             epochs = np.clip(epochs, a_min=-150 * 1e-6, a_max=150 * 1e-6)
#             labels = np.array(np.split(labels, n_epochs, axis=0))
#             print(f"epochs.shape: {epochs.shape}")
#             print(f'labels: {len(labels)}')
#             print(f'select_idx: {select_idx}, len: {len(select_idx)}')
#             epochs = epochs[select_idx]
#             labels = labels[select_idx]
#             cnt = 0
#             filename = f'/home/cuizaixu_lab/huangweixuan/data/data/{MASS_aug_name}/SS2/{expert}'
#             # filename = path_root
#             for idx in range(len(epochs)):
#                 count_label += int(np.sum(labels[idx]) > 0)
#                 save_epochs = epochs[idx]
#                 save_labels = labels[idx]
#
#                 batch_epochs = torch.from_numpy(np.copy(save_epochs))
#                 batch_labels = torch.from_numpy(np.copy(save_labels))
#                 print(f'torch.sum(save_labels): {torch.sum(batch_labels)}')
#
#                 dataframe = pd.DataFrame(
#                     {'x': [save_epochs.tolist()], 'Spindles': [save_labels.tolist()], 'bads': [bads]}
#                 )
#                 table = pa.Table.from_pandas(dataframe)
#                 os.makedirs(f"{filename}/{name}/train", exist_ok=True)
#                 with pa.OSFile(
#                         f"{filename}/{name}/train/{str(cnt).zfill(5)}.arrow", "wb"
#                 ) as sink:
#                     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
#                         writer.write_table(table)
#                 cnt += 1
#                 del dataframe
#                 del table
#                 gc.collect()
#                 print('==========Augment==========')
#                 if torch.sum(batch_labels) <= 25:
#                     print(f'coutinue')
#                     continue
#                 # collate(dict{'Spindle_label': label, 'x':(epochs, channels), index)
#                 # label:tensor(batch_size, seq_length)} epochs:(batch, channel, seq_len) index:no use
#                 for i in range(aug_time):
#                     batch_epoch = batch_epochs.detach().clone() * 1e6
#                     batch = {'Spindle_label': batch_labels.detach().clone().unsqueeze(0),
#                              'x': (batch_epoch, torch.tensor([4, 5, 16, 18, 22, 36, 38, 52])),
#                              'index': torch.tensor(1)}  # index is no use
#                     batch = collate([batch])
#                     print(f'===========begin: {i}===========cnt:{cnt}===========')
#                     res = pre_train(batch, stage='test')
#                     pre_train.first_log_gpu = True
#                     # print(
#                     #     f"pre_train.unpatchify(res['cls_feats']).shape:{pre_train.unpatchify(res['cls_feats']).shape}")
#                     generate_epoch = res['cls_feats'] * res['time_mask_patch'].unsqueeze(-1) + \
#                                      pre_train.patchify(res['batch']['epochs'][0]) * (
#                                                  1 - res['time_mask_patch']).unsqueeze(-1)
#                     generate_epoch = pre_train.unpatchify(generate_epoch)[0, :, :2000]
#                     UN = unnormalize()
#                     generate_epoch = UN(generate_epoch, attention_mask=None)/1e6
#                     # save_epochs = pre_train.unpatchify(generate_epoch)[0, :, :2000]
#                     assert save_epochs.shape[1] == 2000, f'shape : {save_epochs.shape}'
#                     save_label = batch_labels.detach().clone()
#                     inverse_flag = 'N'
#                     if torch.rand(1) < 0.5:
#                         print('inverse the batch')
#                         generate_epoch = torch.flip(generate_epoch, dims=[-1])
#                         save_label = torch.flip(save_label, dims=[-1])
#                         inverse_flag = 'Y'
#                     if plot_cnt <= 5:
#                         plot_generate(generate_epoch=generate_epoch, res=res, batch_epoch=batch_epoch,
#                                       aug_time=aug_time, i=i, save_label=save_label, inverse=inverse_flag)
#                         plot_cnt += 1
#                     dataframe = pd.DataFrame(
#                         {'x': [generate_epoch.tolist()], 'Spindles': [save_label.tolist()], 'bads': [bads],
#                          'time_mask_patch': [res['time_mask_patch'].tolist()]}
#                     )
#                     table = pa.Table.from_pandas(dataframe)
#                     os.makedirs(f"{filename}/{name}/train", exist_ok=True)
#                     with pa.OSFile(
#                             f"{filename}/{name}/train/{str(cnt).zfill(5)}.arrow", "wb"
#                     ) as sink:
#                         with pa.RecordBatchFileWriter(sink, table.schema) as writer:
#                             writer.write_table(table)
#                     cnt += 1
#                     del dataframe
#                     del table
#                     gc.collect()
#
#             ## validation and test
#             cnt = 0
#             filename = f'/home/cuizaixu_lab/huangweixuan/data/data/{MASS_aug_name}/SS2/{expert}'
#             print(f'len(epochs): {len(epochs)}')
#             print(f'test path: {filename}')
#             for idx in range(len(epochs)):
#                 save_epochs = epochs[idx]
#                 save_labels = labels[idx]
#                 dataframe = pd.DataFrame(
#                     {'x': [save_epochs.tolist()], 'Spindles': [save_labels.tolist()], 'bads': [bads]}
#                 )
#                 table = pa.Table.from_pandas(dataframe)
#                 os.makedirs(f"{filename}/{name}/test", exist_ok=True)
#                 with pa.OSFile(
#                         f"{filename}/{name}/test/{str(cnt).zfill(5)}.arrow", "wb"
#                 ) as sink:
#                     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
#                         writer.write_table(table)
#                 cnt += 1
#                 del dataframe
#                 del table
#                 gc.collect()
#                 if idx != (len(epochs) - 1):
#                     save_epochs = np.concatenate((epochs[idx][:, 1000:], epochs[idx + 1][:, :1000]), axis=-1)
#                     save_labels = np.concatenate((labels[idx][1000:], labels[idx + 1][:1000]), axis=-1)
#                     dataframe = pd.DataFrame(
#                         {'x': [save_epochs.tolist()], 'Spindles': [save_labels.tolist()], 'bads': [bads]}
#                     )
#                     table = pa.Table.from_pandas(dataframe)
#                     with pa.OSFile(
#                             f"{filename}/{name}/test/{str(cnt).zfill(5)}.arrow", "wb"
#                     ) as sink:
#                         with pa.RecordBatchFileWriter(sink, table.schema) as writer:
#                             writer.write_table(table)
#                     cnt += 1
#                     del dataframe
#                     del table
#                     gc.collect()
#     print('End')
# E1_sub = f'/home/cuizaixu_lab/huangweixuan/data/data/{MASS_aug_name}/SS2/E1/*'
# names = []
# train_nums = []
# test_nums = []
# train_names = []
# test_names = []
# for sub in glob.glob(E1_sub):
#     names.append(sub)
#     train_names.append(os.path.join(sub, 'train'))
#     test_names.append(os.path.join(sub, 'test'))
# for name in names:
#     print(f'------{name}-------')
#     tmp = 0
#     name_train = os.path.join(name, 'train')
#     for item in os.listdir(name_train):
#         if os.path.isfile(os.path.join(str(name_train), str(item))):
#             tmp += 1
#     print(f'num: {tmp}')
#     train_nums.append(tmp)
#     name_test = os.path.join(name, 'test')
#     tmp = 0
#
#     for item in os.listdir(name_test):
#         if os.path.isfile(os.path.join(str(name_test), str(item))):
#             tmp += 1
#     print(f'num: {tmp}')
#     test_nums.append(tmp)
#
# train_nums = np.array(train_nums, dtype=object)
# test_nums = np.array(test_nums, dtype=object)
# train_names = np.array(train_names)
# test_names = np.array(test_names)
#
# n = len(names)
# idx = np.arange(n)
# np.random.shuffle(idx)
# names = np.array(names)
# k_split = n // 5
# res = {}
# path = f'/home/cuizaixu_lab/huangweixuan/data/data/{MASS_aug_name}/SS2/'
# for i in range(5):
#     st = i * k_split
#     ed = (i + 1) * k_split
#     idx_split = idx[st:ed]
#     idx_train = np.setdiff1d(idx, idx_split)
#     res[f'train_{i}'] = {}
#     res[f'train_{i}']['names'] = train_names[idx_train[1:]]
#     res[f'train_{i}']['nums'] = train_nums[idx_train[1:]]
#     res[f'val_{i}'] = {}
#     res[f'val_{i}']['names'] = test_names[idx_train[:1]]
#     res[f'val_{i}']['nums'] = test_nums[idx_train[:1]]
#     res[f'test_{i}'] = {}
#     res[f'test_{i}']['names'] = test_names[idx_split]
#     res[f'test_{i}']['nums'] = test_nums[idx_split]
#     print(len(res[f'test_{i}']['nums']), len(res[f'test_{i}']['names']), len(res[f'val_{i}']['nums']),
#           len(res[f'train_{i}']['nums']))
# np.save(os.path.join(path, f'all_split_E1'), arr=res, allow_pickle=True)

MASS_aug_name = f"MASS_aug_new_1"
E2_sub = f'/home/cuizaixu_lab/huangweixuan/data/data/{MASS_aug_name}/SS2/E2/*'
names = []
train_nums = []
test_nums = []
train_names = []
test_names = []
for sub in glob.glob(E2_sub):
    names.append(sub)
    train_names.append(os.path.join(sub, 'train'))
    test_names.append(os.path.join(sub, 'test'))
for name in names:
    print(f'------{name}-------')
    tmp = 0
    name_train = os.path.join(name, 'train')
    for item in os.listdir(name_train):
        if os.path.isfile(os.path.join(str(name_train), str(item))):
            tmp += 1
    print(f'num: {tmp}')
    train_nums.append(tmp)
    name_test = os.path.join(name, 'test')
    tmp = 0
    for item in os.listdir(name_test):
        if os.path.isfile(os.path.join(str(name_test), str(item))):
            tmp += 1
    print(f'num: {tmp}')
    test_nums.append(tmp)

train_nums = np.array(train_nums)
test_nums = np.array(test_nums)
train_names = np.array(train_names)
test_names = np.array(test_names)
print(f'train_names: {train_names}')
n = len(names)
idx = np.arange(n)
names = np.array(names)
k_split = 3
res = {}
path = f'/home/cuizaixu_lab/huangweixuan/data/data/{MASS_aug_name}/SS2/'
for i in range(5):
    st = i * k_split
    ed = (i + 1) * k_split
    idx_split = idx[st:ed]
    idx_train = np.setdiff1d(idx, idx_split)
    np.random.shuffle(idx_train)
    res[f'train_{i}'] = {}
    res[f'train_{i}']['names'] = train_names[idx_train[1:]]
    res[f'train_{i}']['nums'] = train_nums[idx_train[1:]]
    res[f'val_{i}'] = {}
    res[f'val_{i}']['names'] = test_names[idx_split]
    res[f'val_{i}']['nums'] = test_nums[idx_split]
    res[f'test_{i}'] = {}
    res[f'test_{i}']['names'] = test_names[idx_split]
    res[f'test_{i}']['nums'] = test_nums[idx_split]
    print(idx, st, ed, idx_train)
    print(f'train name : {res[f"train_{i}"]["names"]}')
    print(f'test name : {res[f"test_{i}"]["names"]}')
    print(f'val name : {res[f"val_{i}"]["names"]}')
np.save(os.path.join(path, f'all_split_E1_new_5'), arr=res, allow_pickle=True)


