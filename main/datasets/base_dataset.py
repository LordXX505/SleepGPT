import sys

import random

import numpy as np
import torch
import io
import pyarrow as pa
import os
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from main.transforms import keys_to_transforms, normalize
from torch.utils import data
from typing import Optional, Union
from pytorch_lightning.utilities.rank_zero import rank_zero_info

# large :
# ['C3' 'C4' 'EMG1' 'EOG1' 'F3' 'FPz' 'O1' 'Pz']

# Huge model:
#choose_channels = np.array(['ABD', 'AIRFLOW', 'C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'Fpz', 'O1','Pz'])

class BaseDatatset(data.Dataset):

    def __init__(
            self,
            transform_keys: dict,
            data_dir: str,
            names: Union[np.ndarray, list],
            nums: None,
            column_names: list,
            split='train',
            fs: int = 100,
            epoch_duration: int = 30,
            stage: bool = True,
            spindle: bool = False,
            concatenate=False,
            random_choose_channels=8,
            settings=None,
            mask_ratio=None,
            all_time=False,
            time_size=100,
            pool_all=False,
            split_len=None,
            patch_size=200,
            show_transform_param=False,
            need_normalize=True,
            mode='large',
            mask_strategies=None
    ):
        """
        :param transform_keys: transform and augment
        :param data_dir: base data dir
        :param names: subject names
        :param column_names: epochs(x), spindle ,stages
        :param fs: 100hz
        :param epoch_duration: 30s. Mass is 20s.
        :param stage: need stage labels.
        :param spindle: nedd spindle labels.
        """
        self.x = None
        self.Spindle_label = None
        self.Stage_label = None
        self.stage = stage
        self.patch_size = patch_size
        self.num_patches = 3000//self.patch_size
        self.spindle = spindle
        self.split = split
        self.time_size = time_size
        self.pool_all = pool_all
        super().__init__()
        self.random_choose_channels = random_choose_channels
        self.transforms = keys_to_transforms(transform_keys['keys'], transform_keys['mode'], show_param=show_transform_param)
        if "train" not in self.split:
            self.transforms = keys_to_transforms([[]], ['full'], show_param=show_transform_param)
        rank_zero_info(f"transforms: {self.transforms}, split: {self.split}")
        self.data_dir = data_dir
        self.all_time = all_time
        names = np.array(names)
        self.idx_2_name = {}
        self.idx_2_nums = []
        self.nums_2_idx = {}
        self.error_files = []
        self.mask_strategies = mask_strategies
        print(f'mask_strategies: {self.mask_strategies}')
        if split_len == None:
            self.split_len = self.time_size
        else:
            self.split_len = split_len
        self.all_num = 0
        if self.all_time:
            assert nums is not None
            self.nums = nums
            assert len(names) == len(nums), f'{len(names)}, {len(nums)}'
            if pool_all is False:
                all_num = 0
                for _, name in enumerate(names):
                    self.idx_2_name[_] = name
                    self.nums_2_idx[_] = all_num
                    self.idx_2_nums.append(all_num)
                    all_num += nums[_] - self.split_len + 1
                self.idx_2_nums = np.array(self.idx_2_nums)
                self.nums_2_idx[len(names)] = all_num
                self.all_num = all_num
            else:
                all_num = 0
                for _, name in enumerate(names):
                    self.idx_2_name[_] = name
                    self.nums_2_idx[_] = all_num
                    self.idx_2_nums.append(all_num)
                    all_num += nums[_] // self.split_len if nums[_] % self.split_len == 0 \
                        else ((nums[_] // self.split_len) + 1)
                    # if nums[_]%self.time_size==0:
                    #     all_num += nums[_]//self.time_size
                    # else:
                    #     all_num += nums[_]//self.time_size+1
                self.nums_2_idx[len(names)] = all_num
                self.idx_2_nums = np.array(self.idx_2_nums)
                self.all_num = all_num
                rank_zero_info(f"Dataset all_num: {all_num}")

        else:
            if concatenate:
                self.names = np.concatenate(names)
            else:
                self.names = names
        self.column_names = column_names
        self.normalize = normalize()
        self.max_channels = 57
        self.need_normalize = need_normalize
        rank_zero_info(f'==============need_normalize: {need_normalize}==============')
        # assert 'x' in self.column_names
        self.mode = mode
        if mode == 'large':
            assert self.random_choose_channels == 8
            self.choose_channels = np.array([4, 5, 16, 18, 22, 36, 38, 52])  # for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]
            # self.choose_channels = np.array([4, 5, 15, 18, 22, 36, 38, 52])
        else:
            assert self.random_choose_channels == 11
            self.choose_channels = np.array([0, 3, 6, 7, 17, 18, 20, 24, 38, 40, 54])  # Large vision
            # ['ABD', 'AIRFLOW', 'C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'O1', 'O2']

        self.settings = settings
        rank_zero_info(f'dataset settings: {self.settings}')
        if isinstance(mask_ratio, list):
            self.mask_ratio = mask_ratio[0]
        else:
            self.mask_ratio = mask_ratio

        if self.random_choose_channels >= self.choose_channels.shape[0]:
            # random_channels_num = self.random_choose_channels-self.choose_channels.shape[0]
            if self.random_choose_channels == 8:
                all_channels = np.array([4, 5, 16, 18, 22, 36, 38, 52])
            else:
                all_channels = np.array([0, 3, 6, 7, 17, 18, 20, 24, 38, 40, 54])

            # select_channels = np.setdiff1d(all_channels, self.choose_channels)
            # # np.random.shuffle(select_channels)
            # # select_channels = select_channels[:random_channels_num]
            # select_channels = np.concatenate([self.choose_channels, select_channels])
            self.select_channels = all_channels

    def pre_spindle(self, spindle):
        self.Spindle_label = []
        for item in spindle:
            self.Spindle_label.append(item)

        self.Spindle_label = np.array(self.Spindle_label)

    def pre_stage(self, stage):
        self.Stage_label = np.array(stage)

    def __len__(self):
        if self.all_time:
            return self.all_num

        else:
            return len(self.names)

    @property
    def all_channels(self):
        return torch.ones(self.max_channels)

    @property
    def channels(self):
        raise NotImplementedError

    def get_name(self, index):
        # print(f'idx_2_nums : {self.idx_2_nums}')
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        # print(f'before start idx: {start_idx}')
        if self.pool_all:
            start_idx *= self.split_len
        # print(f'after start idx: {start_idx}')
        return os.path.join(self.idx_2_name[idx], str(start_idx).zfill(5) + '.arrow')

    def get_epochs(self, data, gc=None):
        print(f'channels: {self.channels}')
        if isinstance(data, pa.ChunkedArray):
            x = np.array(data.to_pylist())
        elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
            x = np.array(data.as_py())
        else:
            x = np.array(data)
        x = x.astype(np.float32)
        if self.settings is not None and self.need_normalize is True:
            if 'ECG' in self.settings:
                if self.mode == 'huge':
                    idx = np.where(self.channels == 17)[0][0]
                    x[idx] = x[idx] * 1000
                else:
                    idx = np.where(self.channels == 15)[0][0]
                    x[idx] = x[idx] * 1000
            if 'AMP' in self.settings:
                x = x * 1e6
        if gc is not None:
            try:
                gc = np.array(gc.as_py())
            except AttributeError:
                gc = np.array(gc)
            channel = np.where(gc == 0, -1, self.channels)
        else:
            zero_rows = np.all(x == 0, axis=1)
            if np.any(zero_rows):
                zero_row_indices = np.where(zero_rows)[0]
                gc = np.ones(x.shape[0])
                gc[zero_row_indices] = 0
            gc = np.array(gc)
            channel = np.where(gc == 0, -1, self.channels)
        x = torch.from_numpy(x).float()
        channel = torch.from_numpy(channel)
        assert x.shape[0] == channel.shape[0], f"x shape: {x.shape[0]}, c shape: {channel.shape[0]}"
        return {'x': [x, channel]}

    def get_stage(self, data):
        return {'Stage_label': torch.from_numpy(np.array(data)).long()}

    def get_spindle(self, data):
        try:
            x = np.array(data.as_py())
        except:
            x = np.array(data.to_pylist())
        return {'Spindle_label': torch.from_numpy(x).squeeze().long()}

    def get_suite(self, index):
        result = None
        if self.all_time:
            ret = dict()
            idx = np.where(self.idx_2_nums <= index)[0][-1]
            # print(f'self.idx_2_nums: {self.idx_2_nums}, idx:{idx}, index: {index}')
            start_idx = index - self.nums_2_idx[idx]
            if self.pool_all:
                start_idx *= self.split_len
                # if start_idx+self.time_size-1>=self.nums_2_idx[idx+1]*self.time_size:
                #     start_idx = self.nums_2_idx[idx+1] - self.time_size
            epochs = []
            channel = []
            stages = []
            spindles = []
            epoch_mask = []
            indexs = []
            idx_2_name = self.get_name(index)
            if start_idx + self.time_size >= self.nums[idx]:
                start_idx = self.nums[idx] - self.time_size
            for i in range(self.time_size):
                if (start_idx + i) >= self.nums[idx]:
                    # print('(start_idx + i) >= self.nums[idx]')
                    # print(start_idx, i, self.nums[idx], idx, index)
                    epochs.append(torch.zeros(len(self.channels), 3000) + 1e-6)
                    channel.append(torch.zeros(len(self.channels)))
                    if self.stage:
                        stages.append(torch.ones(1, dtype=torch.long) * (-100))
                    epoch_mask.append(torch.zeros(1))
                    indexs.append(torch.tensor(-1))
                    continue
                else:
                    # print(f"self.idx_2_name[idx]: {self.idx_2_name[idx]}")
                    name = os.path.join(self.idx_2_name[idx], str(start_idx + i).zfill(5) + '.arrow')
                    # print(f'self.data_dir: {self.data_dir}, name:{name}, self.idx_2_name[idx]: {self.idx_2_name[idx]}')
                    epoch_mask.append(torch.ones(1))
                if not os.path.isfile(name):
                    name2 = os.path.join(self.data_dir, '/'.join(name.split('/')[-1:]))
                    name = os.path.join(self.data_dir, '/'.join(name.split('/')[-2:]))
                    if not os.path.isfile(name):
                        if not os.path.isfile(name2):
                            raise RuntimeError(
                                f"Error while read file idx  {index} in {name} or {name2}, File not exits")
                        else:
                            name = name2
                tables = pa.ipc.RecordBatchFileReader(
                    pa.memory_map(name, "r")
                ).read_all()
                try:
                    if 'good' in tables:
                        x = self.get_epochs(tables['x'][0], tables['good'][0])
                    else:
                        x = self.get_epochs(tables['x'][0])
                    assert x['x'][0].shape[1] == 3000, f"{idx_2_name}, {x['x'][0].shape[1]}"
                    epochs.append(x['x'][0])
                    channel.append(x['x'][1])

                    if self.stage:
                        stage = self.get_stage(tables['stage'])
                        stages.append(stage['Stage_label'])
                    if self.spindle:
                        spindle = self.get_spindle(tables['Spindles'])
                        spindles.append(spindle['Spindle_label'])
                    indexs.append(torch.tensor(start_idx + i))
                except Exception as e:
                    print(f"Error while read file idx {index} in {name} -> {e}")
                    sys.exit(0)
            ret['x'] = (torch.stack(epochs, dim=0), torch.stack(channel, dim=0))
            if self.stage:
                ret['Stage_label'] = torch.stack(stages)
            if self.spindle:
                ret['Spindle_label'] = torch.stack(spindles)
            ret['epoch_mask'] = torch.cat(epoch_mask)
            ret.update({'index': torch.stack(indexs).reshape(-1, 1)})
            if type(idx_2_name) == int:
                ret.update({'name': torch.tensor(idx_2_name)})  # idx_2_name is int
            else:

                ret.update({'name': idx_2_name})
            # rank_zero_info(f'ret["index"]: {ret["index"]}, name:{idx_2_name}')
            return ret
        else:
            name = self.names[index]
            rank_zero_info(f'names: {name}')
            if not os.path.isfile(name):
                name = os.path.join(self.data_dir, '/'.join(name.split('/')[-2:]))
                if not os.path.isfile(name):
                    raise RuntimeError(f"Error while read file idx {name}")
            try:
                tables = pa.ipc.RecordBatchFileReader(
                    pa.memory_map(name, "r")
                ).read_all()
            except Exception as e:
                print(f'reading name: {name} has an error, Exception is {e}')
                sys.exit(0)
            while result is None:
                try:
                    ret = dict()
                    if 'good' in tables.column_names:
                        x = self.get_epochs(tables['x'][0], tables['good'][0])
                    else:
                        x = self.get_epochs(tables['x'][0])
                    ret.update(x)
                    if self.stage:
                        ret.update(self.get_stage(tables['stage']))
                    if self.spindle:
                        ret.update(self.get_spindle(tables['spindle'][0]))
                    ret.update({'index': torch.ones(1) * index})
                except Exception as e:
                    print(f"Error while read file idx {index} in {name} -> {e}")
                    sys.exit(0)
                return ret

    def _random_mask_patches(self, colletc_idx, seq_len_3000):
        random_mask_w_temp = torch.zeros(len(self.choose_channels) * self.num_patches)
        colletc_idx = np.concatenate(colletc_idx)
        if seq_len_3000:
            N = colletc_idx.shape[0]
            noise = torch.rand(N)
            ids_shuffle = torch.argsort(noise)
            len_shuffle = int(N * self.mask_ratio)
            final_choose_idx = colletc_idx[ids_shuffle[:len_shuffle]]
        else:
            raise NotImplementedError
        random_mask_w_temp[final_choose_idx] = 1
        return random_mask_w_temp

    def _mask_channels(self, colletc_idx):
        random_mask_w_temp = torch.zeros(len(self.choose_channels), self.num_patches)
        # print(len(self.choose_channels))
        now_channels = np.array([c[0] // self.num_patches for c in colletc_idx])
        # print(f'now: {now_channels}')
        channel_len = len(now_channels)

        noise = torch.rand(channel_len)
        ids_shuffle = torch.argsort(noise)
        len_shuffle = min(int(channel_len * self.mask_ratio), channel_len - 1)
        mask_cs = ids_shuffle[:len_shuffle]
        # print(f'maskcs: {mask_cs}')
        random_mask_w_temp[now_channels[mask_cs]] = 1
        return random_mask_w_temp.reshape(-1)

    def _predict(self, colletc_idx):
        mask_len = int(self.num_patches * (1 - self.mask_ratio))
        random_mask_w_temp = torch.zeros(len(self.choose_channels), self.num_patches)
        now_channels = np.array([c[0] // self.num_patches for c in colletc_idx])
        random_mask_w_temp[now_channels, mask_len:] = 1
        # print(random_mask_w_temp)
        return random_mask_w_temp.reshape(-1)

    def mask_strategy(self, colletc_idx, seq_len_3000):
        # assert  self.mask_strategies is not None
        #
        if self.mask_strategies is None:
            return self._random_mask_patches(colletc_idx, seq_len_3000)
        elif self.mask_strategies == 'mask_channel':
            return self._mask_channels(colletc_idx)
        elif self.mask_strategies == 'predict':
            return self._predict(colletc_idx)
        else:
            p = torch.rand(1)
            # print(f'prob : {p}')
            if p <= 0.33:
                return self._random_mask_patches(colletc_idx, seq_len_3000)
            elif 0.33 < p <= 0.66:
                return self._mask_channels(colletc_idx)
            else:
                return self._predict(colletc_idx)
    def collate(self, batch_list):
        keys = set(keys for b in batch_list for keys in b.keys())
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch_list] for k in keys}
        dict_batch['epochs'] = []
        dict_batch['mask'] = []
        dict_batch['random_mask'] = []
        if 'Spindle_label' in dict_batch:
            label = dict_batch['Spindle_label']
        else:
            label = None
        for x_idx, x in enumerate(dict_batch['x']):
            epochs = x[0]
            channels = x[1]
            res_multi_epochs = []
            attention_multi_mask = []
            random_mask_w = []
            if not self.all_time:
                epochs = [epochs]
                channels = [channels]
            for _idx, (_x, channel) in enumerate(zip(epochs, channels)):
                if self.random_choose_channels >= self.choose_channels.shape[0]:
                    res_epochs = torch.zeros((self.random_choose_channels, 3000))
                    attention_mask = torch.zeros(self.random_choose_channels)
                    random_mask_w_temp = torch.zeros(len(self.choose_channels) * self.num_patches)
                    colletc_idx = []
                    seq_len_3000 = True
                    for i, index in enumerate(self.select_channels):
                        idx = np.where(channel == index)[0]
                        if idx.shape[0] != 0:
                            if _x[idx].shape[1] != 3000:
                                try:
                                    seq_len_3000 = False
                                    res_epochs[i, :_x[idx].shape[1]] = _x[idx]
                                except:
                                    raise RuntimeError
                            else:
                                res_epochs[i] = _x[idx]
                            attention_mask[i] = 1
                            colletc_idx.append(np.arange(i * self.num_patches, (i + 1) * self.num_patches))
                    if self.mask_ratio is not None and len(random_mask_w) == 0:
                        random_mask_w_temp = self.mask_strategy(colletc_idx, seq_len_3000)
                        random_mask_w.append(random_mask_w_temp)
                    # print(f"res_epochs: {res_epochs}")
                    if self.need_normalize is True:
                        res_epochs = self.normalize(res_epochs, attention_mask)

                    if label is not None:
                        res_epochs, label[x_idx][_idx] = self.transforms(res_epochs, label[x_idx][_idx])
                    else:
                        res_epochs = self.transforms(res_epochs)
                        # rank_zero_info(f'max: {torch.max(res_epochs, dim=-1)}, min: {torch.min(res_epochs, dim=-1)}')
                    # assert res_epochs.ndim == 2, f"{res_epochs.shape}, {self.transforms}"
                    attention_mask = attention_mask
                    # print("attention_mask shape: ", attention_mask.shape)
                    # attention_mask = attention_mask.unsqueeze(0).repeat(res_epochs.shape[0], 1)
                else:
                    attention_mask = torch.zeros(self.max_channels)
                    attention_mask[channel] = 1
                    res_epochs = torch.zeros((self.max_channels, _x.shape[1]))
                    res_epochs[channel] = _x
                    if self.need_normalize is True:
                        res_epochs = self.normalize(res_epochs, attention_mask)
                    res_epochs = self.transforms(res_epochs)
                res_multi_epochs.append(res_epochs)
                attention_multi_mask.append(attention_mask)
            res_multi_epochs = torch.cat(res_multi_epochs, dim=0)
            attention_multi_mask = torch.cat(attention_multi_mask, dim=0)
            # print(f"attention_multi_mask : {attention_multi_mask.shape}")
            dict_batch['epochs'].append(res_multi_epochs)
            dict_batch['mask'].append(attention_multi_mask)
            if self.mask_ratio is not None:
                dict_batch['random_mask'].append(torch.stack(random_mask_w, dim=0))
            # for i in res_epochs:
            #     print(max(i), end=', ')
            # print(' ')
        dict_batch['epochs'] = torch.stack(dict_batch['epochs'], dim=0)
        dict_batch['mask'] = torch.stack(dict_batch['mask'], dim=0)
        dict_batch['index'] = torch.stack(dict_batch['index'], dim=0)
        if self.mask_ratio is not None:
            dict_batch['random_mask'] = torch.stack(dict_batch['random_mask'], dim=0)
        if self.all_time:
            dict_batch['epochs'] = [dict_batch['epochs'].reshape(-1, self.random_choose_channels, 3000)]
            dict_batch['mask'] = [dict_batch['mask'].reshape(-1, self.random_choose_channels)]
            dict_batch['index'] = dict_batch['index'].reshape(-1, 1)
            dict_batch['epoch_mask'] = torch.stack(dict_batch['epoch_mask']).reshape(-1, self.time_size)
        else:
            dict_batch['epochs'] = [dict_batch['epochs']]
            dict_batch['mask'] = [dict_batch['mask']]
        if self.mask_ratio is not None:
            dict_batch['random_mask'] = dict_batch['random_mask'].transpose(0, 1)

        # print(dict_batch['random_mask'].shape)
        dict_batch.pop('x')
        return dict_batch


if __name__ == '__main__':
    bs = BaseDatatset(transform_keys={'keys': [[]], 'mode': ['full']}, data_dir='./', names=['../../00968.arrow'], nums=[1], column_names=['x'])

    items = bs.get_suite(0)
    print(items)