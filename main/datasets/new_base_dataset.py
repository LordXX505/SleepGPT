import os
import re
import sys
import random
import numpy as np
import torch
import h5py
import pyarrow as pa
from torch.utils import data
from typing import Optional, Union
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from main.transforms import keys_to_transforms, normalize


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

class Aug_BaseDataset(data.Dataset):

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
            apnea: bool = False,
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
            mask_strategies=None,
            aug_dir=None,
            aug_prob=0.
    ):
        """
        :param transform_keys: transform and augment
        :param data_dir: base data dir
        :param names: subject names
        :param column_names: epochs(x), spindle, stages
        :param fs: 100hz
        :param epoch_duration: 30s. Mass is 20s.
        :param stage: need stage labels.
        :param spindle: need spindle labels.
        """
        self.x = None
        self.Spindle_label = None
        self.Stage_label = None
        self.stage = stage
        self.patch_size = patch_size
        self.num_patches = 3000 // self.patch_size
        self.spindle = spindle
        self.split = split
        self.time_size = time_size
        self.pool_all = pool_all
        self.apnea = apnea
        super().__init__()
        self.random_choose_channels = random_choose_channels
        self.transforms = keys_to_transforms(transform_keys['keys'], transform_keys['mode'],
                                             show_param=show_transform_param)
        if "train" not in self.split:
            self.transforms = keys_to_transforms([[]], ['full'], show_param=show_transform_param)
        rank_zero_info(f"transforms: {self.transforms}, split: {self.split}")
        self.data_dir = data_dir
        if split == 'train':
            self.aug_dir = aug_dir
            self.aug_prob = aug_prob
            rank_zero_info(f'train aug_dir : {aug_dir}, aug_prob: {aug_prob}')
        else:
            self.aug_dir = None
        self.all_time = all_time
        self.error_files = []
        names = np.array(names)
        self.idx_2_name = {}
        self.idx_2_nums = []
        self.nums_2_idx = {}
        self.mask_strategies = mask_strategies
        if split_len is None:
            self.split_len = self.time_size
        else:
            self.split_len = split_len
        self.all_num = 0
        # print(f'original nums: {nums}, split: {self.split}')

        if isinstance(nums, list):
            nums = np.array(list(flatten(nums)))
        # print(f'isinstance: {isinstance(names, list)}, {type(names)}')
        if isinstance(names, list):
            names = np.array(list(flatten(names)))
        if self.all_time:
            assert nums is not None
            if nums.ndim > 1:
                nums = nums.flatten()
            if names.ndim > 1:
                names = names.flatten()

            self.nums = nums
            print(f'name: {names}, nums: {nums, nums.ndim}, split: {self.split}')
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
                    # print(f'nums: {nums[_]}, {self.split_len}, index: {_}')
                    all_num += nums[_] // self.split_len if nums[_] % self.split_len == 0 \
                        else ((nums[_] // self.split_len) + 1)
                self.nums_2_idx[len(names)] = all_num
                self.idx_2_nums = np.array(self.idx_2_nums)
                self.all_num = int(all_num)
                print(f"Dataset all_num: {all_num}")
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
        # assert 'signal' in self.column_names
        self.mode = mode
        print(f'using mode: {self.mode}')

        if mode == 'large':
            assert self.random_choose_channels == 8
            self.choose_channels = np.array(
                [4, 5, 16, 18, 22, 36, 38, 52])  # for all pertrain [C3, C4, EMG, EOG, F3, Fpz, O1, Pz]
        else:
            assert self.random_choose_channels == 11
            self.choose_channels = np.array([0, 3, 6, 7, 17, 18, 20, 24, 38, 40, 54])  # Large vision

        self.settings = settings
        self.print_test = 0
        self.print_mask_len = True
        rank_zero_info(f'dataset settings: {self.settings}')
        if isinstance(mask_ratio, list):
            self.mask_ratio = mask_ratio[0]
        else:
            self.mask_ratio = mask_ratio

        if self.random_choose_channels >= self.choose_channels.shape[0]:
            if self.random_choose_channels == 8:
                all_channels = np.array([4, 5, 16, 18, 22, 36, 38, 52])
            else:
                all_channels = np.array([0, 3, 6, 7, 17, 18, 20, 24, 38, 40, 54])
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
            return int(self.all_num)
        else:
            return len(self.names)

    @property
    def all_channels(self):
        return torch.ones(self.max_channels)

    @property
    def channels(self):
        return np.array([0, 3, 6, 7, 17, 18, 20, 24, 40])
        # raise NotImplementedError

    def get_name(self, index):
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        if self.pool_all:
            start_idx *= self.split_len
        return os.path.join(self.idx_2_name[idx], str(start_idx).zfill(5) + '.arrow')

    def get_epochs(self, data, gc=None, need_norm=True):
        if isinstance(data, pa.ChunkedArray):
            x = np.array(data.to_pylist())
        elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
            x = np.array(data.as_py())
        else:
            x = np.array(data)
        x = x.astype(np.float32)
        if self.settings is not None and self.need_normalize is True and need_norm is True:
            if 'ECG' in self.settings:
                if self.mode == 'huge':
                    idx = np.where(self.channels == 17)[0][0]
                    x[idx] = x[idx] * 1000
                else:
                    idx = np.where(self.channels == 15)[0][0]
                    x[idx] = x[idx] * 1000
            if 'AMP' in self.settings:
                x = x * 1e6
        # if self.split == 'train':
        #     print(f'x :{x}, norm: {need_norm}')
        # if gc is not None:
        #     try:
        #         gc = np.array(gc.as_py())
        #     except AttributeError:
        #         gc = np.array(gc)
        # else:
        #     zero_rows = np.all(x == 0, axis=1)
        #     if np.any(zero_rows):
        #         zero_row_indices = np.where(zero_rows)[0]
        #         gc = np.ones(x.shape[0])
        #         gc[zero_row_indices] = 0
        #     gc = np.array(gc)
        channel = self.channels
        x = torch.from_numpy(x).float()
        channel = torch.from_numpy(channel)
        assert x.shape[0] == channel.shape[0], f"x shape: {x.shape[0]}, c shape: {channel.shape[0]}"
        return {'x': [x, channel]}

    def get_stage(self, data):
        return {'Stage_label': torch.from_numpy(np.array(data)).long()}

    def get_spindle(self, data):
        if isinstance(data, pa.ChunkedArray):
            x = np.array(data.to_pylist())
        elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
            x = np.array(data.as_py())
        else:
            x = np.array(data)
        return {'Spindle_label': torch.from_numpy(x).squeeze().long()}

    def get_apnea(self, data):
        if isinstance(data, pa.ChunkedArray):
            x = np.array(data.to_pylist())
        elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
            x = np.array(data.as_py())
        else:
            x = np.array(data)
        return {'Apnea_label': torch.from_numpy(x).squeeze().long()}

    def load_pyarrow_data(self, file_path):
        try:
            tables = pa.ipc.RecordBatchFileReader(
                pa.memory_map(file_path, "r")
            ).read_all()
            return tables
        except Exception as e:
            raise RuntimeError(f"Error reading PyArrow file {file_path}: {e}")

    def load_hdf5_data(self, file_path, idx):
        idx = int(idx)
        data_dict = {}
        with h5py.File(file_path, 'r') as hf:
            for key in self.column_names:
                if key == 'signal':
                    try:
                        data_dict['x'] = hf[key][idx, :]
                    except Exception as e:
                        print(f"Error reading HDF5 file {file_path}-{idx}-{key}")
                        raise RuntimeError(f"Error reading HDF5 file {file_path}-{idx}: {e}")
                elif key == 'good_channels':
                    try:
                        data_dict[key] = hf[key][:]
                    except Exception as e:
                        print(f"Error reading HDF5 file {file_path}-{idx}-{key}")
                        raise RuntimeError(f"Error reading HDF5 file {file_path}-{idx}: {e}")
                else:
                    try:
                        data_dict[key] = hf[key][idx]
                    except Exception as e:
                        print(f"Error reading HDF5 file {file_path}-{idx}-{key}")
                        raise RuntimeError(f"Error reading HDF5 file {file_path}-{idx}: {e}")
        # return {'err':'good'}
        return data_dict

    def _get_h5_name(self, name, idx):
        if idx is not None:
            name1 = os.path.join(self.idx_2_name[idx], 'data.h5')
        else:
            name1 = None
        name2 = os.path.join(self.data_dir, name.split('/')[-2], 'data.h5')
        return (name1, name2)

    def _get_base_name(self, name):
        base_name = os.path.join(self.data_dir, '/'.join(name.split('/')[-2:]))
        base_name2 = os.path.join(self.data_dir, '/'.join(name.split('/')[-3:]))
        return base_name, base_name2

    def _get_aug_name(self, name, idx=None):
        aug_dir = np.random.choice(self.aug_dir, 1, replace=False)[0]
        # print(f'aug_dir: {aug_dir}')
        name = os.path.join(aug_dir, '/'.join(name.split('/')[-2:]))
        return name

    def _get_name(self, name, idx=None):
        need_norm = True
        if self.aug_dir is not None and torch.rand(1) < self.aug_prob:
            assert self.split == 'train', f'split: {self.split}'
            assert idx is not None
            name = self._get_aug_name(name, idx)
            need_norm = False
            # print(name)
        if not os.path.isfile(name):
            name_base, name_base2 = self._get_base_name(name)
            nameh5_1, nameh5_2 = self._get_h5_name(name, idx)
            if not os.path.isfile(name_base):
                if not os.path.isfile(name_base2):
                    if not os.path.isfile(nameh5_1):
                        if not os.path.isfile(nameh5_2):
                            raise RuntimeError(f"Error while reading file {name}, {name_base},{name_base2}, {nameh5_1} or {nameh5_2}")
                        else:
                            return nameh5_2, need_norm
                    else:
                        return nameh5_1, need_norm
                return name_base2, need_norm
            else:
                return name_base, need_norm

        return name, need_norm
    def _get_sg_name(self, name):
        match = re.search(r'index=(\d+)', name)
        if match:
            group_name = match.group(1)
            index = int(group_name)
            name, exp_name = os.path.split(name)
            if 'h5' in exp_name:
                return self._get_name(os.path.join(name, 'data.h5')), index
            else:
                return self._get_name(os.path.join(name, f'{index}.arrow')), index
        else:
            return self._get_name(name), None

    def get_suite(self, index):
        result = None
        if self.all_time:
            ret = dict()
            idx = np.where(self.idx_2_nums <= index)[0][-1]
            start_idx = index - self.nums_2_idx[idx]
            if self.pool_all:
                start_idx *= self.split_len
            epochs, channel, stages, spindles, epoch_mask, indexs, need_norms = [], [], [], [], [], [], []
            idx_2_name = self.get_name(index)
            if start_idx + self.time_size >= self.nums[idx]:
                start_idx = self.nums[idx] - self.time_size
            for i in range(self.time_size):
                if (start_idx + i) >= self.nums[idx]:
                    epochs.append(torch.zeros(len(self.channels), 3000) + 1e-6)
                    channel.append(torch.zeros(len(self.channels)))
                    if self.stage:
                        stages.append(torch.ones(1, dtype=torch.long) * (-100))
                    epoch_mask.append(torch.zeros(1))
                    indexs.append(torch.tensor(-1))
                    continue
                else:
                    name = os.path.join(self.idx_2_name[idx], str(start_idx + i).zfill(5) + '.arrow')
                    epoch_mask.append(torch.ones(1))
                name, need_norm = self._get_name(name, idx)
                if self.print_test < 1:
                    print(f'name: {name}, need_norm: {need_norm and self.need_normalize}, {self.print_test}')
                    self.print_test += 1
                # if self.split == 'train':
                #     rank_zero_info(f'name: {name}, norm: {need_norm}')
                need_norms.append(need_norm)
                if name.endswith('.arrow'):
                    tables = self.load_pyarrow_data(name)
                    if 'good' in tables:
                        x = self.get_epochs(tables['x'][0], tables['good'][0], need_norm=need_norm)
                    else:
                        x = self.get_epochs(tables['x'][0], need_norm=need_norm)
                    assert torch.max(x['x'][0]) < 10000, f"x > 10000 : {torch.max(x['x'][0])} need_norm: {need_norm}"
                    assert x['x'][0].shape[1] == 3000 or x['x'][0].shape[1] == 2000, f"{idx_2_name}, {x['x'][0].shape[1]}"
                    epochs.append(x['x'][0])
                    channel.append(x['x'][1])
                    if self.stage:
                        stage = self.get_stage(tables['stage'])
                        stages.append(stage['Stage_label'])
                    if self.spindle:
                        spindle = self.get_spindle(tables['Spindles'])
                        spindles.append(spindle['Spindle_label'])
                    indexs.append(torch.tensor(start_idx + i))
                elif name.endswith('.h5'):
                    data_dict = self.load_hdf5_data(name, start_idx + i)
                    # return data_dict
                    if 'good_channels' in self.column_names:
                        x = self.get_epochs(data_dict['x'], data_dict['good_channels'], need_norm)
                    else:
                        x = self.get_epochs(data_dict['x'], need_norm)
                    epochs.append(x['x'][0])
                    channel.append(x['x'][1])
                    if self.stage and 'stage' in self.column_names:
                        stage = self.get_stage(data_dict['stage'])
                        stages.append(stage['Stage_label'])
                    if self.spindle and 'spindle' in self.column_names:
                        spindle = self.get_spindle(data_dict['spindle'])
                        spindles.append(spindle['Spindle_label'])
                    indexs.append(torch.tensor(start_idx + i))
            # return {}
            ret['x'] = (torch.stack(epochs, dim=0), torch.stack(channel, dim=0))
            ret['norms'] = need_norms
            if self.stage:
                ret['Stage_label'] = torch.stack(stages)
            if self.spindle:
                ret['Spindle_label'] = torch.stack(spindles)
            ret['epoch_mask'] = torch.cat(epoch_mask)
            ret.update({'index': torch.stack(indexs).reshape(-1, 1)})
            if isinstance(idx_2_name, int):
                ret.update({'name': torch.tensor(idx_2_name)})
            else:
                ret.update({'name': idx_2_name})
            return ret
        else:
            need_norms = []
            name = self.names[index]
            # rank_zero_info(f'names: {name}')
            name, need_norm, index = self._get_sg_name(name)
            need_norms.append(need_norm)
            if name.endswith('.arrow'):
                tables = self.load_pyarrow_data(name)
            else:
                tables = self.load_hdf5_data(name, index)
            while result is None:
                try:
                    ret = dict()
                    if 'good' in tables.column_names:
                        x = self.get_epochs(tables['x'][0], tables['good'][0], need_norm)
                    else:
                        x = self.get_epochs(tables['x'][0], need_norm)
                    ret.update(x)
                    if self.stage:
                        ret.update(self.get_stage(tables['stage']))
                    if self.spindle:
                        ret.update(self.get_spindle(tables['spindle'][0]))
                    if self.apnea:
                        ret.update(self.get_apnea(tables['apnea'][0]))
                    ret.update({'index': torch.ones(1) * index})
                except Exception as e:
                    print(f"Error while read file idx {index} in {name} -> {e}")
                    sys.exit(0)
                ret['norms'] = need_norms
                return ret

    def _random_mask_patches(self, colletc_idx, seq_len_3000):
        random_mask_w_temp = torch.zeros(len(self.choose_channels) * self.num_patches)
        try:
            colletc_idx = np.concatenate(colletc_idx)
            if seq_len_3000:
                N = colletc_idx.shape[0]
                noise = torch.rand(N)
                ids_shuffle = torch.argsort(noise)
                len_shuffle = int(N * self.mask_ratio)
                if self.print_mask_len is True:
                    print(f'len_shuffle: {len_shuffle}')
                    self.print_mask_len = False
                final_choose_idx = colletc_idx[ids_shuffle[:len_shuffle]]
            else:
                raise NotImplementedError
            random_mask_w_temp[final_choose_idx] = 1
        except:
            rank_zero_info(f'colletc_idx: {colletc_idx}')
        return random_mask_w_temp

    def _mask_channels(self, colletc_idx):
        random_mask_w_temp = torch.zeros(len(self.choose_channels), self.num_patches)
        # print(len(self.choose_channels))
        now_channels = np.array([c[0] // self.num_patches for c in colletc_idx])
        # print(f'now: {now_channels}')
        channel_len = len(now_channels)

        torch.manual_seed(1)
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
        return random_mask_w_temp.reshape(-1)

    def mask_strategy(self, colletc_idx, seq_len_3000):
        # assert  self.mask_strategies is not None
        if self.mask_strategies is None:
            return self._random_mask_patches(colletc_idx, seq_len_3000)
        else:
            p = torch.rand(1)
            # print(f'prob : {p}')
            if p <= 0.33:
                return self._random_mask_patches(colletc_idx, seq_len_3000)
            elif 0.33 < p <= 0.66:
                return self._mask_channels(colletc_idx)
            else:
                return self._predict(colletc_idx)
    def collate_pre_check(self, dict_batch):
        if 'Spindle_label' in dict_batch:
            label = dict_batch['Spindle_label']
        elif 'Apnea_label' in dict_batch:
            label = dict_batch['Apnea_label']
        else:
            label = None

        return label
    def collate(self, batch_list):
        keys = set(keys for b in batch_list for keys in b.keys())
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch_list] for k in keys}
        dict_batch['epochs'] = []
        dict_batch['mask'] = []
        dict_batch['random_mask'] = []
        label = self.collate_pre_check(dict_batch)
        for x_idx, x in enumerate(dict_batch['x']):
            epochs = x[0]
            channels = x[1]
            res_multi_epochs = []
            attention_multi_mask = []
            random_mask_w = []
            need_norms = dict_batch['norms'][x_idx]
            # print(need_norms)
            if not self.all_time:
                epochs = [epochs]
                channels = [channels]
            for _idx, (_x, channel) in enumerate(zip(epochs, channels)):
                if self.random_choose_channels >= self.choose_channels.shape[0]:
                    res_epochs = torch.zeros((self.random_choose_channels, 3000))
                    attention_mask = torch.zeros(self.random_choose_channels)
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
                        assert colletc_idx != [], f'self.select_channels: {self.select_channels}, channel: {channel}'
                        random_mask_w_temp = self.mask_strategy(colletc_idx, seq_len_3000)
                        random_mask_w.append(random_mask_w_temp)
                    if self.need_normalize is True and need_norms[_idx] is True:
                        res_epochs = self.normalize(res_epochs, attention_mask)
                        # if self.split == 'train':
                        #     print(f'res_epochs: {res_epochs}, normalize: {need_norms[_idx]}')
                    if label is not None:
                        res_epochs, label[x_idx][_idx] = self.transforms(res_epochs, label[x_idx][_idx])
                    else:
                        res_epochs = self.transforms(res_epochs)
                        # rank_zero_info(f'max: {torch.max(res_epochs, dim=-1)}, min: {torch.min(res_epochs, dim=-1)}')
                    attention_mask = attention_mask
                else:
                    attention_mask = torch.zeros(self.max_channels)
                    attention_mask[channel] = 1
                    res_epochs = torch.zeros((self.max_channels, _x.shape[1]))
                    res_epochs[channel] = _x
                    if self.need_normalize is True and need_norms[_idx] is True:
                        res_epochs = self.normalize(res_epochs, attention_mask)
                    res_epochs = self.transforms(res_epochs)
                res_multi_epochs.append(res_epochs)
                attention_multi_mask.append(attention_mask)
            res_multi_epochs = torch.cat(res_multi_epochs, dim=0)
            attention_multi_mask = torch.cat(attention_multi_mask, dim=0)
            dict_batch['epochs'].append(res_multi_epochs)
            dict_batch['mask'].append(attention_multi_mask)
            if self.mask_ratio is not None:
                dict_batch['random_mask'].append(torch.stack(random_mask_w, dim=0))
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
        dict_batch.pop('norms')
        return dict_batch


if __name__ == '__main__':
    '''
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
     '''
    nb = Aug_BaseDataset(transform_keys={'keys': [[]], 'mode': ['full']},
                         data_dir='/Users/hwx_admin/Downloads/test_mgh/ses-1/',
                         names=['/Users/hwx_admin/Downloads/test_mgh/ses-1/'], nums=[20],
                         column_names=['signal', 'good_channels', 'stage'],
                         split='train', random_choose_channels=11, mask_ratio=0.5, mask_strategies=True, time_size=1,
                         split_len=1, all_time=True,
                         patch_size=100, mode='huge')
    items = nb.get_suite(1)
    print(torch.mean(items['x'][0], dim=-1))
    print(torch.sqrt(torch.var(items['x'][0], dim=-1)))
    print(nb.collate([items]))
