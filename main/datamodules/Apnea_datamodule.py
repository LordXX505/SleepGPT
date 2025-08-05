from functools import partial

from lightning import LightningModule
from torch.utils.data import DataLoader
from typing import Any
from . import BaseDataModule
from main.datasets import MASSDataset_SS1
from torch.utils.data import ConcatDataset

class ApneaDataModule(BaseDataModule):

    def __init__(self, _config, idx):
        """
        BaseDataModule
        Args:
            _config: configs
            *args:
            **kwargs: idx
        """
        super().__init__(_config, idx=idx)
        self.collate = None
        self.negtive_index = None
        self.positive_index = None
        if 'SS1' in self.config['mode']:
            dataset_positive = partial(MASSDataset_SS1, file_name='pos.npy')
            dataset_negtive = partial(MASSDataset_SS1, file_name='neg.npy')
            dataset_val = partial(MASSDataset_SS1, file_name='val.npy')
            dataset_test = partial(MASSDataset_SS1, file_name='test.npy')
            self.dataset_positive = dataset_positive
            self.dataset_negtive = dataset_negtive
            self.dataset_val = dataset_val
            self.dataset_test = dataset_test

    @property
    def channels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    @property
    def column_names(self):
        if self.config['mode'] == 'Spindledetection':
            return ['signal', 'spindles']
        elif 'Apneadetection' in self.config['mode']:
            return ['signal', 'apnea']
        else:
            return ['signal', 'stage']

    @property
    def stage(self):
       return False

    @property
    def spindle(self):
        return False

    @property
    def apnea(self):
        return True
    def set_val_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        split = 'val'
        kwargs.pop('settings')

        self.val_dataset = self.dataset_test(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs,
        )

    def set_train_dataset(self, *args, **kwargs):
        print('**************************')
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        kwargs.pop('settings')
        split = 'train'
        dataset_positive = self.dataset_positive(
            patch_size=self.config['patch_size'],
            transform_keys=self.train_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
             * args, **kwargs,
        )
        dataset_negtive = self.dataset_negtive(
            patch_size=self.config['patch_size'],
            transform_keys=self.train_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs,
        )
        self.train_dataset = ConcatDataset([dataset_positive, dataset_negtive])
        self.positive_index = len(dataset_positive)
        self.negtive_index = len(dataset_negtive)
        self.collate = dataset_positive.collate
    def set_test_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        kwargs.pop('settings')

        split = 'test'
        self.test_dataset = self.dataset_test(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs
        )
    def setup(self, stage, kfold=None, expert=None, **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(kfold=kfold, expert=expert,
                                      settings=self.config['data_setting']['MASS'], **kwargs)
                print('MASS S')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(kfold=kfold, expert=expert,
                                       settings=self.config['data_setting']['MASS'], **kwargs)
                self.set_val_dataset(kfold=kfold, expert=expert,
                                     settings=self.config['data_setting']['MASS'], **kwargs)
                self.setup_flag += 1
                print('MASS s')
