from functools import partial

from lightning import LightningModule
from torch.utils.data import DataLoader
from typing import Any
from . import BaseDataModule
from torch.utils.data import ConcatDataset
from main.datasets import UMSDataset

class UMSDataModule(BaseDataModule):

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
        self.dataset = partial(UMSDataset, file_name=f"UMS_20fold_fixed_updated.npy",
                                           )
        self.ods_inj = _config['spo2_ods_settings']['inj']

    @property
    def channels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    @property
    def column_names(self):
        if self.ods_inj:
            return ['signal', 'stage', 'spo2', 'ods', 'ods_valid']
        else:
            return ['signal', 'stage']

    @property
    def stage(self):
       return True

    @property
    def spindle(self):
        return False
    
    @property
    def pathology(self):
        return False
    
    @property
    def apnea(self):
        return False
    
    @property
    def dataset_cls(self):
        return self.dataset

    @property
    def ods(self):
        return self.ods_inj

    def set_val_dataset(self, *args, **kwargs):
        """
        Basedatasets
        :param transform_keys: transform and augment
        :param data_dir: base data dir
        :param names: subject names
        :param column_names: epochs(x), spindle ,stages
        :param fs: 100hz
        :param epoch_duration: 30s. Mass is 20s.
        :param stage: need stage labels.
        :param spindle: nedd spindle labels.
        """

        if "settings" in kwargs.keys():
            settings = kwargs['settings']
            kwargs.pop('settings')
        else:
            settings = None
        if 'kfold' in kwargs.keys():
            kfold = kwargs['kfold']
            kwargs.pop('kfold')
        else:
            kfold = None
        self.val_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split='val',
            stage=self.stage,
            spindle=self.spindle,
            pathology=self.pathology,
            ods=self.ods,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            kfold=kfold,
            split_len=self.config['split_len'],
            **kwargs
        )

    def setup(self, stage, kfold=None, **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['data_setting']['UMS'], kfold=kfold, **kwargs)
                print('UMS_setting s')
                self.setup_flag += 1
        else:
            if self.setup_flag < 1:
                self.set_train_dataset(settings=self.config['data_setting']['UMS'], kfold=kfold, **kwargs)
                self.set_val_dataset(settings=self.config['data_setting']['UMS'], kfold=kfold, **kwargs)
                print('UMS_setting s')
                self.setup_flag += 1

