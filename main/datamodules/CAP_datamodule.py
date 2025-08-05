from functools import partial

from main.datasets import MASSDataset
from main.datasets import CAPDataset_n, CAPDataset_ins, \
    CAPDataset_narco, CAPDataset_nfle, CAPDataset_plm, CAPDataset_rbd, CAPDataset_sdb
from . import BaseDataModule
import os
data_set_cls = {
    'n': CAPDataset_n,
    'ins': CAPDataset_ins,
    'narco': CAPDataset_narco,
    'nfle': CAPDataset_nfle,
    'plm': CAPDataset_plm,
    'rbd': CAPDataset_rbd,
    'sdb': CAPDataset_sdb
}


class CAPDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)
        if self.config['mode'] != 'UMAP':
            if 'Finetune_cap_all' in self.config['mode'] or 'visualization_mask_ratio_dynamic' in self.config['mode']:
                base_data_dir = os.path.basename(self.data_dir)
                print(f'CAP data module using datasets: {base_data_dir}')
                self.dataset = data_set_cls[base_data_dir]
            else:
                raise NotImplementedError
        else:
            base_data_dir = os.path.basename(self.data_dir)
            print(f'CAP data module using datasets: {base_data_dir}')
            self.dataset = data_set_cls[base_data_dir]

    @property
    def channels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    @property
    def column_names(self):
        return ['signal', 'stage', 'good_channels', 'pathology']

    @property
    def stage(self):
        return True

    @property
    def spindle(self):
        return False

    @property
    def pathology(self):
        return True

    @property
    def ods(self):
        return False

    @property
    def dataset_cls(self):
        return self.dataset

    @property
    def dataset_name(self):
        return 'CAP'
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
        else:
            settings = None
        split = 'val'
        kwargs.pop('settings')

        self.val_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],

            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            pathology=self.pathology,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs,

        )

    def setup(self, stage, kfold=None, expert=None, **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(kfold=kfold, expert=expert,
                                      settings=self.config['data_setting']['CAP'], **kwargs)
                print('CAP S')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(kfold=kfold, expert=expert,
                                       settings=self.config['data_setting']['CAP'], **kwargs)
                self.set_val_dataset(kfold=kfold, expert=expert,
                                     settings=self.config['data_setting']['CAP'], **kwargs)
                self.setup_flag += 1
                print('CAP s')
