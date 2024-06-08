from functools import partial

from main.datasets import MASSDataset
from main.datasets import MASSDataset_SS1, MASSDataset_SS2, \
    MASSDataset_SS3, MASSDataset_SS4, MASSDataset_SS5
from . import BaseDataModule
import os

data_set_cls = {
    'SS1': MASSDataset_SS1,
    'SS2': MASSDataset_SS2,
    'SS3': MASSDataset_SS3,
    'SS4': MASSDataset_SS4,
    'SS5': MASSDataset_SS5,
    'Spindle': MASSDataset
}


class MASSDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)
        if self.config['mode'] != 'Spindledetection':
            if 'Finetune_mass_all' in self.config['mode']:
                base_data_dir = os.path.basename(self.data_dir)
                print(f'MASS data module using datasets: {base_data_dir}')
                self.dataset = data_set_cls[base_data_dir]
            else:
                # ss_nums = self.config['mode'].split('_')[-1]
                base_data_dir = os.path.basename(self.data_dir)
                print(f'MASS data module using datasets: {base_data_dir}')
                # self.dataset = data_set_cls[base_data_dir]
                self.dataset = partial(data_set_cls[base_data_dir], file_name=f'MASS_P_{base_data_dir}.npy')
        else:
            self.dataset = MASSDataset

    @property
    def channels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    @property
    def column_names(self):
        if self.config['mode'] != 'Spindledetection':
            return ['x', 'Stage_label']
        else:
            return ['x', 'Spindles']

    @property
    def stage(self):
        if self.config['mode'] != 'Spindledetection':
            return True
        else:
            return False

    @property
    def spindle(self):
        if self.config['mode'] != 'Spindledetection':
            return False
        else:
            return True

    @property
    def dataset_cls(self):
        return self.dataset

    @property
    def dataset_name(self):
        return 'MASS'

    def setup(self, stage, kfold=None, expert='E1', **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(kfold=kfold, expert=expert,
                                      settings=self.config['data_setting'], **kwargs)
                print('MASS S')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(kfold=kfold, expert=expert,
                                       settings=self.config['data_setting'], **kwargs)
                self.set_val_dataset(kfold=kfold, expert=expert,
                                     settings=self.config['data_setting'], **kwargs)
                self.setup_flag += 1
                print('MASS s')
