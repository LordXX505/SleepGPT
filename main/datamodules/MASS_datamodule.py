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
    'Spindle': MASSDataset,
    'Aug_Random': MASSDataset_SS2
}


class MASSDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)
        if self.config['mode'] != 'Spindledetection':
            if 'Finetune_mass_all' in self.config['mode'] or 'visualization_mask_ratio_dynamic' in self.config['mode']:
                base_data_dir = os.path.basename(self.data_dir)
                print(f'MASS data module using datasets: {base_data_dir}')
                self.dataset = data_set_cls[base_data_dir]
            elif 'Finetune_mass_portion' in self.config['mode']:
                portion = self.config['mode'].split('_')[-1]
                base_data_dir = os.path.basename(self.data_dir)
                print(f'MASS data module using datasets: {base_data_dir}')
                aug_test = self.config['aug_test']
                if aug_test is not None:
                    file_name = f'Aug_{portion}_{aug_test}.npy'
                else:
                    file_name = f'Aug_{portion}.npy'
                self.dataset = partial(data_set_cls[base_data_dir],
                                       need_normalize=(base_data_dir != 'Aug_Random'),
                                       file_name=file_name)
            else:
                # ss_nums = self.config['mode'].split('_')[-1]
                base_data_dir = os.path.basename(self.data_dir)
                print(f'MASS data module using datasets: {base_data_dir}')
                # self.dataset = data_set_cls[base_data_dir]
                self.dataset = partial(data_set_cls[base_data_dir], file_name=f'MASS_channel_{base_data_dir}.npy')
        else:
            aug_dir = self.config['aug_dir']
            aug_prob = self.config['aug_prob']
            aug_test = self.config['aug_test']
            if aug_test is not None:
                npy_file = f"all_split_{self.config['expert']}_new_5_{aug_test}.npy"
            else:
                npy_file = f"all_split_{self.config['expert']}_new_5.npy"

            self.dataset = partial(MASSDataset, file_name=npy_file, aug_dir=aug_dir,
                                       aug_prob=aug_prob,)

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
    def pathology(self):
        return False

    @property
    def ods(self):
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
