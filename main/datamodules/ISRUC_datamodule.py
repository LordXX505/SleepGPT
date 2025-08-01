from functools import partial

from main.datasets import ISRUCDataset
from . import BaseDataModule


class ISRUCDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)
        mode = self.config['mode'].split('_')[-1]
        if mode == 'S3':
            self.dataset = partial(ISRUCDataset, file_name='ISRUC_S3_split_k_10_no_val.npy')
        elif mode == 'S1':
            if 'no' in self.config['mode']:
                self.dataset = partial(ISRUCDataset, file_name='ISRUC_S1_split_k_10_noval.npy')
            else:
                self.dataset = partial(ISRUCDataset, file_name='ISRUC_S1_split_k_10.npy')
    @property
    def channels(self):
        return ['EMG1' 'EOG1' 'FPz', 'Pz']

    # C3: 4, c4:5, ecg:15, emg1:16, eog1:18, f3:22, f4:23, o1:38, o2:39
    @property
    def column_names(self):
        if self.config['mode'] != 'pretrain':
            return ['x', 'Stage_label']
        else:
            return ['x']

    @property
    def stage(self):
        if self.config['mode'] != 'pretrain':
            return True
        else:
            return False

    @property
    def spindle(self):
        return False

    @property
    def ods(self):
        return False

    @property
    def dataset_cls(self):
        return self.dataset

    @property
    def dataset_name(self):
        return 'ISRUC'

    def setup(self, stage, kfold=None, **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(kfold=kfold, settings=self.config['data_setting']['ISRUC'], **kwargs)
                print('ISRUC_settings s')
                self.setup_flag += 1
        else:
            if self.setup_flag < 1:
                print(f"ISRUC data_settings: {self.config['data_setting']}")
                self.set_train_dataset(kfold=kfold, settings=self.config['data_setting']['ISRUC'], **kwargs)
                self.set_val_dataset(kfold=kfold, settings=self.config['data_setting']['ISRUC'], **kwargs)
                print('ISRUC_settings s')
                self.setup_flag += 1
