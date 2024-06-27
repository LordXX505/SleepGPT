from functools import partial

from main.datasets import EDFDataset, EDF_Aug_Dataset
from main.datasets import EDf_Pre_Dataset
from . import BaseDataModule


class EDFDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)
        if 'Other' not in self.config['mode']:
            if '2013' in self.config['EDF_Mode']:
                self.dataset = partial(EDFDataset, file_name=f'm_new_split_k_20.npy')
            elif '2018' in self.config['EDF_Mode']:
                self.dataset = partial(EDFDataset, file_name=f'm_new_split_k_10.npy')
            else:
                raise NotImplementedError
        else:
            if self.config['mode'] == 'Other_EDF_Pretrain':
                if self.config['EDF_Mode'] == '9_5_5':
                    self.dataset = partial(EDf_Pre_Dataset, file_name='edf_pretrain_new.npy')
                elif self.config['EDF_Mode'] == 'n2v':
                    self.dataset = partial(EDf_Pre_Dataset, file_name='edf_pretrain_n2v.npy')
            else:
                if self.config['EDF_Mode'] == '9_5_5':
                    self.dataset = partial(EDf_Pre_Dataset, file_name="edf_downstream_9_5_5_new.npy")
                elif self.config['EDF_Mode'] == 'n2v':
                    self.dataset = partial(EDf_Pre_Dataset, file_name='edf_downstream_n2v.npy')
                elif self.config['EDF_Mode'] == 'mul':
                    self.dataset = partial(EDf_Pre_Dataset, file_name='edf_downstream_mul.npy')
                elif self.config['EDF_Mode'] == 'TCC':
                    self.dataset = partial(EDf_Pre_Dataset, file_name='edf_downstream_TCC.npy')
                elif self.config['EDF_Mode'] == 'usleep':
                    self.dataset = partial(EDf_Pre_Dataset, file_name='edf_downstream_usleep.npy')

    @property
    def channels(self):
        return ['EMG1' 'EOG1' 'FPz', 'Pz']

    # C3: 4, c4:5, ecg:15, emg1:16, eog1:18, f3:22, f4:23, o1:38, o2:39
    @property
    def column_names(self):
        if self.config['mode'] != 'pretrain' and 'visualization' not in self.config['data_setting'] \
                and 'EDF_Pretrain' not in self.config['mode']:
            return ['x', 'Stage_label']
        else:
            return ['x']

    @property
    def stage(self):
        if self.config['mode'] != 'pretrain' and 'visualization' not in self.config['data_setting'] \
                and 'EDF_Pretrain' not in self.config['mode']:
            return True
        else:
            return False

    @property
    def spindle(self):
        return False

    @property
    def dataset_cls(self):
        return self.dataset

    @property
    def dataset_name(self):
        return 'EDF'

    def setup(self, stage, kfold=None, **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['data_setting']['EDF'], kfold=kfold, **kwargs)
                print('EDF_settings s')
                self.setup_flag += 1
        else:
            if self.setup_flag < 1:
                self.set_train_dataset(settings=self.config['data_setting']['EDF'], kfold=kfold, **kwargs)
                self.set_val_dataset(settings=self.config['data_setting']['EDF'], kfold=kfold, **kwargs)
                print('EDF_settings s')
                self.setup_flag += 1


class EDFAugDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)
        if '2013' in self.config['EDF_Mode']:
            edf_mode = self.config['EDF_Mode'].split('_')
            if len(edf_mode) > 1:
                edf_mode_aug = '_'.join(edf_mode[1:])
                self.dataset = partial(EDF_Aug_Dataset, file_name=f'm_new_split_Aug_{edf_mode_aug}_k_20.npy',
                                       need_normalize=False)
            else:
                self.dataset = partial(EDF_Aug_Dataset, file_name=f'm_new_split_Aug_k_20.npy')
        elif '2018' in self.config['EDF_Mode']:
            edf_mode = self.config['EDF_Mode'].split('_')
            if len(edf_mode) > 1:
                edf_mode_aug = '_'.join(edf_mode[1:])
                self.dataset = partial(EDF_Aug_Dataset, file_name=f'm_new_split_Aug_{edf_mode_aug}_k_10.npy',
                                       need_normalize=False)
            else:
                self.dataset = partial(EDF_Aug_Dataset, file_name=f'm_new_split_Aug_k_10.npy')

        else:
            raise NotImplementedError
    @property
    def channels(self):
        return ['EMG1' 'EOG1' 'FPz', 'Pz']

    # C3: 4, c4:5, ecg:15, emg1:16, eog1:18, f3:22, f4:23, o1:38, o2:39
    @property
    def column_names(self):
        if self.config['mode'] != 'pretrain' and 'visualization' not in self.config['data_setting'] \
                and 'EDF_Pretrain' not in self.config['mode']:
            return ['x', 'Stage_label']
        else:
            return ['x']

    @property
    def stage(self):
        if self.config['mode'] != 'pretrain' and 'visualization' not in self.config['data_setting'] \
                and 'EDF_Pretrain' not in self.config['mode']:
            return True
        else:
            return False

    @property
    def spindle(self):
        return False

    @property
    def dataset_cls(self):
        return self.dataset

    @property
    def dataset_name(self):
        return 'EDF'

    def setup(self, stage, kfold=None, **kwargs):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['data_setting']['EDF'], kfold=kfold, **kwargs)
                print('EDF_settings s')
                self.setup_flag += 1
        else:
            if self.setup_flag < 1:
                self.set_train_dataset(settings=self.config['data_setting']['EDF'], kfold=kfold, **kwargs)
                self.set_val_dataset(settings=self.config['data_setting']['EDF'], kfold=kfold, **kwargs)
                print('EDF_settings s')
                self.setup_flag += 1
