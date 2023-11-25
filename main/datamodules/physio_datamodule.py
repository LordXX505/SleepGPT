from main.datasets import physioDataset
from . import BaseDataModule


class physioDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)

    @property
    def channels(self):
        return ['C3' 'C4' 'ECG' 'EMG1' 'EOG1' 'F3' 'F4' 'O1' 'O2']
    #C3: 4, c4:5, ecg:15, emg1:16, eog1:18, f3:22, f4:23, o1:38, o2:39

    @property
    def column_names(self):
        if self.config['mode'] != 'pretrain' and 'visualization' not in self.config['physio_settings']:
            return ['x', 'Stage_label']
        else:
            return ['x']

    @property
    def stage(self):
        if self.config['mode'] == 'pretrain' or 'visualization' in self.config['physio_settings']:
            return False
        else:
            return True

    @property
    def spindle(self):
        return False

    @property
    def dataset_cls(self):
        return physioDataset

    @property
    def dataset_name(self):
        return 'physio'


    def setup(self, stage, kfold=None):
        if stage == 'predict' or stage=='test':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['physio_settings'], kfold=kfold)
                print('physio s')
                self.setup_flag += 1
        else:
            if self.setup_flag < 1:
                self.set_train_dataset(settings=self.config['physio_settings'], kfold=kfold)
                self.set_val_dataset(settings=self.config['physio_settings'], kfold=kfold)
                print('physio s')
                self.setup_flag += 1
