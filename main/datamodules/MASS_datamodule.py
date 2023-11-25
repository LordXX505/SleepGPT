from main.datasets import MASSDataset
from . import BaseDataModule


class MASSDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)


    @property
    def channels(self):
        return [0,1 ,2 ,3, 4, 5, 6,7]

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
        return MASSDataset

    @property
    def dataset_name(self):
        return 'MASS'

    def setup(self, stage, kfold=None, expert='E1'):
        if stage == 'predict' or stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(kfold=kfold, expert=expert,
                                      settings=self.config['mass_settings'])
                print('MASS S')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(kfold=kfold, expert=expert,
                                       settings=self.config['mass_settings'])
                self.set_val_dataset(kfold=kfold, expert=expert,
                                     settings=self.config['mass_settings'])
                self.setup_flag += 1
                print('MASS s')
