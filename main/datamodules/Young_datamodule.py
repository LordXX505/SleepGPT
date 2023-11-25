from main.datasets import YoungDataset
from . import BaseDataModule


class YoungDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)

    @property
    def channels(self):
        return ['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5'
            , 'CP6', 'Cz', 'ECG', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'F1', 'F2', 'F3', 'F4', 'F5'
            , 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fpz', 'Fz'
            , 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'POz'
            , 'Pz', 'T7', 'T8', 'TP7', 'TP8']

    @property
    def column_names(self):
        if self.config['spindle'] is True:
            return ['x', 'Stage_label', 'Spindle_label']
        else:
            return ['x', 'Stage_label']

    @property
    def stage(self):
        if self.config['mode'] == 'pretrain':
            return False
        else:
            return True

    @property
    def spindle(self):
        return self.config['spindle'] is True

    @property
    def dataset_cls(self):
        return YoungDataset

    @property
    def dataset_name(self):
        return 'Young'

    def setup(self, stage):
        if self.setup_flag == 0:
            self.set_val_dataset()
            self.set_test_dataset()
            self.setup_flag += 1
            print('Young s')