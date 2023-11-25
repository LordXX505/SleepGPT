from main.datasets import SHHSDataset
from . import BaseDataModule


class SHHSDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)


    @property
    def channels(self):
        return [4, 5, 15, 16, 18]

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
        return SHHSDataset

    @property
    def dataset_name(self):
        return 'SHHS'

    def setup(self, stage):
        if stage == 'predict':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['shhs_settings'])
                print('SHHS s')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(settings=self.config['shhs_settings'])
                self.set_test_dataset(settings=self.config['shhs_settings'])
                self.set_val_dataset(settings=self.config['shhs_settings'])
                self.setup_flag += 1
                print('SHHS s')
