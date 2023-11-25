from main.datasets import SDDataset
from . import BaseDataModule


class SDDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)


    @property
    def channels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]

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
        return SDDataset

    @property
    def dataset_name(self):
        return 'SD'

    def setup(self, stage):
        if stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset()
                print('SD S')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset()
                self.setup_flag += 1
                print('SD s')
