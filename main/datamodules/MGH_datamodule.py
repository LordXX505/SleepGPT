from main.datasets import MGHDataset
from . import BaseDataModule

class MGHDataModule(BaseDataModule):
    def __init__(self, _config, idx):
        super().__init__(_config, idx=idx)


    @property
    def channels(self):

        return [4, 5, 15, 16, 18]

    @property
    def column_names(self):
        if self.config['mode'] != 'pretrain' and 'visualization' not in self.config['data_setting']:
            return ['signal', 'stage', 'good_channels']
        else:
            return ['signal', 'good_channels']

    @property
    def stage(self):
        if self.config['mode'] == 'pretrain' or 'visualization' in self.config['data_setting']:
            return False
        else:
            return True

    @property
    def spindle(self):
        return False

    @property
    def ods(self):
        return False
    @property
    def dataset_cls(self):
        return MGHDataset

    @property
    def dataset_name(self):
        return 'MGH'
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
        # split = 'val_test'

        kwargs.pop('settings')
        self.val_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs
        )

    def set_train_dataset(self, *args, **kwargs):
        print('**************************')
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        kwargs.pop('settings')

        split = 'train' if self.config['mode'] != 'pretrain' else 'pre_train'
        # split = 'train_test'
        self.train_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],

            transform_keys=self.train_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs
        )

    def set_test_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        split = 'test'
        kwargs.pop('settings')
        # split = 'test_test'

        self.test_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split=split,
            stage=self.stage,
            spindle=self.spindle,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            split_len=self.config['split_len'],
            *args, **kwargs
        )
    def setup(self, stage, **kwargs):
        if stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['data_setting']['MGH'], **kwargs)
                print('predict MGH s')
                self.setup_flag += 1
        elif stage == 'validate':
            if self.setup_flag == 0:
                self.set_val_dataset(settings=self.config['data_setting']['MGH'], **kwargs)
                print('predict MGH s')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(settings=self.config['data_setting']['MGH'], **kwargs)
                self.set_test_dataset(settings=self.config['data_setting']['MGH'], **kwargs)
                self.set_val_dataset(settings=self.config['data_setting']['MGH'], **kwargs)
                self.setup_flag += 1
                print('MGH s')
