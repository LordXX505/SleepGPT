from main.datasets import SHHSDataset
from . import BaseDataModule


class SHHS1DataModule(BaseDataModule):
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
        return SHHSDataset

    @property
    def dataset_name(self):
        return 'SHHS'
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
        split = 'Val'
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
            *args, **kwargs,

        )

    def set_train_dataset(self, *args, **kwargs):
        print('**************************')
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        kwargs.pop('settings')
        split = 'train11'
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
             * args, **kwargs,
        )

    def set_test_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        kwargs.pop('settings')
        if "umap_dataset" in kwargs.keys():
            umap_dataset = kwargs['umap_dataset']['save_extra_name']
            mode = kwargs['umap_dataset']['mode']
            kwargs.pop('umap_dataset')
        else:
            umap_dataset = None
            mode = None
        if mode is None or mode != 'UMAP':
            split = 'Test'
        else:
            split_name = umap_dataset.split('_')[:]
            if 'train' in split_name:
                split = 'train_osa_c2_new'
            else:
                split = 'test_osa_c2_new'
        self.predict_dataset = self.dataset_cls(
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

    def set_predict_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
        else:
            settings = None
        kwargs.pop('settings')
        split = 'train11'
        kwargs.pop('umap_dataset')

        self.predict_dataset = self.dataset_cls(
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
            *args, **kwargs,
        )

    def setup(self, stage, **kwargs):
        if stage == 'test':
            if self.setup_flag == 0:
                self.set_test_dataset(settings=self.config['data_setting']['SHHS'], **kwargs)
                print('predict SHHS1 s')
                self.setup_flag += 1
        elif stage == 'validate':
            if self.setup_flag == 0:
                self.set_val_dataset(settings=self.config['data_setting']['SHHS'], **kwargs)
                print('predict SHHS1 s')
                self.setup_flag += 1
        elif stage == 'predict':
            if self.setup_flag == 0:
                self.set_predict_dataset(settings=self.config['data_setting']['SHHS'], **kwargs)
                print('predict SHHS1 s')
                self.setup_flag += 1
        else:
            if self.setup_flag == 0:
                self.set_train_dataset(settings=self.config['data_setting']['SHHS'], **kwargs)
                self.set_test_dataset(settings=self.config['data_setting']['SHHS'], **kwargs)
                self.set_val_dataset(settings=self.config['data_setting']['SHHS'], **kwargs)
                self.setup_flag += 1
                print('SHHS1 s')
