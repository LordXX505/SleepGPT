from lightning import LightningModule
from torch.utils.data import DataLoader
from typing import Any


class BaseDataModule(LightningModule):

    def __init__(self, _config, *args: Any, **kwargs: Any):
        """
        BaseDataModule
        Args:
            _config: configs
            *args:
            **kwargs: idx
        """
        idx = kwargs.pop('idx', 0)
        super().__init__(*args, **kwargs)
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None
        self.config = _config
        self.data_dir = _config['data_dir'][idx]
        print(f"BaseDataModule data_dir : {self.data_dir}")
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["batch_size"]
        self.eval_batch_size = self.batch_size
        #############################
        # self.train_transform_keys = keys:[[]], mode:[[]]
        self.train_transform_keys = (
            _config["transform_keys"]
        )
        self.val_transform_keys = (
            _config["transform_keys"]
        )
        # self.train_transform_keys = (
        #     ["default"]
        #     if len(_config["train_transform_keys"]) == 0
        #     else _config["train_transform_keys"]
        # )
        # if isinstance(self.train_transform_keys, str):
        #     self.train_transform_keys = [self.train_transform_keys]
        #
        # self.val_transform_keys = (
        #     ["default"]
        #     if len(_config["val_transform_keys"]) == 0
        #     else _config["val_transform_keys"]
        # )
        # if isinstance(self.val_transform_keys, str):
        #     self.val_transform_keys = [self.val_transform_keys]
        ##############################

        self.setup_flag = 0


    @property
    def column_names(self):
        raise NotImplementedError("return name of column")

    @property
    def stage(self):
        raise NotImplementedError("return stage")

    @property
    def spindle(self):
        raise NotImplementedError("return spindle")
    @property
    def apnea(self):
        raise NotImplementedError("return spindle")

    @property
    def pathology(self):
        raise NotImplementedError("return pathology")

    @property
    def ods(self):
        raise NotImplementedError("return pathology")

    @property
    def dataset_cls(self):
        raise NotImplementedError("return list of dataset_cls")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    @property
    def channels(self):
        raise NotImplementedError("return channels")

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
            kwargs.pop('settings')
        else:
            settings = None
        if 'kfold' in kwargs.keys():
            kfold = kwargs['kfold']
            kwargs.pop('kfold')
        else:
            kfold = None
        self.val_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split='val',
            stage=self.stage,
            spindle=self.spindle,
            pathology=self.pathology,
            ods=self.ods,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            kfold=kfold,
            split_len=self.config['split_len'],
            **kwargs
        )

    def set_train_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
            kwargs.pop('settings')
        else:
            settings = None
        if 'kfold' in kwargs.keys():
            kfold = kwargs['kfold']
            kwargs.pop('kfold')
        else:
            kfold = None
        self.train_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],
            transform_keys=self.train_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split='train',
            stage=self.stage,
            spindle=self.spindle,
            pathology=self.pathology,
            ods=self.ods,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            kfold=kfold,
            split_len=self.config['split_len'],
            **kwargs

        )

    def set_test_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
            kwargs.pop('settings')
        else:
            settings = None
        if 'kfold' in kwargs.keys():
            kfold = kwargs['kfold']
            kwargs.pop('kfold')
        else:
            kfold = None

        self.test_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],

            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split='test',
            stage=self.stage,
            spindle=self.spindle,
            pathology=self.pathology,
            ods=self.ods,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            kfold=kfold,
            split_len=self.config['split_len'],
            **kwargs

        )

    def set_predict_dataset(self, *args, **kwargs):
        if "settings" in kwargs.keys():
            settings = kwargs['settings']
            kwargs.pop('settings')
        else:
            settings = None
        if 'kfold' in kwargs.keys():
            kfold = kwargs['kfold']
            kwargs.pop('kfold')
        else:
            kfold = None

        self.test_dataset = self.dataset_cls(
            patch_size=self.config['patch_size'],
            transform_keys=self.val_transform_keys,
            data_dir=self.data_dir,
            column_names=self.column_names,
            split='test',
            stage=self.stage,
            spindle=self.spindle,
            pathology=self.pathology,
            ods=self.ods,
            random_choose_channels=self.config['random_choose_channels'],
            settings=settings,
            mask_ratio=self.config['mask_ratio'],
            all_time=self.config['all_time'],
            time_size=self.config['time_size'],
            pool_all=self.config['use_all_label'] == 'all',
            kfold=kfold,
            split_len=self.config['split_len'],
            **kwargs
        )

    def setup(self, stage, **kwargs):
        if not self.setup_flag:
            if stage == 'predict':
                self.set_predict_dataset(**kwargs)
            else:
                self.set_train_dataset(**kwargs)
                self.set_test_dataset(**kwargs)
                self.set_val_dataset(**kwargs)
            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate

        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate
        )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.predict_dataset.collate
        )
        return loader
