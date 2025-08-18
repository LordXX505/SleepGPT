from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from . import _datamodules
import numpy as np
from .sampler import BalancedDistributedSampler
class MultiDataModule(LightningDataModule):

    def __init__(self, _config, kfold=None):
        self.collate_val = None
        self.collate = None
        self.test_sampler = None
        self.val_sampler = None
        self.train_sampler = None
        self.predict_sampler = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.predict_dataset = None
        datamodule_keys = _config['datasets']
        assert len(datamodule_keys) > 0
        super().__init__()
        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config, idx) for idx, key in enumerate(datamodule_keys)}
        # print('dm_dicts: ', self.dm_dicts)
        self.dms = [v for k, v in self.dm_dicts.items()]
        self.batch_size = self.dms[0].batch_size
        self.num_workers = self.dms[0].num_workers
        self.need_BDS = 'Apneadetection' in _config['mode']
        print(f'need_BDS: {self.need_BDS}')
        self.dist = _config['dist_on_itp'] is True and (_config['device'] == 'cuda')
        self.pretrain = _config['mode'] == 'pretrain'
        self.kfold = kfold
        self.expert = _config['expert']
        self.patch_size = _config['patch_size']
        self.subest = _config['subset']
        self.show_transform_param = _config['show_transform_param']
        self.mode = _config['model_arch'].split('_')[1]
        self.mask_strategies = _config['mask_strategies']
        self.negtive_index = None
        self.positive_index = None
        self.print_test = 0
        self.umap_dataset = _config['visual_setting']

    def prepare_data(self) -> None:
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage: str) -> None:
        for dm in self.dms:
            config_dict = {'show_transform_param': self.show_transform_param}
            if self.kfold is not None:
                config_dict['kfold'] = self.kfold
            if self.expert is not None:
                config_dict['expert'] = self.expert
            config_dict['mode'] = self.mode
            if self.mask_strategies is not None:
                config_dict['mask_strategies'] = self.mask_strategies
            if stage == 'fit' or stage == 'validate' or stage == 'test':
                config_dict['print_test'] = self.print_test
                self.print_test += 1
            if self.umap_dataset['mode'] is not None:
                config_dict['umap_dataset'] = self.umap_dataset
            print(f'datamodule config_dict: {config_dict}')
            dm.setup(stage, **config_dict)
            if self.need_BDS:
                self.positive_index = dm.positive_index
                self.negtive_index = dm.negtive_index
                self.collate = dm.collate
        if stage == 'fit':
            if self.collate is None:
                for i in range(len(self.dms)):
                    if hasattr(self.dms[i], 'train_dataset') and self.dms[i].train_dataset is not None:
                        self.collate = self.dms[i].train_dataset.collate
                        break
            for i in range(len(self.dms)):
                if hasattr(self.dms[i], 'val_dataset') and self.dms[i].val_dataset is not None:
                    self.collate_val = self.dms[i].val_dataset.collate
                    break
            if self.pretrain:
                self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms if dm.train_dataset is not None])
                self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms if dm.val_dataset is not None])
                num = 0
                for dm in self.dms:
                    if dm.train_dataset is not None:
                        num += 1
                print(f'***************Using {num} train datasets****************')
                print(f'***************len of train datasets:{len(self.train_dataset )} ****************')
                num = 0
                for dm in self.dms:
                    if dm.val_dataset is not None:
                        num += 1
                print(f'***************Using {num} val datasets****************')
                print(f'***************len of val datasets:{len(self.val_dataset )} ****************')

                assert self.train_dataset is not None
            else:
                self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms if dm.train_dataset is not None])
                num = 0
                for dm in self.dms:
                    if dm.train_dataset is not None:
                        num += 1
                print(f'***************Using {num} train datasets****************')
                print(f'***************len of train datasets:{len(self.train_dataset )} ****************')
                self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms if dm.val_dataset is not None])
                num = 0
                for dm in self.dms:
                    if dm.val_dataset is not None:
                        print(f'***************len of val datasets:{len(dm.val_dataset)} ****************')
                        num += 1
                print(f'***************Using {num} val datasets****************')
                print(f'***************len of val datasets:{len(self.val_dataset )} ****************')
                # self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms if dm.test_dataset is not None])
        elif stage == 'validate':
            self.collate = self.dms[0].val_dataset.collate
            self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms if dm.val_dataset is not None])

        elif stage == 'test':
            self.collate = self.dms[0].test_dataset.collate
            self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
            num = 0
            for dm in self.dms:
                if dm.test_dataset is not None:
                    num += 1
                    print(f'***************len of test datasets:{len(dm.test_dataset)} ****************')
            print(f'***************Using {num} test datasets****************')
            print(f'***************len of test datasets:{len(self.test_dataset)} ****************')
        else:
            self.collate = self.dms[0].predict_dataset.collate
            self.predict_dataset = ConcatDataset([dm.predict_dataset for dm in self.dms])
            num = 0
            for dm in self.dms:
                if dm.predict_dataset is not None:
                    num += 1
                    print(f'***************len of predict datasets:{len(dm.predict_dataset)} ****************')
            print(f'***************Using {num} predict datasets****************')
            print(f'***************len of predict datasets:{len(self.predict_dataset)} ****************')
        if self.subest is not None and stage == 'fit':
            subset_size = int(self.subest * len(self.train_dataset))
            print(f'Train dataloader is partially used. subset_size: {subset_size}')
            indices = np.random.choice(len(self.train_dataset), subset_size, replace=False)
            self.train_dataset = Subset(self.train_dataset, indices)
        if self.dist:
            if stage == 'test':
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            elif stage == 'validate':
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, )
            elif stage == 'predict':
                self.predict_sampler = DistributedSampler(self.predict_dataset, shuffle=False, )
            else:
                if self.need_BDS:
                    positive_indices = np.arange(self.positive_index)
                    negative_indices = np.arange(self.positive_index, self.positive_index+self.negtive_index)
                    self.train_sampler = BalancedDistributedSampler(self.train_dataset, positive_indices=positive_indices,
                                                                    negative_indices=negative_indices, batch_size=self.batch_size,
                                                                    shuffle=True)
                else:
                    self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                # self.train_sampler = None
                if not self.pretrain:
                    self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False,)
                else:
                    self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
        print('setup s')

    def train_dataloader(self):

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate if self.collate_val is None else self.collate_val,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        return loader



