from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from . import _datamodules


class MultiDataModule(LightningDataModule):

    def __init__(self, _config, kfold=None):
        self.collate = None
        self.test_sampler = None
        self.val_sampler = None
        self.train_sampler = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        datamodule_keys = _config['datasets']
        assert len(datamodule_keys) > 0
        super().__init__()
        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config, idx) for idx, key in enumerate(datamodule_keys)}
        # print('dm_dicts: ', len(self.dm_dicts))
        self.dms = [v for k, v in self.dm_dicts.items()]
        self.batch_size = self.dms[0].batch_size
        self.num_workers = self.dms[0].num_workers

        self.dist = _config['dist_on_itp'] and (_config['device'] == 'cuda')
        self.pretrain = _config['mode'] == 'pretrain'
        self.kfold = kfold
        self.expert = _config['expert']

    def prepare_data(self) -> None:
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage: str) -> None:
        for dm in self.dms:
            config_dict = {}
            if self.kfold is not None:
                config_dict['kfold'] = self.kfold
            if self.expert is not None:
                config_dict['expert'] = self.expert
            dm.setup(stage, **config_dict)
        if stage == 'fit':
            self.collate = self.dms[0].train_dataset.collate
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
        if self.dist:
            if stage == 'test':
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            elif stage=='validate':
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True, )
            else:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                # self.train_sampler = None
                if not self.pretrain:
                    self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True,)
                else:
                    self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
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
            collate_fn=self.collate,
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
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        return loader



