import os

import lightning.pytorch as pl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dataset


class TestData(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.data_dir = config['data_path']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10


    def prepare_data(self):

        print("prepare_data")

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data = dataset.MNIST(root="mnist",
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

            self.mnist_train, self.mnist_val = random_split(train_data, [55000, 5000])
            print(stage)
        # Assign test dataset for use in dataloader(s)
        elif stage == 'test' or stage is None:
            self.mnist_test = dataset.MNIST(root="mnist",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
            print(stage)

        else:
            print("val")
            self.mnist_val = None

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)