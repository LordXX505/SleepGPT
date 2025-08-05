import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import BalancedDistributed

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset, positive_indices, negative_indices, batch_size):
        super().__init__()
        self.dataset = dataset
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass
    def train_dataloader(self):
        sampler = BalancedDistributed.BalancedDistributedSampler(self.dataset, self.positive_indices,
                                                                 self.negative_indices, self.batch_size,
                                                                 drop_last=True)
        self.sampler = sampler
        print(f"[DEBUG] DataLoader created with batch_size={self.batch_size}, num_workers=4, len(sampler): {len(self.sampler)}")
        DL = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=4, drop_last=True)
        print(f"[DEBUG] Starting test with epochs={len(DL)}， rank:{self.trainer.local_rank}")
        return DL
    def test_dataloader(self):
        sampler = BalancedDistributed.BalancedDistributedSampler(self.dataset, self.positive_indices,
                                                                 self.negative_indices, self.batch_size,
                                                                 drop_last=True)
        self.sampler = sampler
        print(f"[DEBUG] DataLoader created with batch_size={self.batch_size}, num_workers=4, len(sampler): {len(self.sampler)}")
        DL = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=4, drop_last=True)
        print(f"[DEBUG] Starting test with epochs={len(DL)}， rank:{self.trainer.local_rank}")
        return DL

class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels.unsqueeze(1))
        print(f"Rank {self.global_rank}, Batch data shape: {data.shape}, Batch labels shape: {labels.shape}, Loss: {loss.item()}")
        return loss
    def test_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels.unsqueeze(1))
        print(f"Rank {self.global_rank}, Batch data shape: {data.shape}, batch_idx: {batch_idx}")
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# 假设正样本和负样本已经标记
data = torch.randn(18, 10)
labels = torch.cat([torch.ones(8), torch.zeros(10)])

positive_indices = torch.nonzero(labels == 1).squeeze().tolist()
negative_indices = torch.nonzero(labels == 0).squeeze().tolist()
print(f"[DEBUG] Positive samples: {len(positive_indices)}, Negative samples: {len(negative_indices)}")

dataset = MyDataset(data, labels)
batch_size = 2

# 初始化 PyTorch Lightning Trainer
checkpoint_callback = ModelCheckpoint(monitor="loss", mode="min")
trainer = Trainer(
    accelerator="gpu",
    devices=2,  # number of GPUs
    strategy="ddp",  # DistributedDataParallel
    max_epochs=5,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1
)
# 初始化数据模块和模型
data_module = MyDataModule(dataset, positive_indices, negative_indices, batch_size)
model = MyModel()
print(f"[DEBUG] Starting training with max_epochs=5")

# 训练模型
trainer.fit(model, data_module)
