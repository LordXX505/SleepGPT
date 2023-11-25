import numpy as np

from main.config import ex
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
import os
from main.modules import Test
from main.datamodules import TestData, physioDataModule
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from main import SDDataset, YoungDataset, physioDataset
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules import Model

# @ex.automain
# def main(_config):
#
#     dm = MultiDataModule(_config)
#     dm.setup(stage='train')
#     for i in dm.train_dataloader():
#         print(i)
#         break
import glob
data_dir = '/data/data/shhs'
root_dir = "/data/shhs/polysomnography/edfs"
shhs1 = []
shhs2 = []
for group in ['shhs1', 'shhs2']:
    root = os.path.join(root_dir, group)
    all_sub = glob.glob(root + '/*')
    for data in all_sub:
        data = os.path.basename(data).split('.')[0]
        print(data)
        if group == 'shhs1':
            shhs1.append(data)
        else:
            shhs2.append(data)
shhs1 = np.array(shhs1)
shhs2 = np.array(shhs2)
print(shhs1.shape)
print(shhs2.shape)
idx1 = np.arange(shhs1.shape[0])
idx2 = np.arange(shhs2.shape[0])

np.random.shuffle(idx1)
np.random.shuffle(idx2)
n1 = int(shhs1.shape[0]*0.3)
test1 = shhs1[idx1[:n1]]
_train1 = shhs1[idx1[n1:]]
train1 = _train1[100:]
val1 = _train1[:100]

n2 = int(shhs2.shape[0]*0.3)
test2 = shhs2[idx2[:n2]]
_train2 = shhs2[idx2[n2:]]

train2 = _train2[100:]
val2 = _train2[:100]
print(val1, val2, train1, train2, test1, test2)

def search_ar(names):
    root_dir = '/data/data/shhs'
    ans = []
    for name in names:
        path = os.path.join(root_dir, name)
        res = glob.glob(path+'/*')
        ans.append(np.array(res))
    return np.concatenate(ans)

val1_res = search_ar(val1)
val2_res = search_ar(val2)
train1_res = search_ar(train1)
train2_res = search_ar(train2)
test1_res = search_ar(test1)
test2_res = search_ar(test2)
np.save('/data/data/shhs/val1.npy', val1_res)
np.save('/data/data/shhs/val2.npy', val2_res)
np.save('/data/data/shhs/train1.npy', train1_res)
np.save('/data/data/shhs/train2.npy', train2_res)
np.save('/data/data/shhs/test1.npy', test1_res)
np.save('/data/data/shhs/test2.npy', test2_res)
print(val1_res.shape, val2_res.shape, train1_res.shape, train2_res.shape, test1_res.shape, test2_res.shape)

# len = 0
# data_dir = '/data/data/shhs'
# for splt in ['train.npy', 'val.npy', 'test.npy']:
#     temp = np.load(os.path.join(data_dir, splt))
#     len += temp.shape[0]
# print(len)
