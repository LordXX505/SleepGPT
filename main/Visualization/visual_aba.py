# import numpy as np
# import matplotlib.pyplot as plt
# path='../../result'#图片输出路径
# fig = plt.figure()#创建画板
# ax = fig.add_subplot()
# x1 = [2.5e-4, 5e-4, 7.5e-4, 1e-3]
# y1 = [0.882, 0.886, 0.884, 0.885]
# print(x1, y1)
# ax.plot(x1, y1, 'b', marker='*', label='Transformer'
#         )
#
# x2 = [5e-4, 1e-3]
# y2 = [0.891, 0.892]
# ax.plot(x2, y2, 'r', marker='x', label='SwinTransformer'
#         )
#
# # x3 = [50, 75, 100]
# # y3 = [0.799, 0.810, 0.811]
#
# # ax.plot(x3, y3, 'r', marker='o', label='Only Finetune with top two layers in Physio2018'
# #         )
# plt.ylim((0.88, 0.90))
# plt.xlabel('Learning Rate')
# plt.ylabel('ACC.')
# plt.legend()
# import os
# fig.savefig(os.path.join(path,'lr_t_w.svg'),format='svg',dpi=150)#输出
# plt.show()
# import os
# import sys
#
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from main.modules.backbone import Model

from main.config import ex
import matplotlib.pyplot as plt
from typing import List
from main.modules.mixup import Mixup

def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0', '#e67e22','#B883D4'
             , '#9E9E9E']
    return color[:nums]


def get_names():
    # return ['C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'Fpz', 'O1', 'O2',
    #    'Pz']
    return ['C3', 'C4', 'EMG', 'EOG1', 'F3',  'Fpz', 'O1',
           'Pz']


@ex.automain
def main(_config):
    # pre_train = Model_Pre(_config)
    if _config['mode'] == 'pretrain' or _config['mode'] == 'visualization':
        pre_train = Model_Pre(_config)
    else:
        pre_train = Model(_config)
    print(_config)
    dm = MultiDataModule(_config)
    dm.setup(stage='test')
    pre_train.training = True
    # pre_train.eval()
    c = pre_train.transformer.choose_channels.shape[0]
    pre_train.set_task()
    print(c)
    cnt = 0
    for _, _dm in enumerate(dm.dms):
        n = len(_dm.test_dataset)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for id in idx:
            cnt += 1
            if cnt>2:
                sys.exit(0)
            batch = _dm.test_dataset[id]
            batch2 = _dm.test_dataset[id+1]
            batch = dm.collate([batch, batch2])
            # mixup = Mixup(mixup_alpha=0.8,
            #     prob=1, switch_prob=0.5, mode='batch',
            #     label_smoothing=0.1, num_classes=5)
            fig, Axes = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(30, 32))
            batch['epochs'][0] = batch['epochs'][0][:, 0]
            Axes[0].plot(batch['epochs'][0][0])
            Axes[1].plot(batch['epochs'][0][1])
            plt.savefig('../../result/orig.svg', format='svg')
            plt.show()
            # batch, target, (x1, x2) = mixup(batch=batch['epochs'][0], target=torch.tensor([0, 1]), return_box=True)
            fig, Axes = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(30, 32))
            Axes[0].plot(batch[0])
            Axes[1].plot(batch[1])
            print(x1, x2)
            plt.savefig('../../result/mixup.svg', format='svg')
            plt.show()

