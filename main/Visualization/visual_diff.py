import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from main.config import ex
import matplotlib.pyplot as plt
from typing import List


def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0', '#e67e22','#B883D4'
             , '#9E9E9E']
    return color[:nums]


def get_names():
    return ['C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'Fpz', 'O1', 'O2',
       'Pz']


@ex.automain
def main(_config):
    import copy
    _config = copy.deepcopy(_config)
    _config['time_only'] = True

    pre_train = Model_Pre(config=_config)
    print(_config)
    dm = MultiDataModule(_config)
    _config['time_only'] = False

    _config['load_path'] = '/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_cosine_backbone_base_patch200_l1_pretrain/version_6/checkpoints/epoch=49-step=32586.ckpt'
    pre_train2 = Model_Pre(config=_config)
    dm.setup(stage='predict')
    pre_train.eval()
    c = pre_train.transformer.choose_channels.shape[0]
    print(c)
    for _, _dm in enumerate(dm.dms):
        n = len(_dm.test_dataset)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for id in idx[:10]:
            batch = _dm.test_dataset[id]
            batch = dm.collate([batch])
            batch2 = copy.deepcopy(batch)
            fig, Axes = plt.subplots(nrows=c, ncols=3, sharex='all', figsize=(33, 48))
            fig.suptitle('Masked RandomPlot')
            color = get_param(c)

            res = pre_train(batch, stage='test')
            res2 = pre_train2(batch2, stage='test')

            epochs = res['batch']['epochs'][:, :pre_train.transformer.max_channels]

            loss2 = pre_train2.forward_masked_loss_channel(res2['cls_feats'], epochs, res['time_mask_patch'])
            loss = pre_train.forward_masked_loss_channel(res['cls_feats'], epochs, res['time_mask_patch'])
            patch_epochs = pre_train.patchify(epochs)
            mask = res['time_mask_patch'].bool()
            print('loss:', loss)
            patch_epochs_mask = patch_epochs.masked_fill(mask[:, :, None], np.nan)
            patch_epochs_mask = pre_train.unpatchify(patch_epochs_mask)[0]
            masked_time = pre_train.unpatchify(res['cls_feats'].masked_fill(~mask[:, :, None], np.nan))[0]
            masked_time2 = pre_train2.unpatchify(res2['cls_feats'].masked_fill(~mask[:, :, None], np.nan))[0]
            patch_epochs = pre_train.unpatchify(patch_epochs)[0]
            names = get_names()
            for i, channels in enumerate(pre_train.transformer.choose_channels):
                axes = Axes[i][0]
                axes.plot(range(3000), patch_epochs_mask[i].detach().numpy(), color[i])
                axes.grid(True)
                axes.set_title(names[i] + '_' + str(loss[0][i].item()) + '__' + str(loss2[0][i].item()))
                axes.set_xticks(np.arange(0, 3000, 1000))
                axes.set_yticks(np.arange(0, 1, 0.1))
                axes = Axes[i][1]
                axes.plot(range(3000), masked_time[i].detach().numpy(), color[i])
                axes.plot(range(3000), patch_epochs[i].detach().numpy(), 'r', alpha=0.2)
                axes.set_title(names[i])

                axes.set_yticks(np.arange(0, 1, 0.1))
                axes.set_xticks(np.arange(0, 3000, 1000))
                axes.grid(True)

                axes = Axes[i][2]
                axes.plot(range(3000), masked_time2[i].detach().numpy(), color[i])
                axes.plot(range(3000), patch_epochs[i].detach().numpy(), 'r', alpha=0.2)
                axes.set_title(names[i])

                axes.set_yticks(np.arange(0, 1, 0.1))
                axes.set_xticks(np.arange(0, 3000, 1000))
                axes.grid(True)
            path = '/'.join(_config['load_path'].split('/')[-4:-2])
            print(f"../../result/{path}_diff/{_config['datasets'][_]}")
            os.makedirs(f"../../result/{path}_diff/{_config['datasets'][_]}", exist_ok=True)
            plt.savefig(f"../../result/{path}_diff/{_config['datasets'][_]}/mask{id}.png")
            plt.close("all")







