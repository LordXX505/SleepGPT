import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from main.modules.backbone import Model
import pytorch_lightning as pl
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
    pl.seed_everything(512)
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
            if cnt>3:
                sys.exit(0)
            batch = _dm.test_dataset[id]
            batch2 = _dm.test_dataset[id+1]
            batch = dm.collate([batch, batch2])
            batch['random_mask'][0][1] = torch.zeros(120)
            # batch['random_mask'][0][1][:15] = torch.ones(15)
            # batch['random_mask'][0][1][60:75] = torch.ones(15)
            # batch['random_mask'][0][1][105:120] = torch.ones(15)
            for i in range(8):
                batch['random_mask'][0][1][(i*15+10):(i*15+15)] = torch.ones(5)

            fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(30, 32))
            fig.suptitle('Masked RandomPlot')
            color = get_param(c)
            pre_train.set_task()
            res = pre_train(batch, stage='test')
            epochs = res['batch']['epochs'][0]
            epochs_fft = res['batch']['epochs'][1]
            loss = pre_train.forward_masked_loss_channel(res['mtm_logits'], epochs, res['time_mask_patch'])
            loss2 = pre_train.forward_masked_loss_2D(res['mtm_logits_fft'], epochs_fft, res['fft_mask_patch'])
            # mix_batch, target, box = Mixup(epochs, [0, 1], return_box=True)
            # print(target, box)
            # for i, channels in enumerate(pre_train.transformer.choose_channels):
            #     axes = Axes[i][0]
            #     print(patch_epochs_mask[i])
            #     axes.plot(range(3000), mix_batch[0][i][:3000].detach().numpy(), color[i])
            #     axes.grid(True)
            #     axes.set_xticks(np.arange(0, 3000, 200))
            #     axes.set_yticks(np.arange(0, 2, 0.1))
            #     axes = Axes[i][1]
            #     axes.plot(range(3000), mix_batch[1][i][:3000].detach().numpy(), color[i])
            #
            #     axes.set_yticks(np.arange(0, 2, 0.1))
            #     axes.set_xticks(np.arange(0, 3000, 200))
            #     axes.grid(True)
            # plt.plot()
            # return
            patch_epochs = pre_train.patchify(epochs)
            patch_epochs_fft = pre_train.patchify_2D(epochs_fft)
            mask = res['time_mask_patch'].bool()
            mask_fft = res['fft_mask_patch'].bool()
            print('loss:', loss)
            print('loss2', loss2)
            patch_epochs_mask = patch_epochs.masked_fill(mask[:, :, None], np.nan)
            patch_epochs_mask2 = patch_epochs_fft.masked_fill(mask_fft[:, :, None], np.nan)
            patch_epochs_mask = pre_train.unpatchify(patch_epochs_mask)[1]
            patch_epochs_mask2 = pre_train.unpatchify_2D(patch_epochs_mask2)[1]

            masked_time = pre_train.unpatchify(res['mtm_logits'].masked_fill(~mask[:, :, None], np.nan))[1]
            print(mask_fft.shape, res['mtm_logits_fft'].shape)
            masked_fft = pre_train.unpatchify_2D(res['mtm_logits_fft'].masked_fill(~mask_fft[:, :, None], np.nan))[1]
            # masked_fft = pre_train.unpatchify_2D(res['mtm_logits_fft'])[1]
            patch_epochs = pre_train.unpatchify(patch_epochs)[1]
            patch_epochs_fft = pre_train.unpatchify_2D(patch_epochs_fft)[1]
            names = get_names()
            for i, channels in enumerate(pre_train.transformer.choose_channels):
                axes = Axes[i][0]
                print(patch_epochs_mask[i])
                axes.plot(range(3000), patch_epochs_mask[i][:3000].detach().numpy(), color[-2])
                # axes.grid(True)
                axes.set_title(names[i] + ' ' + format(loss[1][i].item(), '.3f'))
                # axes.set_xticks(np.arange(0, 3000, 200))
                # axes.set_yticks(np.arange(0, 2, 0.1))
                axes = Axes[i][1]
                axes.plot(range(3000), masked_time[i].detach().numpy(), color[-2])
                axes.plot(range(3000), patch_epochs[i].detach().numpy(), 'r', alpha=0.2)
                axes.set_title(names[i])

                # axes.set_yticks(np.arange(0, 2, 0.1))
                # axes.set_xticks(np.arange(0, 3000, 200))
                # axes.grid(True)
            path = '/'.join(_config['load_path'].split('/')[-4:-2])
            print(f"../../result/{path}/{_config['datasets'][_]}")
            os.makedirs(f"../../result/{path}/{_config['datasets'][_]}", exist_ok=True)
            plt.savefig(f"../../result/{path}/{_config['datasets'][_]}/predict_{id}.svg", format='svg')

            plt.figure()
            fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(30, 32))
            for i, channels in enumerate(pre_train.transformer.choose_channels):
                axes = Axes[i][0]
                axes.imshow(masked_fft[i].detach().numpy(), aspect='auto', origin='lower')
                axes.set_title(names[i] + '_' + str(loss2.item()))
                axes = Axes[i][1]
                axes.set_title(names[i])
                axes.imshow(patch_epochs_fft[i].detach().numpy(), aspect='auto', origin='lower')
            print('save fft png')
            os.makedirs(f"../../result/{path}/{_config['datasets'][_]}", exist_ok=True)
            plt.savefig(f"../../result/{path}/{_config['datasets'][_]}/predict_fft_nu_{id}.svg",  format='svg')
            plt.close("all")




