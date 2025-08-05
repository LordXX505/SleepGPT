# visualize_masked_reconstruction_demo.py

import os
import sys
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')

from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone_pretrain import Model_Pre
from typing import List


def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0', '#e67e22','#B883D4', '#9E9E9E']
    return color[:nums]

def get_names():
    return ['abd', 'ari', 'C3', 'C4', 'ecg', 'EMG', 'EOG1', 'F3',  'Fpz', 'O1', 'Pz']


def run_demo(config):
    model = Model_Pre(config)
    model.mask_same = True
    pl.seed_everything(512)

    dm = MultiDataModule(config, kfold=config['kfold'])
    dm.setup(stage='test')
    model.eval()
    c = model.transformer.choose_channels.shape[0]
    model.set_task()

    cnt = 0
    load_path = os.path.basename(config['load_path'])
    pattern = r"epoch=\d+"
    try:
        match = re.search(pattern, load_path).group()
    except:
        match = 'last'

    for _, _dm in enumerate(dm.dms):
        n = len(_dm.test_dataset)
        idx = np.arange(n)
        np.random.shuffle(idx)

        for id in idx:
            if cnt > 20:
                return
            cnt += 1
            batch = _dm.test_dataset[id]
            batch = dm.collate([batch])

            fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(30, 32))
            fig.suptitle('Masked RandomPlot')
            color = get_param(c)

            model.set_task()
            res = model(batch, stage='test')

            epochs = res['batch']['epochs'][0]
            epochs_fft = res['batch']['epochs'][1]
            loss = model.forward_masked_loss_channel(res['mtm_logits'], epochs, res['time_mask_patch'])

            patch_epochs = model.patchify(epochs)
            patch_epochs_fft = model.patchify_2D(epochs_fft)

            mask = res['time_mask_patch'].bool()
            mask_fft = res['fft_mask_patch'].bool()
            patch_epochs_mask = patch_epochs.masked_fill(mask[:, :, None], np.nan)
            patch_epochs_mask2 = patch_epochs_fft.masked_fill(mask_fft[:, :, None], np.nan)

            patch_epochs_mask = model.unpatchify(patch_epochs_mask)[0]
            patch_epochs_mask2 = model.unpatchify_2D(patch_epochs_mask2)[0]
            masked_time = model.unpatchify(res['mtm_logits'].masked_fill(~mask[:, :, None], np.nan))[0]
            masked_fft = model.unpatchify_2D(res['mtm_logits_fft'].masked_fill(~mask_fft[:, :, None], np.nan))[0]
            patch_epochs = model.unpatchify(patch_epochs)[0]
            patch_epochs_fft = model.unpatchify_2D(patch_epochs_fft)[0]

            names = get_names()
            for i, channels in enumerate(model.transformer.choose_channels):
                axes = Axes[i][0]
                axes.plot(range(3000), patch_epochs_mask[i][:3000].detach().numpy(), color[-2])
                axes.set_title(names[i] + ' ' + format(loss[0][i].item(), '.3f'))

                axes = Axes[i][1]
                axes.plot(range(3000), masked_time[i].detach().numpy(), color[-2])
                axes.plot(range(3000), patch_epochs[i].detach().numpy(), 'r', alpha=0.2)
                axes.set_title(names[i])

            path = '/'.join(config['load_path'].split('/')[-4:-2])
            out_path = f"./result/{path}/{config['datasets'][_]}/epoch_{match}"
            os.makedirs(out_path, exist_ok=True)
            plt.savefig(f"{out_path}/predict_{id}_{loss.mean():.4f}.svg", format='svg')
            plt.close("all")

            # FFT plot
            fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(30, 32))
            for i, channels in enumerate(model.transformer.choose_channels):
                Axes[i][0].imshow(masked_fft[i].detach().numpy(), aspect='auto', origin='lower')
                Axes[i][0].set_title(names[i] + '_0')
                Axes[i][1].imshow(patch_epochs_fft[i].detach().numpy(), aspect='auto', origin='lower')
                Axes[i][1].set_title(names[i])

            plt.savefig(f"{out_path}/predict_fft_{id}.svg", format='svg')
            plt.close("all")