import os
import sys

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import numpy as np
sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')
import re
import matplotlib.pyplot as plt
from typing import List

def get_param(nums) -> List[str]:
    color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#4DBBD5CC", '#2ecc71', '#2980b9', '#FFEDA0', '#e67e22','#B883D4'
             , '#9E9E9E']
    return color[:nums]

def get_names():
    # return ['C3', 'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'Fpz', 'O1', 'O2',
    #    'Pz']
    return ['C3', 'C4', 'EMG', 'EOG1', 'F3',  'Fpz', 'O1', 'Pz']
def visual(batch, pl_module, persub):
    pre_train = pl_module
    C = pre_train.transformer.choose_channels.shape[0]
    _config = pl_module._config
    load_path = os.path.basename(_config['load_path'])
    pattern = r"epoch=\d+"
    match = re.search(pattern, load_path).group()
    res = pre_train(batch, stage='test')
    epochs = res['batch']['epochs'][0]
    epochs_fft = res['batch']['epochs'][1]
    loss = pre_train.forward_masked_loss_channel(res['cls_feats'], epochs, res['time_mask_patch'])
    loss2 = pre_train.forward_masked_loss_2D_channel(res['cls_feats_fft'], epochs_fft, res['fft_mask_patch'])
    pl_module.gpu_monitor(batch['epochs'][0], phase='compute visual', block_log=True)
    loss = loss.detach()
    loss2 = loss2.detach()
    idx = torch.where(loss > 100)
    loss = torch.where((loss > 100) | torch.isnan(loss), 0, loss)
    rank_zero_info(f'loss: {loss.mean(0)}')

    # rank_zero_info(f'loss2: {loss2}')
    if persub is True:
        rank_zero_info(f"batch name: {len(batch['name'])}")
        for index, (v, v2) in enumerate(zip(loss, loss2)):
            real_index = int(index / (pl_module.time_size))
            name = batch['name'][real_index]
            persub_v1 = getattr(pl_module, f"test_{name}_loss")(v.view(1, -1))
            persub_v2 = getattr(pl_module, f"test_{name}_loss2")(v2.view(1, -1))
    patch_epochs = pre_train.patchify(epochs)
    patch_epochs_fft = pre_train.patchify_2D(epochs_fft)
    mask = res['time_mask_patch'].bool()
    mask_fft = res['fft_mask_patch'].bool()
    all_loss = loss + loss2
    all_loss = all_loss.mean(dim=-1)
    min_idx = torch.argmin(all_loss)
    if pl_module.min_loss > all_loss[min_idx]:
        pl_module.min_loss = all_loss[min_idx]
    loss_update = getattr(pl_module, f"test_loss")(loss)
    loss2_update = getattr(pl_module, f"test_loss2")(loss2)
    rank_zero_info(f'update loss: {loss_update}, loss2: {loss2_update}')
    patch_epochs_mask = patch_epochs.masked_fill(mask[:, :, None], np.nan)
    patch_epochs_mask2 = patch_epochs_fft.masked_fill(mask_fft[:, :, None], np.nan)
    patch_epochs_mask = pre_train.unpatchify(patch_epochs_mask)[1]
    patch_epochs_mask2 = pre_train.unpatchify_2D(patch_epochs_mask2)[1]
    masked_time = pre_train.unpatchify(res['cls_feats'].masked_fill(~mask[:, :, None], np.nan))[1]
    masked_fft = pre_train.unpatchify_2D(res['cls_feats_fft'].masked_fill(~mask_fft[:, :, None], np.nan))[1]
    # masked_fft = pre_train.unpatchify_2D(res['mtm_logits_fft'])[1]
    patch_epochs = pre_train.unpatchify(patch_epochs)[1]
    patch_epochs_fft = pre_train.unpatchify_2D(patch_epochs_fft)[1]
    names = get_names()

    c = pre_train.transformer.choose_channels.shape[0]
    color = get_param(c)

    fig, Axes = plt.subplots(nrows=c, ncols=2, sharex='all', figsize=(32, 20))

    for i, channels in enumerate(pre_train.transformer.choose_channels):
        axes = Axes[i][0]
        print(patch_epochs_mask[i])
        axes.plot(range(3000), patch_epochs_mask[i][:3000].detach().numpy(), color[-2])
        # axes.grid(True)
        axes.set_title(names[i] + ' ' + format(loss[1][i].item(), '.3f'))
        axes = Axes[i][1]
        axes.plot(range(3000), masked_time[i].detach().numpy(), color[-2])
        axes.plot(range(3000), patch_epochs[i].detach().numpy(), 'r', alpha=0.2)
        axes.set_title(names[i])
    path = '/'.join(_config['load_path'].split('/')[-4:-2])
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
    os.makedirs(f"/home/cuizaixu_lab/huangweixuan/Sleep/{path}/{_config['datasets'][0]}/epoch_{match}", exist_ok=True)
    plt.savefig(f"/home/cuizaixu_lab/huangweixuan/Sleep/{path}/{_config['datasets'][0]}/epoch_{match}/predict_fft_nu_{id}.svg",  format='svg')
    plt.close("all")
    return min_idx

