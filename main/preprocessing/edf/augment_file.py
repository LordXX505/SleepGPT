import mne
import numpy as np
import pyarrow as pa
import os
import glob
import pandas as pd
import gc
import torch
import os
import sys
import matplotlib.patches as patches
import pywt  # 用于小波变换

sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')
from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules.backbone import Model
from main.transforms import unnormalize
import pytorch_lightning as pl
from main.config import ex
import matplotlib.pyplot as plt
from typing import List
import torch
import re
from main.transforms import FFT_Transform
from skimage.metrics import structural_similarity as ssim


def get_epochs(data):
    try:
        x = np.array(data.as_py())
    except:
        x = np.array(data.to_pylist())
    x = x * 1e6
    x = torch.from_numpy(x).float()
    return {'x': x}


def get_stage(data):
    return {'Stage_label': torch.from_numpy(np.array(data)).long()}


def random_plot_regenerate(origin, new, cnt, mode, sub=None, ms_ssim_values=0):
    # if torch.rand(1) > 0.5:
    #     return
    fig = plt.figure(figsize=(20, 20))

    for c in range(8):
        axes = fig.add_subplot(8, 2, c * 2 + 1)
        axes.plot(origin[c])

    for c in range(8):
        axes = fig.add_subplot(8, 2, (c + 1) * 2)
        axes.plot(new[c])
    fig.suptitle(f'ms_ssim_values={ms_ssim_values}')
    if sub is not None:
        try:
            os.makedirs(f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{sub.split('/')[-1]}/EDF/epoch_{cnt}",
                        exist_ok=True)
            plt.savefig(
                f"/home/cuizaixu_lab/huangweixuan/Sleep/result/{sub.split('/')[-1]}/EDF/epoch_{cnt}/predict_{mode}.svg",
                format='svg')
        except:
            print('save figure failed')
    plt.show()
    plt.close('all')


def save_epoch(save_epochs, save_labels, filename, name, cnt):
    dataframe = pd.DataFrame(
        {'x': [save_epochs.tolist()], 'stage': save_labels}
    )
    table = pa.Table.from_pandas(dataframe)
    os.makedirs(f"{filename}/{name}/", exist_ok=True)
    print(f"save path: {filename}/{name}/{str(cnt).zfill(5)}.arrow, stage:{save_labels}")
    with pa.OSFile(
            f"{filename}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    del dataframe
    del table


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        raise NotImplementedError


def clone_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.clone()
    elif isinstance(batch, dict):
        return {k: clone_batch(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [clone_batch(v) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(clone_batch(v) for v in batch)
    else:
        raise NotImplementedError


def calculate_ms_ssim(real_data, generated_data):
    scales = 5
    ms_ssim_scores = []

    for real, generated in zip([real_data], [generated_data]):
        real_coeffs = pywt.wavedec(real, 'db1', level=scales)
        generated_coeffs = pywt.wavedec(generated, 'db1', level=scales)
        ssim_scores = [ssim(real_c, gen_c, data_range=real_c.max()-real_c.min()) for real_c, gen_c in
                       zip(real_coeffs, generated_coeffs)]
        ms_ssim_scores.append(np.mean(ssim_scores))
    return np.mean(ms_ssim_scores)


def consecutive_generate(batch_orig, model, batch_first, device, max_time=20,
                         start_index=0, epoch_start=0, sub='consecutive', new_cnt=0):
    # if torch.rand(1) < 0.001:
    #     plot_flag = True
    # else:
    #     plot_flag = False
    batch = clone_batch(batch_first)
    res_batch = []
    res_ms_ssim_mean = []
    for time_stamp_start in range(0, 7):
        now_batch = clone_batch(batch)
        now_batch['mask'][0][:, 2] = 0
        now_batch['random_mask'][0][0] = torch.zeros(120)
        for c in [3, 5, 7]:
            if time_stamp_start % 2 == 0:
                now_batch['random_mask'][0][0, (c * 15 + 8):(c * 15 + 15)] = torch.ones(7, device=device)
            else:
                now_batch['random_mask'][0][0, (c * 15 + 7):(c * 15 + 15)] = torch.ones(8, device=device)

        with torch.no_grad():
            res = model(now_batch, stage='test', aug_fft=True)
        generate_epoch = res['cls_feats'] * res['time_mask_patch'].unsqueeze(-1) + \
                         model.patchify(res['batch']['epochs'][0]) * (
                                 1 - res['time_mask_patch']).unsqueeze(-1)
        generate_epoch = model.unpatchify(generate_epoch).detach().clone()[0]
        mask_matrix = torch.zeros((8, 3000))
        batch_orig_index = (int(time_stamp_start / 2) + 1) + start_index
        if time_stamp_start % 2 == 0:
            if time_stamp_start != 6:
                batch['epochs'][0][0] = torch.cat([generate_epoch[:, 1600:],
                                                   batch_orig['epochs'][0][batch_orig_index, :, :1600]], dim=-1)
        else:
            batch['epochs'][0][0] = torch.cat([generate_epoch[:, 1400:],
                                               batch_orig['epochs'][0][batch_orig_index, :, 1600:]], dim=-1)
        if time_stamp_start % 2 == 0:
            res_batch.append(clone_batch(generate_epoch))
            print(time_stamp_start / 2)
            plot_orig = batch_orig['epochs'][0][int(time_stamp_start / 2) + start_index]
            ms_ssim_values = [calculate_ms_ssim(plot_orig[i, :].detach().cpu(), generate_epoch[i, :].detach().cpu()) for
                              i in [3, 5, 7]]
            ms_ssim_mean = int(np.mean(ms_ssim_values))
            res_ms_ssim_mean.append(ms_ssim_mean)
            if torch.rand(1) < 0.01:
                random_plot_regenerate(plot_orig.detach().cpu(), generate_epoch.detach().cpu(),
                                       cnt=(epoch_start + batch_orig_index, new_cnt + batch_orig_index),
                                       mode='consecutive', sub=sub,
                                       ms_ssim_values=ms_ssim_mean)

    return res_batch, np.mean(res_ms_ssim_mean)


@ex.automain
def main(_config):
    print(_config)
    pre_train = Model(_config)
    pl.seed_everything(512)
    pre_train.set_task()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_train.to(device)
    dm = MultiDataModule(_config, kfold=0)
    dm.setup(stage='test')
    collate = dm.collate
    path = '/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed'
    store_path = '/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed'
    # path = '/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/processed/'
    batch_size = 20
    generate_size = 4
    generate_patches = int(batch_size / generate_size)
    edf_items = sorted(glob.glob(os.path.join(path, '*')))
    partition = int(len(edf_items) / 2) + 1
    partition_idx = 0
    resume_flag = False
    save_ms_ssim_mean_res = {}
    for edf_items_index, sub in enumerate(edf_items[(partition_idx * partition):((partition_idx + 1) * partition)]):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            #SC4452F0, SC4822G0
            # if base_name == 'SC4452F0':
            #     resume_flag = True
            # if resume_flag is False:
            #     continue
            sub_arr_items = sorted(glob.glob(os.path.join(sub, '*')))
            cnt = 0
            save_ms_ssim_mean_res[base_name] = {}
            res_ms_ssim_mean_res = {}
            for random_times in range(0, 50):
                # start_index = 200
                start_index = torch.randint(0, len(sub_arr_items) - batch_size, (1,))
                print(f'random start: {start_index}')
                batch_list = []
                label_list = []
                for sub_arr_index, sub_arr in enumerate(sub_arr_items[start_index:start_index + batch_size]):
                    print(
                        f'===========begin: {base_name}===========cnt:{start_index + sub_arr_index}==========={sub_arr}=========')
                    tables = pa.ipc.RecordBatchFileReader(
                        pa.memory_map(sub_arr, "r")
                    ).read_all()

                    epoch = get_epochs(tables['x'][0])['x']
                    stage = get_stage(tables['stage'])['Stage_label']
                    label_list.append(stage)
                    # if stage[0] == 0:
                    #     continue
                    batch_orig = {'stage': stage.detach().clone().unsqueeze(0),
                                  'x': (epoch, torch.tensor([16, 18, 36, 52])),
                                  'index': torch.tensor(1)}  # index is no use
                    batch_list.append(batch_orig)
                print(label_list)

                batch_cpu = collate(batch_list)

                batch_cuda = move_to_device(batch_cpu, device)

                batch = clone_batch(batch_cuda)
                res_generate_epoch = []
                res_ms_ssim_mean_patches = []
                for p in range(generate_patches):
                    print(f'aug start idx: {p * generate_size}')
                    batch_cpu_first = collate([batch_list[p * generate_size]])

                    batch_cuda_first = move_to_device(batch_cpu_first, device)

                    batch_first = clone_batch(batch_cuda_first)
                    consecutive_generate_epochs, res_ms_ssim_mean = consecutive_generate(batch, model=pre_train,
                                                                                         batch_first=batch_first,
                                                                                         device=device,
                                                                                         start_index=p * generate_size,
                                                                                         epoch_start=start_index.item(),
                                                                                         new_cnt=cnt + p * generate_size)
                    res_generate_epoch += consecutive_generate_epochs
                    res_ms_ssim_mean_patches.append(res_ms_ssim_mean)
                res_ms_ssim_mean_res[cnt] = np.mean(res_ms_ssim_mean_patches)
                for sub_arr_index, sub_arr in enumerate(sub_arr_items[start_index:start_index + batch_size]):
                    # random_plot_regenerate(origin[sub_arr_index].detach().cpu().numpy(), generate_epoch[sub_arr_index].detach().cpu().numpy(),
                    #                        sub=base_name, cnt=sub_arr_index, mode='Aug_file')
                    save_epoch(res_generate_epoch[sub_arr_index].detach().cpu().numpy(),
                               label_list[sub_arr_index].detach().cpu().numpy()[0],
                               filename=os.path.join('/', *store_path.split('/')[:-1], 'Aug_consecutive'),
                               name=base_name
                               , cnt=cnt)
                    cnt += 1
            save_ms_ssim_mean_res[base_name] = res_ms_ssim_mean_res

            assert cnt == 1000, f'cnt: {cnt}'
    torch.save(save_ms_ssim_mean_res, os.path.join('/', *store_path.split('/')[:-1], 'Aug_consecutive'))
    print('------------------all finished------------------')
