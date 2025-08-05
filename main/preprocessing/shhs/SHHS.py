import time
import os
import sys
from multiprocessing import Process, current_process
from multiprocessing import Pool
from tqdm import tqdm
import pyarrow as pa
import gc
import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import xml.etree.ElementTree as ET
import torch
import numpy as np
import mne
from threading import Thread
import multiprocessing
import glob as glob
bads_idx = {
    'C3': 4,
    'C4': 5,
    'ECG': 15,
    'EMG1': 16,
    'EOG1': 18
}
# self.choose_channels = np.array([4, 5, 15, 16, 18, 22, 23, 36, 38, 39, 52])
# [C3, C4, ECG, EMG, EOG, F3, F4, Fpz, O1, O2, Pz]
required_channels = ['C3', 'C4', 'EMG1', 'EOG1', 'EOG2', 'ECG', 'AIRFLOW', 'ABD']
res_channels = ['C3', 'C4', 'EMG1', 'EOG1', 'ECG', 'AIRFLOW', 'ABD']

import  logging

def setup_logger(log_filename):
    """Sets up the logger with the given log filename."""
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
def process_shhs(idx, path_list, anno_list, process_idx):
    log_filename = ''
    logger = setup_logger(log_filename)
    start_time = time.time()
    logger.info(f"Process {process_idx} started with {len(path_list)} files.")
    assert len(path_list) == len(anno_list), f'{len(path_list)}, {len(anno_list)}'

    print_ = False
    sucess = []
    wrong = []
    channel_sum_dict = {}
    channel_count_dict = {}

    for i in tqdm(range(len(anno_list))):
        data_path = path_list[i]
        anno_path = anno_list[i]
        good_channels = np.ones(len(res_channels))
        name = os.path.basename(data_path).split('.')[0]
        anno_name = os.path.basename(anno_path).split('.')[0]
        assert name.split('-')[-1] == anno_name.split('-')[1], f"Mismatch in filenames: {name} != {anno_name}"
        logger.info(f"********************starting {name}********************")
        filename = '/mnt/myvol/data/shhs'
        os.makedirs(f"{filename}/{name}", exist_ok=True)
        logger.info(f"{filename}/{name}")
        labels = []
        # Check if success file exists
        success_file = f"{filename}/{name}/success"
        if os.path.exists(success_file):
            logger.info(f"Skipping {name}, already processed.")
            continue
        # Read annotation and its header
        t = ET.parse(anno_path)
        r = t.getroot()
        faulty_File = 0
        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:  # Assign label 4 for REM stage
                labels.append(4)
            else:
                labels.append(lbl)
            if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
                faulty_File = 1

        if faulty_File == 1:
            logger.warning(f"============================== Faulty file: {name} ==================")
            continue
        ########################################################################################################

        raw = mne.io.read_raw_edf(data_path, preload=True)
        logger.info(f'reading data finished')
        # print(raw.info)
        channle_names = ['EEG2', 'EEG 2', 'EEG(SEC)', 'EEG sec', 'EEG(sec)']
        EEG2_name = ""
        for i in channle_names:
            if i in raw.ch_names:
                EEG2_name = i
                break
        logger.info(f'chnames: {raw.ch_names}')
        rename_dict = {'EEG': 'C4', 'ABDO RES': 'ABD', EEG2_name: 'C3', 'EMG': 'EMG1', 'EOG(L)': 'EOG1', 'EOG(R)': 'EOG2'}
        existing_channels = {old_name: new_name for old_name, new_name in rename_dict.items() if
                             old_name in raw.ch_names}
        raw.rename_channels(existing_channels)
        logger.info(f'renaming')
        for ch in required_channels:
            if ch not in raw.ch_names:
                raw.add_channels(
                    [mne.io.RawArray(np.zeros((1, len(raw.times))), mne.create_info([ch], raw.info['sfreq']))])
        raw.pick(required_channels)
        def func(array, EOG):
            array -= EOG[0]
            return array
        if 'EOG2' in raw.ch_names:
            data, times = raw['EOG2']
            raw.apply_function(func, picks=['EOG1'], EOG=data)
            raw.drop_channels(['EOG2'])

        new_ch_names = sorted(raw.ch_names)
        raw = raw.reorder_channels(new_ch_names)
        for c_index, _c in enumerate(res_channels):
            if (_c not in raw.ch_names) or (np.all(raw.get_data(picks=[_c]) == 0)):
                good_channels[c_index] = 0
        raw.resample(100)  # down sample to 100hz
        raw._data *= 1e6  # uv
        logger.info('down sample to 100hz')
        logger.info(f'good_channels : {good_channels}')

        raw.filter(l_freq=0.3, h_freq=35, method='fir')
        bads = raw.info['bads']
        badsidx = [bads_idx[_] for _ in bads]
        badsidx = sorted(badsidx)
        logger.info(f'{raw.info["bads"]}, idx: {badsidx}')
        labels = np.asarray(labels)

        logger.info('Remove movement and unknown stages if any')
        if not print_:
            logger.info(raw.ch_names)

        data = raw.get_data()
        channel_means = np.mean(data, axis=1)
        for ch_name, mean_value in zip(raw.ch_names, channel_means):
            if ch_name not in channel_sum_dict:
                channel_sum_dict[ch_name] = 0
                channel_count_dict[ch_name] = 0
            channel_sum_dict[ch_name] += mean_value
            channel_count_dict[ch_name] += 1

        n_epochs = data.shape[1] / (30 * 100)
        if data.shape[1] % (30 * 100) != 0:
            raise Exception("Something wrong")
        x = np.asarray(np.split(data, n_epochs, axis=1)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y), f'y: {len(y)}, x: {len(x)}'

        w_edge_mins = 30
        nw_idx = np.where(y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        x = x[select_idx]
        y = y[select_idx]

        cnt = 0

        for _x, _y in zip(x, y):
            dataframe = pd.DataFrame(
                    {'x': [_x.tolist()], 'stage': _y, 'bads': [bads],
                     'good': [good_channels.tolist()]}
                )
            table = pa.Table.from_pandas(dataframe)
            with pa.OSFile(
                        f"{filename}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
                ) as sink:
                    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)
            cnt += 1
            del dataframe
            del table
            gc.collect()
        del raw
        gc.collect()
        with open(f"{success_file}/", 'w') as f:
            f.write('success')
        sucess.append(name)
        print_ = True
    logger.info("Logging channel sums and counts:")
    for ch_name in channel_sum_dict.keys():
        logger.info(
            f"Channel {ch_name} - Sum Mean: {channel_sum_dict[ch_name]}, Numbers: {channel_count_dict[ch_name]}")
    end_time = time.time()
    logger.info(f"Process {process_idx} finished. Total time: {end_time - start_time} seconds.")
    return (sucess, wrong, channel_sum_dict, channel_count_dict)

if __name__ == '__main__':
    process_shhs(0,['/data/shhs_raw/polysomnography/edfs/shhs1/shhs1-204355.edf'],
                 ['/data/shhs_raw/polysomnography/annotations-events-profusion/shhs1/shhs1-204355-profusion.xml'],0)