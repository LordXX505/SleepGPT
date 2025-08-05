import os
from multiprocessing import Process, current_process
from multiprocessing import Pool
from tqdm import tqdm
import h5py
import gc
import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import xml.etree.ElementTree as ET
import numpy as np
import mne
from threading import Thread
import multiprocessing
import time
import glob as glob
import sys
from sklearn.linear_model import LinearRegression

import concurrent.futures
import logging
import torch
import zlib
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


def write_data(h5file, data_dict, compression_opts=5):
    with h5py.File(h5file, 'w') as f:
        for name, data in data_dict.items():
            f.create_dataset(name, data=data, compression="gzip", compression_opts=compression_opts)
def check_hdf5_file(file_path):
    if os.path.isfile(file_path) is False:
        return True
    try:
        with h5py.File(file_path, 'r') as hf:
            for key in hf.keys():
                try:
                    data = hf[key][:]
                except Exception as e:
                    # error_files.put(file_path)
                    print(f"Error reading dataset {key}: {e}, file_path: {file_path}")
        return True
    except Exception as e:
        # error_files.put(file_path)
        print(f"Error opening file {file_path}: {e}, file_path: {file_path}")
        return False
required_channels = ['C3', 'C4', 'EMG1', 'EOG1', 'EOG2', 'ECG', 'AIRFLOW', 'ABD']
res_channels = ['C3', 'C4', 'EMG1', 'EOG1', 'ECG', 'AIRFLOW', 'ABD']
def process(idx, path_list, anno_list, process_idx, local_test=False):
    if local_test is True:
        Root_path = '/Users/hwx_admin/Downloads/'
        log_filename = './bdsp.log'
        store_path = ''
    else:
        Root_path = '/data/SHHS/'
        os.makedirs('/mnt/LOG_FILE/LOG_SHHS/', exist_ok=True)
        log_filename = f'/mnt/LOG_FILE/LOG_SHHS/process_{process_idx}.log'
        store_path = '/mnt/myvol/data/shhs'
    logger = setup_logger(log_filename)
    start_time = time.time()
    logger.info(f"Process {process_idx} started with {len(path_list)} files.")


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
        success_file = f"{store_path}/{name}/success"
        # if check_hdf5_file('/mnt/myvol/data/shhs/shhs1-204664/data.h5') is True:
        #     continue
        if check_hdf5_file(os.path.join(store_path, name, 'data.h5')) is True:
            continue
        # if os.path.exists(success_file):
        #     logger.info(f"Skipping {name}, already processed.")
        #     continue
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
        rename_dict = {'EEG': 'C4', 'ABDO RES': 'ABD', EEG2_name: 'C3', 'EMG': 'EMG1', 'EOG(L)': 'EOG1',
                       'EOG(R)': 'EOG2'}
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
        labels = np.asarray(labels)
        logger.info('Remove movement and unknown stages if any')
        data = raw.get_data()
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
        data_dict = {
            'signal': x,
            'stage': y,
            'good_channels': good_channels
        }
        write_data(os.path.join(store_path, name, 'data.h5'), data_dict, 2)
        end_time = time.time()
        logger.info(f"Process {process_idx} finished. Total time: {end_time - start_time} seconds.")
        # / mnt / myvol / data / shhs / shhs2 - 203742 / data.h5

        with open(success_file, 'w') as f:
                f.write('success')
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Logging channel sums and counts:")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python process_shhs.py <process_id>")
        sys.exit(1)
    current_directory = os.getcwd()
    print(current_directory)
    procs_number = 64
    process_id = int(sys.argv[1])
    Root_path = '/data/shhs_raw/polysomnography/edfs'
    shhs1_root_path = os.path.join(Root_path, 'shhs1')
    shhs2_root_path = os.path.join(Root_path, 'shhs2')
    shhs1_data_path_list = sorted(glob.glob(shhs1_root_path+'/*'))
    shhs2_data_path_list = sorted(glob.glob(shhs2_root_path+'/*'))
    all_data_path_list = shhs1_data_path_list + shhs2_data_path_list
    print(len(all_data_path_list))
    anna_root_path = '/data/shhs_raw/polysomnography/annotations-events-profusion'
    anna_shhs1_path = os.path.join(anna_root_path, 'shhs1')
    anna_shhs2_path = os.path.join(anna_root_path, 'shhs2')
    anna_shhs1_path_list = sorted(glob.glob(anna_shhs1_path+'/*'))
    anna_shhs2_path_list = sorted(glob.glob(anna_shhs2_path+'/*'))
    all_anno_path_list = anna_shhs1_path_list + anna_shhs2_path_list

    if current_directory.split('/')[1] == 'Users':
        local = True
        start = 55
        end = 57
        all_text = pd.read_csv('/Users/hwx_admin/Downloads/bdsp_psg_master_20231101.csv')
    else:
        local = False
        piece = len(all_data_path_list) // procs_number + 1
        print(piece)
        start = process_id * piece
        end = min((process_id + 1) * piece, len(all_data_path_list))

    process(process_id % 8, all_data_path_list[start:end], all_anno_path_list[start:end], process_id, local_test=False)
