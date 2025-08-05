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
from bdsp_sleep_functions import load_bdsp_signal, annotations_preprocess, vectorize_respiratory_events, \
    vectorize_sleep_stages, vectorize_arousals, vectorize_limb_movements
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
def process(idx, path_list, process_idx, local_test=False):
    if local_test is True:
        Root_path = '/Users/hwx_admin/Downloads/'
        log_filename = './bdsp.log'
        store_path = ''
    else:
        Root_path = '/data/MGH/'
        os.makedirs('/mnt/LOG_FILE/LOG_BDSP/', exist_ok=True)
        log_filename = f'/mnt/LOG_FILE/LOG_BDSP/process_{process_idx}.log'
        store_path = '/mnt/myvol/data/MGH_NEW'
    logger = setup_logger(log_filename)
    start_time = time.time()
    logger.info(f"Process {process_idx} started with {len(path_list)} files.")
    # try:
    #     mne.utils.set_config('MNE_USE_CUDA', "true")
    #     mne.cuda.set_cuda_device(idx)
    #     print('set cuda')
    #     n_jobs = 'cuda'
    # except Exception as e:
    #     logger.error(f"CUDA initialization failed: {e}")
    #     mne.utils.set_config('MNE_USE_CUDA', "false")
    #     n_jobs = 24

    for i in tqdm(range(0, len(path_list))):

        items = path_list.iloc[i]
        BDSPPatientID = items['BDSPPatientID']
        BidsFolder = items['BidsFolder']
        SessionID = str(items['SessionID'])
        HasAnnotations = items['HasAnnotations']
        HasStaging = items['HasStaging']
        filename = os.path.join(store_path, BidsFolder, 'ses-' + str(SessionID))
        os.makedirs(filename, exist_ok=True)
        success_file = f'{filename}/success'
        # if os.path.exists(success_file):
        #     logger.info(f"Skipping {filename}, already processed.")
        #     continue
        anno_root_path = os.path.join(Root_path, BidsFolder)
        data_path = os.path.join(Root_path, BidsFolder, 'ses-' + str(SessionID), "eeg", '_'.join(
                                                             [BidsFolder, 'ses-' + SessionID, 'task-psg', 'eeg.edf']))

        # print(f'data_path : {data_path}')
        # if check_hdf5_file(os.path.join(filename, 'data.h5')) is True:
        #     continue
        try:
            signal, params, good_channels = load_bdsp_signal(data_path, n_jobs=-1)
            if HasAnnotations == 'Y':
                path_annotations = os.path.join(anno_root_path, 'ses-' + str(SessionID), 'eeg', '_'.join([BidsFolder, 'ses-' + SessionID, 'task-psg', 'annotations.csv']))
                fs_original = 100
                signal_bs = np.copy(signal)
                annotations = pd.read_csv(path_annotations)
                annotations, annotations_quality = annotations_preprocess(annotations, fs_original, return_quality=True)
                signal_len = signal_bs.shape[1]
                resp = vectorize_respiratory_events(annotations, signal_len)
                stage = vectorize_sleep_stages(annotations, signal_len)
                arousal = vectorize_arousals(annotations, signal_len)
                idx_stages = np.where(~np.isnan(stage))[0]
                stage_start = idx_stages[0]
                stage_end = idx_stages[-1] + 1
                durations = stage_end-stage_start
                assert durations % 3000 == 0, f'durations wrong {durations}'

                n_epochs = durations // 3000
                stage = np.array(np.split(stage[stage_start:stage_end], n_epochs))[:, 0]
                select_idx = np.where(~np.isnan(stage))[0]
                logger.info(f'removed before: {len(stage)}, after: {len(select_idx)}')
                stage = stage[select_idx]
                signal_bs = np.array(np.split(signal_bs[:, stage_start:stage_end], n_epochs, axis=1))[select_idx]
                resp = np.array(np.split(resp[stage_start:stage_end], n_epochs))[select_idx]
                arousal = np.array(np.split(arousal[stage_start:stage_end], n_epochs))[select_idx]

                if store_path == '':
                    continue

                h5_filename = f"{filename}/data.h5"
                data_dict = {
                    'signal': signal_bs,
                    'resp': resp,
                    'arousal': arousal,
                    'stage': stage,
                    'good_channels': good_channels
                }

                h5_filename = f"{filename}/data.h5"
                write_data(h5_filename, data_dict, 2)
                with open(success_file, 'w') as f:
                        f.write('success')
                del signal_bs
                gc.collect()
                torch.cuda.empty_cache()
            else:
                signal_len = signal.shape[1]
                n_epochs = signal_len // 3000
                signal = np.array(np.split(signal[:, n_epochs*3000], n_epochs, axis=1))
                data_dict = {
                    'signal': signal,
                    'good_channels': good_channels
                }
                h5_filename = f"{filename}/data.h5"
                write_data(h5_filename, data_dict, 2)

                with open(success_file, 'w') as f:
                    f.write('success')
                del signal
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'=================={BidsFolder}-{SessionID} is Wrong, Exception is {e}==================')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_shhs.py <process_id> <resume>")
        sys.exit(1)
    current_directory = os.getcwd()
    print(current_directory)
    procs = []
    procs_number = 96
    result = []
    process_id = int(sys.argv[1])
    resume = 'false'
    if len(sys.argv) > 2:
        resume = str(sys.argv)
    if current_directory.split('/')[1] == 'Users':
        local = True
        start = 55
        end = 57
        all_text = pd.read_csv('/Users/hwx_admin/Downloads/bdsp_psg_master_20231101.csv')

    else:
        local = False
        if resume == 'false':
            all_text = pd.read_csv('/data/MGH/bdsp_psg_master_20231101.csv')[0:10000]
            piece = len(all_text) // procs_number + 1
            print(piece)
            start = 1 + process_id * piece
            end = min(1 + (process_id + 1) * piece, len(all_text))
        else:
            all_text = pd.read_csv('/data/MGH/bdsp_psg_master_20231101.csv')[0:10000]
            store_path = '/mnt/myvol/data/MGH'
            success = []
            for i in tqdm(range(0, len(all_text))):
                items = all_text.iloc[i]
                BDSPPatientID = items['BDSPPatientID']
                BidsFolder = items['BidsFolder']
                SessionID = str(items['SessionID'])
                HasAnnotations = items['HasAnnotations']
                HasStaging = items['HasStaging']
                filename = os.path.join(store_path, BidsFolder, 'ses-' + str(SessionID))
                if os.path.exists(os.path.join(filename,'success')) is True:
                    success.append(os.path.join(filename,'success'))
            # 收集所有要删除的条件
            delete_conditions = []
            for file_sc in success:
                bdsid = file_sc.split('/')[-3]
                sessionid = int(file_sc.split('/')[-2].split('-')[-1])
                delete_conditions.append((bdsid, sessionid))
            if delete_conditions:
                delete_conditions_df = pd.DataFrame(delete_conditions, columns=['BidsFolder', 'SessionID'])
                delete_conditions_df['BidsFolder'] = delete_conditions_df['BidsFolder'].astype(str)
                delete_conditions_df['SessionID'] = delete_conditions_df['SessionID'].astype(int)
                all_text['BidsFolder'] = all_text['BidsFolder'].astype(str)
                all_text['SessionID'] = all_text['SessionID'].astype(int)
                # 创建一个布尔索引，标记需要删除的行
                delete_index = all_text.apply(
                    lambda row: any((row['BidsFolder'] == delete_conditions_df['BidsFolder']) &
                                    (row['SessionID'] == delete_conditions_df['SessionID'])), axis=1)

                # 删除需要删除的行
                all_text = all_text[~delete_index]
            piece = len(all_text) // procs_number + 1
            start = process_id * piece
            end = min((process_id + 1) * piece, len(all_text))

    process(process_id % 8, all_text[start:end], process_id, local)