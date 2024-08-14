import os
from multiprocessing import Process, current_process
from multiprocessing import Pool
from tqdm import tqdm
import pyarrow as pa
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-log'
import logging
import torch


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


def process(idx, path_list, process_idx, local_test=False):
    if local_test is True:
        Root_path = '/Users/hwx_admin/Downloads/'
        log_filename = './bdsp.log'
        store_path = ''
    else:
        Root_path = '/data/MGH/'
        os.makedirs('/mnt/LOG_FILE/LOG_BDSP/', exist_ok=True)
        log_filename = f'/mnt/LOG_FILE/LOG_BDSP/process_{process_idx}.log'
        store_path = '/mnt/myvol/data/MGH'
    logger = setup_logger(log_filename)
    start_time = time.time()
    logger.info(f"Process {process_idx} started with {len(path_list)} files.")
    try:
        mne.utils.set_config('MNE_USE_CUDA', "true")
        mne.cuda.set_cuda_device(idx)
        print('set cuda')
        n_jobs = 'cuda'
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        mne.utils.set_config('MNE_USE_CUDA', "false")
        n_jobs = 24

    for i in tqdm(range(0, len(path_list))):

        items = path_list.iloc[i]
        BDSPPatientID = items['BDSPPatientID']
        ## degub
        BidsFolder = items['BidsFolder']
        SessionID = str(items['SessionID'])
        HasAnnotations = items['HasAnnotations']
        HasStaging = items['HasStaging']
        filename = os.path.join(store_path, BidsFolder, 'ses-' + str(SessionID))
        os.makedirs(filename, exist_ok=True)
        success_file = f'{filename}/success'
        if os.path.exists(success_file):
            logger.info(f"Skipping {filename}, already processed.")
            continue
        anno_root_path = os.path.join(Root_path, BidsFolder)
        data_path = os.path.join(Root_path, BidsFolder, 'ses-' + str(SessionID), "eeg", '_'.join(
                                                             [BidsFolder, 'ses-' + SessionID, 'task-psg', 'eeg.edf']))
        print(f'data_path : {data_path}')
        try:
            signal, params, good_channels = load_bdsp_signal(data_path, n_jobs=n_jobs)
            if HasAnnotations == 'Y':
                path_annotations = os.path.join(anno_root_path, 'ses-' + str(SessionID), 'eeg', '_'.join([BidsFolder, 'ses-' + SessionID, 'task-psg', 'annotations.csv']))
                fs_original = 100
                signal_bs = np.copy(signal)
                annotations = pd.read_csv(path_annotations)
                annotations, annotations_quality = annotations_preprocess(annotations, fs_original, return_quality=True)
                signal_len = signal_bs.shape[1]
                resp = vectorize_respiratory_events(annotations, signal_len)
                # Vectorize the sleep stages from the annotations
                stage = vectorize_sleep_stages(annotations, signal_len)
                # Vectorize the arousals from the annotations
                arousal = vectorize_arousals(annotations, signal_len)
                idx_stages = np.where(~np.isnan(stage))[0]
                stage_start = idx_stages[0]
                stage_end = idx_stages[-1] + 1
                durations = stage_end-stage_start
                assert durations%3000 == 0, f'durations wrong {durations}'

                n_epochs = durations//3000
                stage = np.array(np.split(stage[stage_start:stage_end], n_epochs))[:, 0]
                select_idx = np.where(~np.isnan(stage))[0]
                logger.info(f'removed before: {len(stage)}, after: {len(select_idx)}')
                stage = stage[select_idx]
                signal_bs = np.array(np.split(signal_bs[:, stage_start:stage_end], n_epochs, axis=1))[select_idx]
                resp = np.array(np.split(resp[stage_start:stage_end], n_epochs))[select_idx]
                arousal = np.array(np.split(arousal[stage_start:stage_end], n_epochs))[select_idx]
                cnt = 0
                if store_path == '':
                    continue

                for _signal, _resp, _arousal, _stage in zip(signal_bs, resp, arousal, stage):
                    dataframe = pd.DataFrame(
                        {'x': [_signal.tolist()], 'stage': _stage,
                         'resp': [_resp.tolist()], 'arousal': [_arousal.tolist()],
                         'good': [good_channels.tolist()]},
                    )
                    table = pa.Table.from_pandas(dataframe)
                    with pa.OSFile(
                            f"{filename}/{str(cnt).zfill(5)}.arrow", "wb"
                    ) as sink:
                        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                            writer.write_table(table)
                    cnt += 1
                    del dataframe
                    del table
                    gc.collect()
                del signal_bs
                gc.collect()
                torch.cuda.empty_cache()
                with open(f"{success_file}", 'w') as f:
                    f.write('success')
            else:
                cnt = 0
                signal_len = signal.shape[1]
                n_epochs = signal//3000
                signal = np.array(np.split(signal, n_epochs, axis=1))
                filename = os.path.join(store_path, BidsFolder, 'ses-' + str(SessionID))
                os.makedirs(filename, exist_ok=True)
                for _signal in signal:
                    dataframe = pd.DataFrame(
                        {'x': [_signal.tolist()],
                         'good': [good_channels.tolist()]},
                    )
                    table = pa.Table.from_pandas(dataframe)
                    with pa.OSFile(
                            f"{filename}/{str(cnt).zfill(5)}.arrow", "wb"
                    ) as sink:
                        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                            writer.write_table(table)
                    cnt += 1
                    del dataframe
                    del table
                    gc.collect()
                del signal
                gc.collect()
                torch.cuda.empty_cache()

                with open(f"{success_file}", 'w') as f:
                    f.write('success')

        except Exception as e:
            print(f'=================={BidsFolder}-{SessionID} is Wrong, Exception is {e}==================')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python process_shhs.py <process_id>")
        sys.exit(1)
    current_directory = os.getcwd()
    print(current_directory)
    procs = []
    procs_number = 8
    result = []
    process_id = int(sys.argv[1])
    if current_directory.split('/')[1] == 'Users':
        local = True
        start = 55
        end = 57
        all_text = pd.read_csv('/Users/hwx_admin/Downloads/bdsp_psg_master_20231101.csv')

    else:
        local = False
        all_text = pd.read_csv('/data/MGH/bdsp_psg_master_20231101.csv')[6000:7000]
        piece = len(all_text) // procs_number + 1
        print(piece)
        start = 1 + process_id * piece
        end = min(1 + (process_id + 1) * piece, len(all_text))


    process(process_id % 8, all_text[start:end], process_id, local)
