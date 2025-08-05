import os
from tqdm import tqdm
import h5py
import gc
import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import xml.etree.ElementTree as ET
import numpy as np
import mne
import time
import glob as glob
import sys
import tkinter as tk
from tkinter import filedialog
import logging
import torch
import re


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


def parse_remlogic_report(file_path, store_path):
    if not file_path:
        print("No file selected.")
        return

    print("Reading file...")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Initialize variables
    start = False
    hyp = []
    timevector = []
    duration = []
    type_ar = []
    h = []
    m = []
    s = []

    for tline in lines:
        tline = tline.strip()
        if len(tline) > 10:
            if tline.startswith("Sleep Stage"):
                start = True
                k = 0  # hyp counter
                j = 0  # CAP counter
                continue

            if start:
                p = [pos for pos, char in enumerate(tline) if char in [":", "."]]

                if len(p) < 2:
                    continue

                if "SL" in tline[p[1] + 4:p[1] + 6]:  # Sleep stage
                    k += 1
                    if "E" in tline[p[1] + 11]:  # REM
                        hyp.append([4])
                    elif "T" in tline[p[1] + 11]:  # MT
                        hyp.append([7])
                        print("MT")
                    else:
                        stage = int(tline[p[1] + 11])
                        if stage == 4:
                            stage = 3
                        hyp.append([stage])  # Sleep stage 0, 1, 2, 3

                    h.append(int(tline[p[0] - 2:p[0]]))
                    m.append(int(tline[p[0] + 1:p[0] + 3]))
                    s.append(int(tline[p[1] + 1:p[1] + 3]))

                    if h[-1] < 10:
                        hyp[-1].append((h[-1] + 24) * 3600 + m[-1] * 60 + s[-1])
                    else:
                        hyp[-1].append(h[-1] * 3600 + m[-1] * 60 + s[-1])

                elif "MC" in tline[p[1] + 4:p[1] + 6]:  # CAP A phase
                    j += 1
                    t = tline.find("-")
                    type_ar.append(int(tline[t + 2]))
                    duration.append(int(tline[t + 4:t + 6]))

                    hCAP = int(tline[p[0] - 2:p[0]])
                    mCAP = int(tline[p[0] + 1:p[0] + 3])
                    sCAP = int(tline[p[1] + 1:p[1] + 3])

                    if hCAP < 10:
                        timevector.append((hCAP + 24) * 3600 + mCAP * 60 + sCAP)
                    else:
                        timevector.append(hCAP * 3600 + mCAP * 60 + sCAP)

    # Process hypnogram
    print("Processing the hypnogram")
    hyp = np.array(hyp)
    jump = []
    nj = []
    x = 0
    for j in range(len(hyp) - 1):
        if hyp[j + 1, 1] - hyp[j, 1] > 30:
            jump.append(j)
            diff = hyp[j + 1, 1] - hyp[j, 1]
            nj.append(diff // 30)
    for i, j in enumerate(jump):
        if nj[i] > 0:
            insert = [[hyp[j, 0], hyp[j, 1] + 30 * n] for n in range(1, nj[i])]
            hyp = np.insert(hyp, j + 1, insert, axis=0)
            jump[i:] = [index + nj[i] - 1 for index in jump[i:]]

    # Verify the intervals
    di = np.diff(hyp[:, 1])
    if np.all(di == 30):
        print("Check completed")
    else:
        print("Error")

    # Adjust time vectors
    start_time = {'h': h[0], 'm': m[0], 's': s[0]}
    anno_start_time = hyp[0, 1]
    anno_end_time = hyp[-1, 1]
    print(start_time)
    timestart = hyp[0, 1]
    time_tot = np.array(timevector) - timestart
    hyp[:, 1] -= hyp[0, 1]

    # Save results
    output_prefix = file_path.split("/")[-1].split(".")[0]
    np.save(os.path.join(store_path, f"micro_str_{output_prefix}.npy"),
            {'time_tot': time_tot, 'duration': duration, 'type_ar': type_ar, 'start_time': start_time})

    np.save(os.path.join(store_path, f"hyp_{output_prefix}.npy"), hyp)
    print("Saving complete")

    return hyp, anno_start_time, anno_end_time


def normalize_channel_name(name):
    """
    Normalize EEG/EMG/EOG channel names to a consistent standard.

    Parameters:
    - name (str): The original channel name from the data.

    Returns:
    - str: The normalized channel name.
    """
    name = name.lower()  # Convert to lowercase for uniformity

    # Dictionary to map categories to normalized names
    normalization_map = {
        'loc': ['loc', 'eog-l', 'loc-a1', 'eog-loc', 'loc / a2'],
        'roc': ['roc', 'eog-r', 'roc-a2', 'roc / a1'],
        'eog': ['roc-loc', 'eog dx'],
        'chin1': ['chin1', 'emg1'],
        'chin2': ['chin2', 'emg2'],
        'emg': ['emg1-emg2', 'emg', 'emg-emg'],
        'f3': ['f3', 'f3a2', 'f3-c3'],
        'c3': ['c3', 'c3a2', 'c3-a2', 'c3-p3'],
        'c4': ['c4', 'c4a1', 'c4-a1', 'c4-p4'],
        'o1': ['o1', 'o1a2', 'o1-a2'],
    }

    # Iterate through the map and normalize
    for normalized_name, possible_names in normalization_map.items():
        if name in possible_names:
            return normalized_name  # Return the normalized name if found

    # If no match, return the original name (or raise an exception if needed)
    return name


def resolve_duplicates(channel_dict):
    resolved_dict = {}
    for raw_name, normalized_name in channel_dict.items():
        if normalized_name in resolved_dict:
            existing_raw_name = resolved_dict[normalized_name]
            if 'a' or 'A' in existing_raw_name:
                continue
            else:
                resolved_dict[normalized_name] = raw_name
        else:
            resolved_dict[normalized_name] = raw_name
    return resolved_dict


def process(pathology=None, local_test=False):
    if local_test is True:
        Root_path = '/Volumes/T7/data/cap-sleep-database-1.0.0'
        log_filename = './cap.log'
        store_path = f'/Volumes/T7/data/cap/process/{pathology}'
        os.makedirs(store_path, exist_ok=True)
    else:
        Root_path = '/data/MGH/'
        os.makedirs('/mnt/LOG_FILE/LOG_BDSP/', exist_ok=True)
        log_filename = f'/mnt/LOG_FILE/LOG_BDSP/{pathology}/process.log'
        store_path = f'/mnt/myvol/data/MGH_NEW/{pathology}'
        os.makedirs(store_path, exist_ok=True)
    pattern_edf = re.compile(rf"{pathology}\d{{1,2}}\.edf$")
    pattern_text = re.compile(rf"{pathology}\d{{1,2}}\.txt$")

    # List all files in the directory
    all_files = os.listdir(Root_path)
    # Filter files for .edf and .text using regex
    path_sig_list = [
        os.path.join(Root_path, file)
        for file in all_files
        if pattern_edf.match(file)
    ]

    path_anno_list = [
        os.path.join(Root_path, file)
        for file in all_files
        if pattern_text.match(file)
    ]
    path_sig_list = sorted(path_sig_list)
    path_anno_list = sorted(path_anno_list)
    print(path_sig_list, path_anno_list)
    logger = setup_logger(log_filename)
    start_time = time.time()
    logger.info(f"Process started with {pathology}: {len(path_sig_list)} files.")
    for i in tqdm(range(1, len(path_sig_list))):
        filename = os.path.join(store_path, 'subject_' + str(i))
        os.makedirs(filename, exist_ok=True)
        logger.info(f'store_path: {filename}')
        signal_path = path_sig_list[i]
        anno_path = path_anno_list[i]
        raw_data = mne.io.read_raw_edf(signal_path, verbose=True, preload=True)
        ch_available = raw_data.ch_names
        channel_names_to_load = ['loc', 'roc', 'eog-l', 'loc-a1', 'loc / a2', 'eog dx', 'roc / a1', 'roc-a2', 'eog-r',
                                 'roc-loc', 'chin1', 'chin2',
                                 'emg1-emg2', 'emg', 'emg-emg', 'emg1', 'emg2', 'f3', 'f3a2', 'f3-c3', 'c3',
                                 'c3a2', 'c3-a2', 'c3-p3', 'c4', 'c4a1', 'c4-a1', 'c4-p4', 'o1', 'o1a2', 'o1-a2']
        channel_names_to_load = [x for x in ch_available if x.lower() in channel_names_to_load]
        raw_data.pick_channels(channel_names_to_load)
        channel_names_to_load = {name: normalize_channel_name(name) for name in channel_names_to_load}
        resolved_dict = resolve_duplicates(channel_names_to_load)
        raw_data.rename_channels({v: k for k, v in resolved_dict.items()})

        def func(array, EOG):
            array -= EOG[0]
            return array

        def func2(array, EMG):
            array -= EMG[0]
            return array

        raw_data.load_data(verbose=False)
        if 'loc' in raw_data.ch_names and 'roc' in raw_data.ch_names:
            data, times = raw_data['loc']
            raw_data.apply_function(func, picks=['roc'], EOG=data)
            raw_data.drop_channels(['loc'])
            raw_data.rename_channels({'roc': 'eog'})

        if 'chin1' in raw_data.ch_names and 'chin2' in raw_data.ch_names:
            data, times = raw_data['chin1']
            raw_data.apply_function(func2, picks=['chin2'], EMG=data)
            raw_data.drop_channels(['chin1'])
            raw_data.rename_channels({'chin2': 'emg'})
        raw_data.resample(100)
        anno, anno_start_time, anno_end_time = parse_remlogic_report(anno_path, store_path=filename)
        meas_date = raw_data.info['meas_date']
        if meas_date is not None:
            if meas_date.hour < 10:
                meas_start_time_seconds = (
                        (meas_date.hour + 24) * 3600 +
                        meas_date.minute * 60 +
                        meas_date.second
                )
            else:
                meas_start_time_seconds = (
                        meas_date.hour * 3600 +
                        meas_date.minute * 60 +
                        meas_date.second
                )
        else:
            meas_start_time_seconds = 0
        sfreq = raw_data.info['sfreq']
        clip_start_idx = (anno_start_time - meas_start_time_seconds)
        clip_end_idx = (anno_end_time - meas_start_time_seconds + 30)
        if clip_start_idx < 0:
            logger.warning(f"Anno start time is earlier than signal start. Adjusting {signal_path}.")
            clip_start_idx = 0
        if clip_end_idx > len(raw_data.times) / sfreq:
            logger.warning(f"Anno end time is later than signal end. Adjusting {signal_path}.")
            clip_end_idx = (len(raw_data.times)/sfreq - clip_start_idx) // 30 * 30
            logger.warning(f'clip_start_idx: {clip_start_idx}, clip_orig_end_idx: {(anno_end_time - meas_start_time_seconds + 30)}, clip_end_idx: {clip_end_idx},'
                           f'len(raw_data.times: {len(raw_data.times)}, epochs: {len(raw_data.times) // (30 * sfreq)}')

        raw_data.crop(tmin=clip_start_idx, tmax=clip_end_idx, include_tmax=False)
        res_channels = ['c3', 'c4', 'emg', 'eog', 'f3', 'fpz', 'o1', 'pz']
        good_channels = np.ones(len(res_channels))
        signal_map = {}
        for c_index, _c in enumerate(res_channels):
            if _c in raw_data.ch_names:  # 如果通道存在
                data = raw_data.get_data(picks=[_c])
                if np.all(data == 0):  # 数据全为零
                    good_channels[c_index] = 0
                    signal_map[_c] = np.zeros(len(raw_data.times))  # 用空信号代替
                else:
                    signal_map[_c] = data.flatten()  # 保存实际信号
            else:
                good_channels[c_index] = 0
                signal_map[_c] = np.zeros(len(raw_data.times))  # 用空信号代替
        signal = np.array([signal_map[ch] for ch in res_channels])
        signal = signal * 1e6
        n_epochs = int(signal.shape[-1] // 3000)
        logger.info(f'orig len: {anno.shape, signal.shape}')
        anno = anno[:n_epochs]
        signal = signal[:, :(n_epochs*3000)]
        logger.info(f'pro len: {anno.shape, signal.shape}')
        logger.info(f'good_channels: {good_channels}, {raw_data.ch_names}  ****  nepochs= {n_epochs}')
        logger.info(f'meas_date: {meas_date}, meas_start_time_seconds: {meas_start_time_seconds} *** '
                    f'anno_start_time: {anno_start_time}, anno_end_time: {anno_end_time}, raw_data.times: {raw_data.times}')

        assert anno[-1, 1] * 100 + 3000 == signal.shape[-1], f'{anno[-1, 1] * 100}, {signal.shape[-1]}'
        signal = np.split(signal, n_epochs, axis=1)
        stage = anno[:, 0]
        pathology = pathology

        h5_filename = f"{filename}/data.h5"
        data_dict = {
            'signal': signal,
            'stage': stage,
            'good_channels': good_channels,
            'pathology': [pathology]
        }
        write_data(h5_filename, data_dict, 2)
        del signal
        gc.collect()

if __name__ == '__main__':
    for pathology in ['plm']:
        process(pathology=pathology, local_test=True)
