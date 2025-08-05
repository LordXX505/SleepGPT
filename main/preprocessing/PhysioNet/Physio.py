import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import wfdb
from tqdm import tqdm
import os, glob, functools
from logger import logger as logger
import gc
import sys
from scipy.signal import butter, filtfilt
from multiprocessing import Pool
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

"""
Change it according to your store path.
Check RECORDS file in the path.
"""
Root_path = "/data/Physio"
#/tr03-0005.mat
all_channel = ['ABD',  'AF3', 'AF4', 'AIRFLOW', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'Cz',
               'ECG', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
               'FC5', 'FC6', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3',
               'PO4', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8'] # same with others.
Sleep_stage = ['W', 'N1', 'N2', 'N3', 'R']  # AASM
Stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}

fs = 100.0

l_freq = 0.3
h_freq = 35.0
import h5py
def butter_bandpass(lowcut, highcut, fs, order=5):

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def write_data(h5file, data_dict, compression_opts=2):
    with h5py.File(h5file, 'w') as f:
        for name, data in data_dict.items():
            f.create_dataset(name, data=data, compression="gzip", compression_opts=compression_opts)

def main(which, std, ed, process_idx, paths):
    """
    File:Root_Path + trainning/test + {FileName}
    All file names are in RECORDS.This will download with the Physio data.
    :return: None
    """
    log_filename = f'./log/process_{process_idx}.log'
    logger = setup_logger(log_filename)
    logger.info(f'std: {std}, ed: {ed}')
    for i in tqdm(paths.index[std:ed]):
        subject_path = paths.iloc[i]['subname']
        logger.info(os.path.join(root_path, subject_path, subject_path[:-1]))
        if which == 'train':
            anno = wfdb.rdann(os.path.join(root_path, subject_path, subject_path[:-1]), extension='arousal')
        record = wfdb.rdrecord(os.path.join(root_path, subject_path, subject_path[:-1]))
        # record = wfdb.rdrecord(os.path.join(Root_path, 'tr03-0005'))
        assert record.fs == 200
        df = record.to_dataframe().rename(
            columns={'F3-M2': "F3", 'F4-M1': 'F4', 'C3-M2': 'C3', 'C4-M1': 'C4', 'O1-M2': 'O1', 'O2-M1': 'O2',
                     'E1-M2': 'EOG1',
                     'Chin1-Chin2': 'EMG1', 'ABD': 'ABD', 'CHEST': 'CHEST', 'AIRFLOW': 'AIRFLOW', 'SaO2': 'SaO2',
                     'ECG': 'ECG'})  # Resample to 100hz using mean.
        print(len(df.index))
        # ['ABD', 'AIRFLOW', 'C3' ,'C4', 'ECG', 'EMG1', 'EOG1', 'F3', 'F4', 'O1', 'O2']
        df = df.resample('0.01S').mean()
        # Get the epochs and channels(ascending order)
        epochs = []
        channels = []
        for col in df.columns:
            if col in all_channel:
                tmp = []
                val = df[col].values
                for i in np.arange(0, len(df.index) - 3000, 3000):
                    tmp.append(val[i:i + 3000])
                channels.append(col)
                epochs.append(tmp)
        assert len(channels) == 11, f'{len(channels)}, {subject_path}'
        idx = np.argsort(channels)  # sort channels
        epochs = np.stack(epochs, axis=0)
        print(epochs.shape)
        # b, a = butter_bandpass(l_freq, h_freq, fs, order=5)
        epochs = epochs[idx]
        # filtered_epoch = np.zeros_like(epochs)
        # for i in range(epochs.shape[0]):
        #     filtered_epoch[i, :] = filtfilt(b, a, epochs[i, :])
        # epochs = filtered_epoch
        epochs = epochs.transpose((1, 0, 2))
        # Get the sleep stage.
        # Anno: Onset, Label. ///// Duration = Next record onset - onset
        stages = []
        label = 'UNKnow'
        cnt = 0
        onsets = []
        labels = []
        if which == 'train':
            for i in zip(anno.sample, anno.aux_note):
                if i[1] in Sleep_stage:
                    onsets.append(i[0] // 2)
                    labels.append(i[1])
                    assert (i[0] // 2) % 3000 == 0
        # 0 is Wake.
        # 1 is stage
        # N1.
        # 2 is stage
        # N2.
        # 3 is stage
        # N3.
        # 4 is Rem.
            idx = -1
            for i in np.arange(0, len(df.index) - 3000, 3000):
                if cnt < len(onsets) and i == onsets[cnt]:
                    label = labels[cnt]
                    cnt += 1
                if label != 'UNKnow':
                    if idx == -1:
                        idx = i // 3000
                    stages.append(Stage_map[label])
            epochs = epochs[idx:]
            assert len(epochs) == len(stages), f'len is different {len(epochs)}, {len(stages)}'
            data_dict = {
                'signal': epochs,
                'stage': stages,
            }
        else:
            idx = 0
            stages = None
            epochs = epochs[idx:]
            data_dict = {
                'signal': epochs,
            }

        save_path = f'/mnt/myvol/data/physio/{which}/'
        os.makedirs(os.path.join(save_path, subject_path[:-1]), exist_ok=True)
        try:
            write_data(os.path.join(save_path, subject_path[:-1], 'data.h5'), data_dict)
        except Exception as e:
            print(e)
        logger.info(f"{subject_path[:-1]} success")
        with open(f"{save_path}/{subject_path[:-1]}/success", 'w') as f:
            f.write('success')



if __name__ =='__main__':
    procs = []
    procs_number = 64
    sets = ['test']
    for which in sets:
        # sets = ['test']
        print('**********Doing in {} data***********'.format(which))
        root_path = os.path.join(Root_path, which)
        paths = pd.read_table(os.path.join(root_path, "RECORDS"), names=['subname'])
        # paths = pd.read_table(os.path.join(Root_path, "RECORDS"), names=['subname'])
        print(paths.index)
        piece = len(paths.index)//procs_number
        print(piece)
        result = []
        p = Pool(procs_number)
        start= 0
        end = 1
        # main(which, start, end, f'{which}_{0}', paths)
        for i in range(procs_number):
            start = i * piece
            if i == procs_number - 1:
                end = len(paths.index)
            else:
                end = min((i + 1) * piece, len(paths.index))
            print(f'start {start}, end {end}')
            result.append(p.apply_async(main, args=(which, start, end, f'{which}_{i}', paths)))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
