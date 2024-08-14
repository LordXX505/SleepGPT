import gc

import mne
import os
import re
import h5py
import argparse
import logging
import numpy as np
import glob
import copy
picks = ['EEG C3-CLE', 'EEG C4-CLE', 'EMG Chin1', 'EMG Chin3', 'EOG Left Horiz', 'EEG F3-CLE', 'EEG O1-CLE', 'EEG Pz-CLE'],
def write_data(h5file, data_dict, compression_opts=5):
    with h5py.File(h5file, 'w') as f:
        for name, data in data_dict.items():
            f.create_dataset(name, data=data, compression="gzip", compression_opts=compression_opts)
def check_chanels(channels, k):
    res = []
    temp_pick = copy.deepcopy(picks[k-1])
    if k == 1:
        anchor = channels[0].split()[-1].split('-')[-1]
        if anchor == 'LER':
            for i in range(len(temp_pick)):
                if 'CLE' in temp_pick[i]:
                    temp_pick[i] = temp_pick[i].replace('CLE', 'LER')
    return temp_pick

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7/",
                        # default="/home/cuizaixu_lab/huangweixuan/DATA_C/data/data",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str,
                        default="/Volumes/T7/",
                        # default="/home/cuizaixu_lab/huangweixuan/DATA_C/data/data",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    cnt = 0
    for k in range(1, 2):
        logger.info(f'----------Using Mass SS{k} dataset------------')
        base_data_dir = os.path.join(args.data_dir, f'SS{k}', 'bio')
        anno_base_data_dir = os.path.join(args.data_dir, f'SS{k}', 'ana')
        # Read raw and annotation from EDF files
        psg_fnames = glob.glob(os.path.join(base_data_dir, "*PSG.edf"))
        stage_fnames = glob.glob(os.path.join(base_data_dir, "*Base.edf"))
        anno_fnames = glob.glob(os.path.join(anno_base_data_dir, "*Annotations.edf"))
        psg_fnames.sort()
        stage_fnames.sort()
        anno_fnames.sort()
        psg_fnames = np.asarray(psg_fnames)
        stage_fnames = np.asarray(stage_fnames)
        anno_fnames = np.asarray(anno_fnames)
        resume_idx = 9
        # for i in range(resume_idx, len(psg_fnames)):
        for i in range(resume_idx, 10):
            logger.info("Loading ...")
            logger.info("Signal file: {}".format(psg_fnames[i]))
            logger.info("Annotation file: {}".format(anno_fnames[i]))
            name = os.path.split(psg_fnames[i])[1].split(' ')[0]
            psg_f = mne.io.read_raw_edf(psg_fnames[i], preload=True)
            anno_f = mne.read_annotations(anno_fnames[i])
            stage_f = mne.read_annotations(stage_fnames[i])
            pick_channels = check_chanels(psg_f.ch_names, k)
            psg_f.pick(pick_channels)
            psg_f.set_annotations(stage_f)
            psg_f.resample(100)
            if 'EMG Chin3' in pick_channels:
                def func(array, EMG3):
                    array -= EMG3[0]
                    return array

                data, times = psg_f['EMG Chin3']
                psg_f.apply_function(func, picks=['EMG Chin1'], EMG3=data)
                psg_f.drop_channels(['EMG Chin3'])  # drop channel EMG3
            labels = np.zeros(len(psg_f))
            n_epochs = len(psg_f)//3000
            print(f'n_epochs: {n_epochs}')
            total_time = 0
            for onset, duration, item in zip(anno_f.onset, anno_f.duration, anno_f.description):
                if item.startswith('<Event'):
                    match = re.search(r'groupName="([^"]+)"', item)
                    if match:
                        group_name = match.group(1)
                        if group_name == 'ObstructiveApnea':
                            print(group_name, onset, duration)
                            cnt += 1
                            begin_ind = psg_f.time_as_index(times=onset)[0]
                            # print(f'times after: {begin_ind}')
                            # print(f'end: {times["onset"] + times["duration"]}')
                            end_ind = psg_f.time_as_index(times=onset + duration)[0]
                            labels[begin_ind:end_ind] = 1
                            total_time += duration
                    else:
                        print("groupName not found")
            print(f'name: {psg_fnames[i]} apnea duration: {total_time}, accounts for {total_time/(n_epochs*30)*100}%')
            epochs = psg_f[:, :n_epochs * 3000][0]
            labels = labels[:n_epochs * 3000]
            epochs = epochs * 1e6
            print(f'max: {np.max(epochs)}, min:{np.min(epochs)}')
            labels = np.array(np.split(labels, n_epochs, axis=0))
            select_idx = []
            for _, item in enumerate(labels):
                if np.sum(item) != 0:
                    select_idx.append(_)
            print(f"epochs.shape: {epochs.shape}")
            print(f'labels: {len(labels)}')
            print(f'select_idx: {select_idx}, len: {len(select_idx)}')
            filename = os.path.join(args.output_dir, f'MASS_{k}_Apnea', name)
            os.makedirs(filename, exist_ok=True)
            h5_filename = f"{filename}/data.h5"
            data_dict = {
                'signal': epochs,
                'apnea': labels
            }
            write_data(h5_filename, data_dict, 2)
            del epochs, labels
            gc.collect()


if __name__ == '__main__':
    main()
