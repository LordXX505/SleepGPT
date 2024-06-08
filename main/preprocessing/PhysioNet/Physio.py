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
"""
Change it according to your store path.
Check RECORDS file in the path.
"""
Root_path = "/nd_disk1/weixuan/PhysioNet"

all_channel = ['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5'
    , 'CP6', 'Cz', 'ECG', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'F1', 'F2', 'F3', 'F4', 'F5'
    , 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fpz', 'Fz'
    , 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'POz'
    , 'Pz', 'T7', 'T8', 'TP7', 'TP8']  # same with others.
Sleep_stage = ['W', 'N1', 'N2', 'N3', 'R']  # AASM
Stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}

def main():
    """
    File:Root_Path + trainning/test + {FileName}
    All file names are in RECORDS.This will download with the Physio data.
    :return: None
    """
    # sets = ['training', 'test']
    sets = ['test']
    for which in sets:
        logger.info('**********Doing in {} data***********'.format(which))
        root_path = os.path.join(Root_path, which)
        paths = pd.read_table(os.path.join(root_path, "RECORDS"), names=['subname'])
        print(paths.index)
        for i in tqdm(paths.index):
            subject_path = paths.iloc[i]['subname']
            dataset_root = os.path.join('/nd_disk1/weixuan',
                                        'Physio_process', which)  # make dataset root path
            logger.info(os.path.join(root_path, subject_path, subject_path[:-1]))
            if which == 'training':
                anno = wfdb.rdann(os.path.join(root_path, subject_path, subject_path[:-1]), extension='arousal')
            record = wfdb.rdrecord(os.path.join(root_path, subject_path, subject_path[:-1]))
            assert record.fs == 200
            df = record.to_dataframe().rename(
                columns={'F3-M2': "F3", 'F4-M1': 'F4', 'C3-M2': 'C3', 'C4-M1': 'C4', 'O1-M2': 'O1', 'O2-M1': 'O2',
                         'E1-M2': 'EOG1',
                         'Chin1-Chin2': 'EMG1', 'ABD': 'ABD', 'CHEST': 'CHEST', 'AIRFLOW': 'AIRFLOW', 'SaO2': 'SaO2',
                         'ECG': 'ECG'})  # Resample to 100hz using mean.
            print(len(df.index))

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
            idx = np.argsort(channels)  # sort channels
            epochs = np.stack(epochs, axis=0)
            print(epochs.shape)
            epochs = epochs[idx]
            channels = [np.sort(channels)]
            epochs = epochs.transpose((1, 0, 2))
            # Get the sleep stage.
            # Anno: Onset, Label. ///// Duration = Next record onset - onset
            stages = []
            label = 'UNKnow'
            cnt = 0
            onsets = []
            labels = []
            if which == 'training':
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
            else:
                idx = 0
                stages = None
            epochs = epochs[idx:].tolist()
            save_dict = {
                "x": epochs,  # epochs * [channels * samples]
                "Stage_label": stages,  # Stage_label [epochs]
                "fs": 100,  # 100
                "start_datetime": 0,  # datetime.datetime(1989, 4, 24, 16, 13)
                "file_duration": len(epochs) * 30,  # max time
                "epoch_duration": 30,  # 30.0
                "n_epochs": len(epochs),  # epochs
            }
            dataframe = pd.DataFrame(save_dict)
            table = pa.Table.from_pandas(dataframe)
            os.makedirs(dataset_root, exist_ok=True)
            print(dataset_root)
            with pa.OSFile(
                    f"{dataset_root}/{subject_path[:-1]}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del save_dict
            gc.collect()



if __name__ =='__main__':
    main()
