import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
import os
import gc
import pandas as pd
import pyarrow as pa
path_Extracted = "/Volumes/T7/data/ISRUC_s3/ExtractedChannels/"
path_RawData = "/Volumes/T7/data/ISRUC_s3/RawData/"
path_output = "/Volumes/T7/data/ISRUC_s3/processed"
channels = ['C3_A2', 'C4_A1', 'X1', 'LOC_A2', 'F3_A2', 'O1_A2']
import torch


def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    # mu = np.array([-6.8460e-02, 1.9104e-01, 3.8937e-01, -2.0938e+00,
    #                         1.6496e-03, -4.8439e-05, 8.1125e-04,
    #                         7.1748e-05])
    # std = np.array([34.6887, 34.9556, 23.2826, 35.4035, 26.8738,
    #                          4.9272, 25.1366, 3.6142])
    # using = np.array([0, 1, 2, 3, 4, 6])
    for idx, c in enumerate(channels):
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
        # result_dis = (psg[c]-mu[using[idx]])/std[using[idx]]
        # print(f'sub_id: {sub_id}, {result_dis.mean(), result_dis.std()}')
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C]
        Fold_len:   [k-fold] list
'''

for sub in range(1, 11):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label == 5] = 4  # make 4 correspond to REM
    # Save
    cnt = 0
    name = str(sub)
    for _x, _y in zip(psg, label):
        dataframe = pd.DataFrame(
            {'x': [_x.tolist()], 'stage': _y, }
        )
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(f"{path_output}/{name}", exist_ok=True)
        with pa.OSFile(
                f"{path_output}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        cnt += 1
        del dataframe
        del table
        gc.collect()


print('Preprocess over.')

