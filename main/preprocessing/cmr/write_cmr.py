import os
import pandas as pd
import numpy as np
import torch
import torchvision
import mne
from logger import logger as logger
from tqdm import tqdm
import os, glob, functools
import pyarrow as pa
import gc


def main(root_path):
    for which in tqdm(['SD']):
        print('***********{}**************'.format(which))
        root = os.path.join(root_path, which)
        # names = np.load(os.path.join(root, 'test.npy'), allow_pickle=True)
        # names2 = np.load(os.path.join(root, 'val.npy'), allow_pickle=True)
        # print(names, names2)
        # names = np.concatenate([names, names2])
        names = np.load(os.path.join(root, 'train.npy'), allow_pickle=True)
        print(names)
        train = []
        dataset_root = '/gpfs/share/home/2201210064/Sleep/main/data/{}'.format(which)

        for name in names:
            if 'sub1002' in name:
                name = name[:-1]
            print('***********{}**************'.format(name))
            tables = [pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{root}/{name}.arrow", "r")
            ).read_all()]
            table = pa.concat_tables(tables, promote=True)
            x = table['x'].to_numpy()
            res_x = []
            for item in x:
                tmp = []
                for chanel in item:
                    tmp.append(chanel)
                res_x.append(tmp)
            res_x = np.array(res_x)
            Stage_label = np.array(table['Stage_label'].to_numpy())
            Spindle_label = []
            for item in np.array(table['Spindle_label'].to_numpy()):
                Spindle_label.append(item)
            Spindle_label = np.array(Spindle_label)
            print(res_x.shape[0])
            cnt = 0
            for save_x, save_spindle, save_stage in zip(res_x, Spindle_label, Stage_label):
                dataframe = pd.DataFrame(
                    {'x': [save_x.tolist()], 'spindle': [save_spindle.tolist()], 'stage': save_stage}
                )
                print(dataframe)
                save_table = pa.Table.from_pandas(dataframe)
                os.makedirs(f"{dataset_root}/{name}", exist_ok=True)
                print(f"{dataset_root}/{name}/{str(cnt).zfill(5)}.arrow")
                with pa.OSFile(
                    f"{dataset_root}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
                ) as sink:
                    with pa.RecordBatchFileWriter(sink, save_table.schema) as writer:
                        writer.write_table(save_table)
                del dataframe
                del save_table
                gc.collect()
                cnt = cnt + 1
            train.append([f"{dataset_root}/{name}/{str(i).zfill(5)}.arrow" for i in range(cnt)])
        np.save(f"{dataset_root}/train.npy", np.array(train))

def mean_std(root_path):
    for which in tqdm(['SD']):
        print('***********{}**************'.format(which))
        root = os.path.join(root_path, which)
        # names = np.load(os.path.join(root, 'test.npy'), allow_pickle=True)
        # names2 = np.load(os.path.join(root, 'val.npy'), allow_pickle=True)
        # print(names, names2)
        # names = np.concatenate([names, names2])
        names = np.load(os.path.join(root, 'train.npy'), allow_pickle=True)
        print(names)
        train = []
        dataset_root = '/gpfs/share/home/2201210064/Sleep/main/data/{}'.format(which)
        train_2 = []
        for name in names:
            if 'sub1002' in name:
                name = name[:-1]
            print('***********{}**************'.format(name))
            tables = [pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{root}/{name}.arrow", "r")
            ).read_all()]
            table = pa.concat_tables(tables, promote=True)
            x = table['x'].to_numpy()
            res_x = []
            for item in x:
                tmp = []
                for chanel in item:
                    tmp.append(chanel)
                res_x.append(tmp)
            res_x = np.array(res_x)
            std = np.mean(res_x)
            train.append(std)
            std_2 = np.mean(np.square(res_x))
            train_2.append(std_2)
        train = np.array(train)
        train_2 = np.array(train_2)
        train = np.mean(train)
        train_2 = np.mean(train_2)
        print(train, train_2)
        # np.save(f"{dataset_root}/train.npy", np.array([train, train_2]))

if __name__ == '__main__':
    # main(root_path='/gpfs/share/home/2201210064/Sleep/data/')
    mean_std(root_path='/gpfs/share/home/2201210064/Sleep/data/')