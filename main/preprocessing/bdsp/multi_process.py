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
from sklearn.linear_model import LinearRegression
from bdsp_sleep_functions import load_bdsp_signal, annotations_preprocess, vectorize_respiratory_events, vectorize_sleep_stages, vectorize_arousals, vectorize_limb_movements
def process_mgh(idx, path_list, anno_list):

if __name__ == '__main__':
    procs = []
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    p = Pool(96)
    result = []
    all_data_path_list = []
    piece = len(all_data_path_list)//96
    for i in range(96):
        start = i*piece
        if i == 96:
            end = len(all_data_path_list)
        else:
            end = min((i+1)*piece, len(all_data_path_list))
        print(f'start {start}, end {end}')
        # process_shhs(i, all_data_path_list[start:end], all_anno_path_list[start:end])
        result.append(p.apply_async(process_mgh, args=(i%8, all_data_path_list[start:end], all_anno_path_list[start:end])))
        if end == len(all_data_path_list):
            break
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    k = ""
    for res in result:
        suc = res.get()
        # suc=res
        for _ in suc[0]:
            k += _
            k += "\n"
        print(suc[1])
    with open('./shhs_log', 'w') as f:
        f.write(k)
    print('write to shhs_log')