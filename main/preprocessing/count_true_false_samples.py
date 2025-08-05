import glob
import pyarrow as pa
import numpy as np
import torch

import os
import sys
sys.path.append('/home/cuizaixu_lab/huangweixuan/Sleep')
from main.transforms import normalize

path = glob.glob('/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_2/SS2/E2/*/train/*')
cnt = 0
cnt_true = 0
res_name = ['01-02-0006', '01-02-0012']
for item in path:
    item_name = item.split('/')[-3]
    if item_name not in res_name:
        continue
    tables = pa.ipc.RecordBatchFileReader(
        pa.memory_map(item, "r")
    ).read_all()
    # label = tables['Spindles']
    x = tables['x'][0]
    try:
        x = np.array(x.as_py())
    except:
        x = np.array(x.to_pylist())
    x = torch.from_numpy(x).squeeze().float() * 1e6
    norm = normalize()
    print(f'Un x : {x}')
    x = norm(x)
    print(x.shape)
    print(f'normalize {x[0]}')
    print(f'item: {item}, name: {item_name}, min: {x[0].min()}, max: {x[0].max()}')
