import glob
import pyarrow as pa
import numpy as np
import torch
path = glob.glob('/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_2/SS2/E2/*/train/*')
cnt = 0
cnt_true = 0
for item in path:
    tables = pa.ipc.RecordBatchFileReader(
        pa.memory_map(item, "r")
    ).read_all()
    label = tables['Spindles']
    try:
        x = np.array(label.as_py())
    except:
        x = np.array(label.to_pylist())
    x = torch.from_numpy(x).squeeze().long()
    spindle_e = torch.sum(x) > 25
    cnt_true += int(spindle_e)
    cnt += 1
print(cnt, cnt_true)