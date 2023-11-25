import mne
import numpy as np
import pyarrow as pa
import os
import glob
import pandas as pd
import gc
path_root ="/home/cuizaixu_lab/huangweixuan/data/SS2"

ana_spindle = os.path.join(path_root, 'SS2_ana')
# print(ana_spindle)
spindle_path_E1 = glob.glob(ana_spindle+'/*Spindles_E1*')
spindle_path_E2 = glob.glob(ana_spindle+'/*Spindles_E2*')
# print(spindle_path_E1)
spindle_path_E1 = sorted(spindle_path_E1)
spindle_path_E2 = sorted(spindle_path_E2)
epoch = os.path.join(path_root, 'SS2_bio')
# print(os.path.split(spindle_path_E1[0])[1])
name = os.path.split(spindle_path_E1[0])[1].split(' ')[0]
mne.io.read_raw_edf(glob.glob(epoch+f"/{name}*PSG*")[0])

E1_anno = []
E2_anno = []
epoch = os.path.join(path_root, 'SS2_bio')
for _, path in enumerate([spindle_path_E1, spindle_path_E2]):
    expert = 'E1'
    if _ == 1:
        expert = 'E2'
    for items in path:

        name = os.path.split(items)[1].split(' ')[0]
        print(f"----------{name}-----------")
        epochs = mne.io.read_raw_edf(glob.glob(epoch+f"/{name}*PSG*")[0])
        anno = mne.read_annotations(items)
        print(f"epochs: {epochs}")
        epochs.load_data()
        epochs.filter(l_freq=0.3, h_freq=35, n_jobs='cuda', method='fir')
        epochs = epochs.resample(100)

        bads = epochs.info['bads']
        badsidx = [epochs[_] for _ in bads]
        badsidx = sorted(badsidx)
        print(f'{epochs.info["bads"]}, idx: {badsidx}')
        epochs.rename_channels({'EEG C3-CLE':'C3', 'EEG C4-CLE':'C4', 'EEG F3-CLE':'F3', 'EEG O1-CLE': 'O1', 'EEG Fpz-CLE': 'Fpz', 'EMG Chin':'EMG', 'EEG Pz-CLE':'Pz','EOG Left Horiz':'EOG'})
        epochs.pick(['C3', 'C4', 'EMG', 'EOG', 'F3', 'Fpz', 'O1', 'Pz'])
        labels = np.zeros(len(epochs))
        choose_idx = {}
        n_epochs = len(epochs)//2000
        for i in range(n_epochs):
            choose_idx[i] = 0
        for times in anno:
            begin_ind = epochs.time_as_index(times=times['onset'])[0]
            end_ind = epochs.time_as_index(times=times['onset']+times['duration'])[0]
            bucket_begin = begin_ind//2000
            butcket_end = end_ind//2000
            print(f"bucket_begin: {bucket_begin}, butcket_end: {butcket_end}")
            if butcket_end==bucket_begin:
                if butcket_end in choose_idx:
                    choose_idx[butcket_end] = 1
                    print(f"saving bucket_begin:{bucket_begin}")
            else:
                end_begin = butcket_end*2000
                if (end_begin-begin_ind) //(butcket_end-bucket_begin) > 0.25:
                    if bucket_begin in choose_idx:
                        choose_idx[bucket_begin] = 1
                        print(f"saving bucket_begin:{bucket_begin}, end_begin: {end_begin}, begin_ind:{begin_ind}, butcket_end:{butcket_end}")
                if (butcket_end-end_begin) //(butcket_end-bucket_begin) > 0.25:
                    if butcket_end in choose_idx:
                        choose_idx[butcket_end] = 1
            labels[begin_ind:end_ind] = 1
        print(type(labels))
        epochs = epochs[:, :n_epochs*2000][0]
        print(epochs.shape)
        labels = labels[:n_epochs*2000]
        epochs = np.split(epochs, n_epochs, axis=1)
        labels = np.split(labels, n_epochs, axis=0)
        print(f"epochs.shape: {len(epochs)}")
        print(f'labels: {len(labels)}')
        cnt=0
        filename='/home/cuizaixu_lab/huangweixuan/data/MASS/SS2' + f"/{expert}"
        for k, v in choose_idx.items():
            if v==1:
                save_epochs = epochs[k]
                save_labels = labels[k]
                dataframe = pd.DataFrame(
                        {'x': [save_epochs.tolist()], 'Spindles': [save_labels.tolist()], 'bads': [bads]}
                    )
                table = pa.Table.from_pandas(dataframe)
                os.makedirs(f"{filename}/{name}", exist_ok=True)
                with pa.OSFile(
                            f"{filename}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
                    ) as sink:
                        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                            writer.write_table(table)
                cnt += 1
                del dataframe
                del table
                gc.collect()

print('End')
E1_sub = '/home/cuizaixu_lab/huangweixuan/data/MASS/SS2/E1/*'
names = []
nums = []
for sub in glob.glob(E1_sub):
    names.append(sub)
for name in names:
    print(f'------{name}-------')
    tmp = 0
    for item in os.listdir(name):
        if os.path.isfile(os.path.join(str(name), str(item))):
            tmp += 1
    print(f'num: {tmp}')
    nums.append(tmp)
nums = np.array(nums)
n = len(names)
idx = np.arange(n)
np.random.shuffle(idx)
names = np.array(names)
nums = np.array(nums)
k_split = n//5
res = {}
path = '/home/cuizaixu_lab/huangweixuan/data/MASS/SS2/'
for i in range(5):
    st = i*k_split
    ed = (i+1)*k_split
    idx_split = idx[st:ed]
    idx_train = np.setdiff1d(idx, idx_split)
    res[f'train_{i}'] = {}
    res[f'train_{i}']['names'] = names[idx_train[1:]]
    res[f'train_{i}']['nums'] = nums[idx_train[1:]]
    res[f'val_{i}'] = {}
    res[f'val_{i}']['names'] = names[idx_train[:1]]
    res[f'val_{i}']['nums'] = nums[idx_train[:1]]
    res[f'test_{i}'] = {}
    res[f'test_{i}']['names'] = names[idx_split]
    res[f'test_{i}']['nums'] = nums[idx_split]
    print(len(res[f'test_{i}']['nums']), len(res[f'test_{i}']['names']), len(res[f'val_{i}']['nums']), len(res[f'train_{i}']['nums']))
np.save(os.path.join(path, f'all_split_E1'), arr= res,allow_pickle=True)


E1_sub = '/home/cuizaixu_lab/huangweixuan/data/MASS/SS2/E2/*'
names = []
nums = []
for sub in glob.glob(E1_sub):
    names.append(sub)
for name in names:
    print(f'------{name}-------')
    tmp = 0
    for item in os.listdir(name):
        if os.path.isfile(os.path.join(str(name), str(item))):
            tmp += 1
    print(f'num: {tmp}')
    nums.append(tmp)
nums = np.array(nums)
n = len(names)
idx = np.arange(n)
np.random.shuffle(idx)
names = np.array(names)
nums = np.array(nums)
k_split = n//5
res = {}
path = '/home/cuizaixu_lab/huangweixuan/data/MASS/SS2'
for i in range(5):
    st = i*k_split
    ed = (i+1)*k_split
    idx_split = idx[st:ed]
    idx_train = np.setdiff1d(idx, idx_split)
    res[f'train_{i}'] = {}
    res[f'train_{i}']['names'] = names[idx_train[1:]]
    res[f'train_{i}']['nums'] = nums[idx_train[1:]]
    res[f'val_{i}'] = {}
    res[f'val_{i}']['names'] = names[idx_train[:1]]
    res[f'val_{i}']['nums'] = nums[idx_train[:1]]
    res[f'test_{i}'] = {}
    res[f'test_{i}']['names'] = names[idx_split]
    res[f'test_{i}']['nums'] = nums[idx_split]
    print(len(res[f'test_{i}']['nums']), len(res[f'test_{i}']['names']), len(res[f'val_{i}']['nums']), len(res[f'train_{i}']['nums']))
np.save(os.path.join(path, f'all_split_E2'), arr= res,allow_pickle=True)