import glob
import numpy as np
import os

path = '/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_aug_new_1/SS2/all_split_E2_new_5.npy'
items = np.load(path, allow_pickle=True).item()

def replace2orig(path):
    path = path.split('/')
    path[6] = 'MASS_aug_new_0'
    return os.path.join('/'.join(path))
save_path = "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_aug_new_1/SS2"
st = 0
ed = 0
for k in range(5):
    res = {}
    st = k*3
    ed = (k+1)*3
    for i in range(5):
        names_train = items[f'train_{i}']['names']
        idx = np.arange(len(names_train))
        res[f'train_{i}'] = {}
        onums = []
        on = []
        for idx, tn in enumerate(names_train):
            if st<=idx<ed:
                orig_names = replace2orig(tn)
                on.append(replace2orig(tn))
                search_path = orig_names.split('/')
                search_path[4] = 'DATA'
                search_path = '/'.join(search_path)
                orig_nums = len(glob.glob(os.path.join(search_path, '*')))
                onums.append(orig_nums)
            else:
                on.append(tn)
                onums.append(items[f'train_{i}']['nums'][idx])
        res[f'train_{i}']['names'] = np.array(on)
        res[f'train_{i}']['nums'] = np.array(onums)
        res[f'val_{i}'] = {}
        res[f'val_{i}']['names'] =  items[f'val_{i}']['names']
        res[f'val_{i}']['nums'] = items[f'val_{i}']['nums']
        res[f'test_{i}'] = {}
        res[f'test_{i}']['names'] = items[f'test_{i}']['names']
        res[f'test_{i}']['nums'] = items[f'test_{i}']['nums']

    np.save(os.path.join(save_path, f'all_split_E2_new_5_{k}'), arr=res, allow_pickle=True)
