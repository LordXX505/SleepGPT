import numpy as np

file = np.load(f'/home/cuizaixu_lab/huangweixuan/data/data/MASS_aug_new_2/SS2/all_split_E1_new_5.npy', allow_pickle=True)
all_item = []
for k in range(5):
    all_item.append(file.item()[f'test_{k}']['names'])
all_item = np.concatenate(all_item)
all_item = np.sort(all_item)

print(all_item)

