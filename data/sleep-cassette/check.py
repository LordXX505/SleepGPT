import numpy as np


path_pro = './processed/m_new_split_k_10.npy'
path_con = './Aug_consecutive/m_new_split_Aug_consecutive_k_10.npy'


pro_data = np.load(path_pro, allow_pickle=True).item()
con_data = np.load(path_con, allow_pickle=True).item()
for i in range(10):
    pro_names = pro_data[f'train_{i}']['names']
    pro_nums = pro_data[f'train_{i}']['nums']
    con_names = con_data[f'train_{i}']['names']
    con_nums = con_data[f'train_{i}']['nums']
    print(len(pro_names), len(con_names))
    print(sorted(pro_names))
    print(f'{np.sum(pro_nums)}, {np.sum(con_nums)}')