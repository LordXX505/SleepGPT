import os

import numpy as np


path_pro = './processed/m_new_split_k_20.npy'
path_con = './processed/m_new_split_Aug_Half_Orig_k_20.npy'


pro_data = np.load(path_pro, allow_pickle=True).item()
con_data = np.load(path_con, allow_pickle=True).item()
con_names = []
flag_con = {}
flag_pro = {}
for i in range(0, 20):
    for n in con_data[f'test_{i}']['names']:
        temp = os.path.basename(n)[3:5]
        if temp not in flag_con:
            flag_con[temp] = 1
            con_names.append(os.path.basename(n)[3:5])
con_names = np.array(con_names)
print(con_names)
print(np.argsort(con_names))
