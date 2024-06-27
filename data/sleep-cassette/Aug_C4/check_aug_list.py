import numpy as np

path = './m_new_split_Aug_C4_k_10.npy'

aug_list = np.load(path, allow_pickle=True).item()
print(aug_list)