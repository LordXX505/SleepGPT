import glob
import os.path

import numpy as np

root_path = '/mnt/myvol/data/SD'

res = {}
names = sorted(glob.glob(os.path.join(root_path, 'sub*')))
nums = []
for n in names:
    nums.append(len(glob.glob(os.path.join(n, '*'))))
print(len(names), len(nums))
res['names'] = names
res['nums'] = nums
print(nums)
np.save(os.path.join(root_path, 'new_train'), arr=res, allow_pickle=True)
