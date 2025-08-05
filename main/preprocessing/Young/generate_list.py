import glob
import os.path

import numpy as np

root_path = '/mnt/myvol/data/Young'

res = {}
names = glob.glob(os.path.join(root_path, 'sub*'))
nums = []
for n in names[10:]:
    nums.append(len(glob.glob(os.path.join(n, '*'))))
print(len(names), len(nums))
res['names'] = names[10:]
res['nums'] = nums
print(nums)
np.save(os.path.join(root_path, 'new_test'), arr=res, allow_pickle=True)

res = {}
nums = []
for n in names[:10]:
    nums.append(len(glob.glob(os.path.join(n, '*'))))
print(len(names), len(nums))
res['names'] = names[:10]
res['nums'] = nums
print(nums)
np.save(os.path.join(root_path, 'new_val'), arr=res, allow_pickle=True)
