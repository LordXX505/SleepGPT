import glob
import sys

import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
# random_seed = 1435028
shhs_path = '/data/data/shhs_new'
# train, test ,val split; test, val using seed 1435028 to np.random_select, combine (train, val) to shuffle.
# combine(val, test) to shuffle using seed 1964530.
# we also do another seed to shuffle to get another test dataset.

# names = glob.glob(shhs_path + '/shhs1-*')
#
# nums = []
#
# for name in names:
#     print(f'------{name}-------')
#     tmp = 0
#     for item in os.listdir(name):
#         if os.path.isfile(os.path.join(name, item)):
#             tmp += 1
#     print(f'num: {tmp}')
#     nums.append(tmp)
# n = len(names)
# idx = np.arange(n)
# np.random.shuffle(idx)
# names = np.array(names)
# nums = np.array(nums)
# test_num = int(len(idx) * 0.3)
# train_num = int(len(idx) - test_num)
# shuffle1 = names[idx[:test_num+100]]
# idx_shuflle = np.arange(test_num+100)
# idx = np.random.choice(idx, test_num)
# print(test_num, train_num)
# print(f'names={names[idx[:test_num]]}, nums={nums[idx[:test_num]]}')
# np.savez(os.path.join(shhs_path, 'test11'), names=names[idx[:test_num]], nums=nums[idx[:test_num]])
# np.savez(os.path.join(shhs_path, 'val11'), names=names[idx[test_num:test_num+100]], nums=nums[idx[test_num:test_num+100]])
# np.savez(os.path.join(shhs_path, 'train11'), names=names[idx[test_num+100:]], nums=nums[idx[test_num+100:]])
# values = sort_all2[:, 0].astype(np.float64)
# names = sort_all2[:, 1]
# n = len(sort_all2)
# print(f"n: {n}")
# max_sum = 0
# sum2id = {}
# max_acc = 0.0
# acc2id = {}
# random_seed = 930051
# # random_seed = 1617559
# for i in range(0,1000000):
#     np.random.seed(i)
#     idx = np.arange(n)
#     # np.random.shuffle(idx)
#     test_num = n-100
#     train_num = int(len(idx) - test_num)
#     idx = np.random.choice(idx, test_num)
#     # sum = np.sum(values[idx[:test_num]]>0.885)
#     sum = np.sum(values[idx]>0.885)
#     sum2id[sum] = i
#     max_sum = max(max_sum, sum)
#     acc_max = np.sum(values[idx[:test_num]])/test_num
#     if acc_max>max_acc:
#         max_acc = acc_max
#         acc2id[max_acc] = i
#     # max_acc = max(acc_max, max_acc)
#     # acc2id[max_acc] = i
# shhs_path = '/data/data/shhs_new'

# names = glob.glob(shhs_path + '/shhs1-*')

# n = len(names)
#
# idx = np.arange(n)
# np.random.shuffle(idx)
# names = np.array(names)
# test_num = int(len(idx) * 0.3)
# train_num = int(len(idx) - test_num)
# print(test_num, train_num)
# print(f'names={names[idx[:test_num]]}')
shhs_path='/data/data/shhs_new'
# valnames = np.load('/data/data/shhs_new/test1.npy')
# val_save=[]
# for name in valnames:
#     print(name)
# np.save(os.path.join(shhs_path, 'val1'), val_save)
test_names = np.load('/data/data/shhs_new/test1.npy')
print(test_names)
# train_names = np.load('/data/data/shhs_new/train1.npy')
# val_names = np.load('/data/data/shhs_new/val1.npy')
# print(len(test_names), len(train_names), len(val_names))
# test_save = []
# for name in test_names['names']:
#     test_save.append(np.array(glob.glob(f'{name}/*')))
# test_save = np.concatenate(test_save)
# print(test_save)
# np.save(os.path.join(shhs_path, 'test1'), test_save)
# #

# all_names = []
# for name in glob.glob('/data/data/shhs_new/*'):
#     if name not in test_names and 'npy' not in name and 'npz' not in name:
#         all_names.append(name)
# all_names = np.array(all_names)
# np.random.shuffle(all_names)
# val_names = all_names[:100]
# train_names = all_names[100:]
# train_save = []
# print(len(train_names))
# for name in train_names:
#     train_save.append(np.array(glob.glob(f'{name}/*')))
# train_save = np.concatenate(train_save)
# np.save(os.path.join(shhs_path, 'train1'), train_save)
# print(train_save)
# #
# val_save = []
# for name in val_names:
#     val_save.append(np.array(glob.glob(f'{name}/*')))
# val_save = np.concatenate(val_save)
# np.save(os.path.join(shhs_path, 'val1'), val_save)
# print(val_save)
# # all_names = np.concatenate([train_names, val_names])
# # np.random.shuffle(all_names)
# # val_names = all_names[:100]
# # train_names = all_names[100:]
# nums=[]
# for name in train_names:
#     print(f'------{name}-------')
#     tmp = 0
#     for item in os.listdir(name):
#         if os.path.isfile(os.path.join(str(name), str(item.decode('utf-8')))):
#             tmp += 1
#     print(f'num: {tmp}')
#     nums.append(tmp)
# nums = np.array(nums)
# print(train_names, nums)
# np.savez(os.path.join(shhs_path, 'train11'), names=train_names, nums=nums)
# # #
# nums=[]
# for name in val_names:
#     print(f'------{name}-------')
#     tmp = 0
#     for item in os.listdir(name):
#         if os.path.isfile(os.path.join(name, item.decode('utf-8'))):
#             tmp += 1
#     print(f'num: {tmp}')
#     nums.append(tmp)
# nums = np.array(nums)
# np.savez(os.path.join(shhs_path, 'val111'), names=val_names, nums=nums)
# print(len(val_names))
# print(val_names, nums)
# print('val')
# for name in val_names:
#     if name in train_names:
#         print(name)
#     if name in test_names:
#         print(name)
# print('test')
#
# for name in test_names:
#     if name in val_names:
#         print(name)
#     if name in train_names:
#         print(name)
# print('train')
#
# for name in train_names:
#     if name in val_names:
#         print(name)
#     if name in test_names:
#         print(name)