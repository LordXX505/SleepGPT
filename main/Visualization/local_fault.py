import os
import mne
import matplotlib.pyplot as plt
import torch
import numpy as np
val_ = np.load('/data/data/shhs_new/val111.npz')
for name in val_['names']:
    print(name)
for num in val_['nums']:
    print(num)
print(len(val_['names']))
print(len(val_['nums']))
# mne_path = "../../data/shhs1-202343.edf"
# local_path = mne_path
# raw_data = mne.io.read_raw_edf(local_path).load_data()
# raw_data = raw_data.copy().filter(l_freq=0.3, h_freq=35)
# # raw_data.plot()
# data, time = raw_data[2]
# data = torch.from_numpy(data)
# data = data[0][:2048]
# print(data.shape)
# print('fft begin')
# fft_plot = torch.fft.fft(data)
# x_axis = len(data)//2*0.35
# plt.plot(torch.abs(fft_plot))
# plt.xlim((0, x_axis))
# plt.show()