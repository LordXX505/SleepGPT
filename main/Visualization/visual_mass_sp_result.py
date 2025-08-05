import glob

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ipywidgets import interact, IntSlider
import ipywidgets as widgets
import numpy as np
from matplotlib.widgets import Slider
root_dir = '../../data/ver_log/test/2'
import mne

ckpt_path = glob.glob(f'{root_dir}/*')
time_len = 1000

for ckpt in ckpt_path:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
    plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.5)
    start_index = 0
    end_index = time_len
    result = torch.load(ckpt, map_location=torch.device('cpu'))
    eeg_data = result['real'].numpy()[0]
    eeg_predictg = result['epoch'].numpy()
    eeg_label = result['label'].numpy()
    line1, = ax1.plot(eeg_data[start_index:end_index])
    line2, = ax2.plot(eeg_predictg[start_index:end_index])
    line3, = ax3.plot(eeg_label[start_index:end_index])
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(np.min(eeg_data), np.max(eeg_data))
        # 设置初始x轴范围
        ax.set_xlim(start_index, end_index)
        ax.set_xlabel('Samples')
        ax.set_ylabel('EEG Signal')
    # 创建滑动条
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Start', 0, len(eeg_data) - time_len, valinit=0, valstep=0.0001)
    # 将滑动条和更新函数绑定
    def update(val):
        start = int(val // 100) * 100
        end = start + time_len
        x_values = np.arange(start, end)
        line1.set_xdata(x_values)  # 更新x数据
        line1.set_ydata(eeg_data[start:end])

        line2.set_xdata(x_values)  # 更新x数据
        line2.set_ydata(eeg_predictg[start:end])

        line3.set_xdata(x_values)  # 更新x数据
        line3.set_ydata(eeg_label[start:end])
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(start, end)
        fig.canvas.draw_idle()
    slider.on_changed(update)
    # 显示图表
    plt.show()


