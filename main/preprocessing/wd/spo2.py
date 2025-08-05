import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os

def detect_desaturation_events(signal, drop_threshold=3, min_duration_sec=10, fs=1):
    signal = np.array(signal)
    baseline = np.maximum.accumulate(signal)
    drop = baseline - signal
    below_thresh = drop >= drop_threshold

    events = []
    start = None
    for i, val in enumerate(below_thresh):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration_sec * fs:
                events.append((start, i))
            start = None
    if start is not None and len(signal) - start >= min_duration_sec * fs:
        events.append((start, len(signal)))
    return events


def select_most_eventful_segment(signal, fs, segment_length_sec):
    segment_len = segment_length_sec * fs
    max_count = -1
    best_start = 0
    for start in range(0, len(signal) - segment_len, segment_len // 10):
        segment = signal[start:start + segment_len]
        events = detect_desaturation_events(segment, fs=fs)
        if len(events) > max_count:
            max_count = len(events)
            best_start = start
    return best_start, best_start + segment_len


def align_ums_to_psg_by_spo2(ums_spo2, psg_spo2, fs=1, segment_length_sec=180, max_delay_sec=3600, plot=False):
    # 1. 选择PSG中事件最密集片段
    psg_start, psg_end = select_most_eventful_segment(psg_spo2, fs, segment_length_sec)
    psg_template = psg_spo2[psg_start:psg_end]

    # 2. 标准化
    psg_template = (psg_template - np.mean(psg_template)) / (np.std(psg_template) + 1e-6)
    ums_spo2_norm = (ums_spo2 - np.mean(ums_spo2)) / (np.std(ums_spo2) + 1e-6)


    # 3. 滑动相关
    corr = correlate(ums_spo2_norm, psg_template, mode="valid")

    # 限制在最大时延范围内搜索
    max_offset = int(max_delay_sec * fs)
    valid_range_start = max(psg_start - max_offset, 0)
    valid_range_end = min(psg_start + max_offset, len(corr))
    
    search_corr = corr[valid_range_start:valid_range_end]
    relative_best_idx = np.argmax(search_corr)
    best_idx = valid_range_start + relative_best_idx
    if plot is True:
        # 可视化
        plt.plot(corr)
        plt.axvline(best_idx, color='r', linestyle='--', label='Best Match')
        plt.title("Cross-correlation with PSG SpO2")
        plt.grid(True)
        plt.legend()
        plt.show()

    # 4. 输出对齐段
    ums_aligned = ums_spo2[best_idx:best_idx + len(psg_template)]
    psg_aligned = psg_spo2[psg_start:psg_end]
    template_events = detect_desaturation_events(psg_aligned, fs=fs)
    return ums_aligned, psg_aligned, best_idx, psg_start, template_events

def save_figure(filename, save_root, subfolder='figures'):
    """
    保存当前 Matplotlib 图像并关闭
    """
    fig_dir = os.path.join(save_root, subfolder)
    os.makedirs(fig_dir, exist_ok=True)
    full_path = os.path.join(fig_dir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()
    
def plot_alignment_with_events(ums_aligned, psg_aligned, template_events, sub_folder, save_root, filename, save=False, fs=1):
    time = np.arange(len(ums_aligned)) / fs
    plt.figure(figsize=(12, 5))
    plt.plot(time, ums_aligned, label='UMS SpO2', alpha=0.7)
    plt.plot(time, psg_aligned, label='PSG SpO2 Template', alpha=0.7)
    for s, e in template_events:
        plt.axvspan(s / fs, e / fs, color='red', alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("SpO₂ (%)")
    plt.title("SpO₂ Alignment and Desaturation Events")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save is True:
        save_figure(filename=filename, save_root=save_root, subfolder=sub_folder)
    else:
        plt.show()

