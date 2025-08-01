import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os

def map_auto_labels_to_aasm(auto_labels):
    """
    将自定义自动标签映射为AASM标准标签:
    2 → 3 (N3)
    3 → 1 (N1+N2 → 映射为 N1)
    4 → 4 (REM)
    5 → 0 (Wake)
    其余值 → -1（无效）
    """
    mapped = []
    for label in auto_labels:
        if label == 2:
            mapped.append(3)  # N3
        elif label == 3:
            mapped.append(1)  # N1 (可调整为 N2 如果更合适)
        elif label == 4:
            mapped.append(4)  # REM
        elif label == 5:
            mapped.append(0)  # Wake
        else:
            mapped.append(-1)  # unknown
    return np.array(mapped)

def map_manual_labels_to_auto(manual_labels):
    """
    将PSG的AASM标准标签转换为自动分期的标签体系：
    Wake (0) → 4
    N1/N2 (1/2) → 3
    N3 (3) → 2
    REM (4) → 5
    其他 → -1
    """
    mapped = []
    for label in manual_labels:
        if label == 0:
            mapped.append(4)
        elif label in [1, 2]:
            mapped.append(3)
        elif label == 3:
            mapped.append(2)
        elif label == 4:
            mapped.append(5)
        else:
            mapped.append(-1)  # 无效或未标注
    return np.array(mapped)

def plot_original_hypnogram(org_labels, org_auto_labels, filename, save_root, sub_folder, save=False, epoch_sec=30):
    org_labels = np.asarray(org_labels).squeeze()
    org_auto_labels = np.asarray(org_auto_labels).squeeze()

    time_manual = np.arange(len(org_labels)) * epoch_sec / 60
    time_auto = np.arange(len(org_auto_labels)) * epoch_sec / 60

    manual_mapped = map_manual_labels_to_auto(org_labels)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].step(time_manual, manual_mapped, where='post', label='Original Manual (PSG)')
    axes[0].set_ylabel("Sleep Stage")
    axes[0].set_title("Manual Hypnogram (PSG)")
    axes[0].set_yticks([2, 3, 4, 5])
    axes[0].set_yticklabels(['N3', 'N1/N2', 'REM', 'Wake'])
    axes[0].invert_yaxis()
    axes[0].grid(True)

    axes[1].step(time_auto, org_auto_labels, where='post', linestyle='--', color='orange', label='Original Auto (UMS)')
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Sleep Stage")
    axes[1].set_title("Auto Hypnogram (UMS)")
    axes[1].set_yticks([2, 3, 4, 5])
    axes[1].set_yticklabels(['N3', 'N1/N2', 'REM', 'Wake'])
    axes[1].invert_yaxis()
    axes[1].grid(True)

    plt.tight_layout()
    if save is True:
        save_figure(filename=filename, save_root=save_root, subfolder=sub_folder)
    else:
        plt.show()


def plot_manual_vs_auto_hypnogram(manual_labels, auto_labels,  sub_folder, save_root, filename, save=True, epoch_sec=30):
    """
    可视化手动（PSG）和自动（UMS）睡眠分期对比图。
    manual_labels: PSG人工标注（AASM标签）
    auto_labels: 自动分期结果（2=深睡, 3=浅睡, 4=REM, 5=醒）
    """
    assert len(manual_labels) == len(auto_labels), "标签长度不一致，无法对齐"

    # 映射手动标签到自动标签体系
    manual_mapped = map_manual_labels_to_auto(manual_labels)

    # 去除无效标签
    valid_idx = manual_mapped != -1
    manual_mapped = manual_mapped[valid_idx]
    auto_labels = auto_labels[valid_idx]

    time = np.arange(len(manual_mapped)) * epoch_sec / 60  # 转换为分钟

    plt.figure(figsize=(14, 4))
    plt.step(time, manual_mapped, label='Manual (PSG)', where='post', linewidth=2)
    plt.step(time, auto_labels, label='Auto (UMS)', where='post', linestyle='--', alpha=0.8)

    # 根据映射的自动标签设置 y 轴
    plt.yticks([2, 3, 4, 5], ['N3', 'N1/N2', 'REM', 'Wake'])
    plt.gca().invert_yaxis()  # 上为醒，下为深睡
    plt.xlabel('Time (minutes)')
    plt.ylabel('Sleep Stage')
    plt.title('Sleep Staging: Manual vs Auto (Aligned)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save is True:
        save_figure(filename=filename, save_root=save_root, subfolder=sub_folder)
    else:
        plt.show()

def align_spo2_signals_with_offset(psg_signal, ums_signal, offset_sec, labels, auto_labels, fs=1, epoch_sec=30):
    samples_per_epoch = epoch_sec * fs
    offset_sec = int(offset_sec)

    if offset_sec > 0:
        # UMS 早启动，丢掉前面 offset 秒的 UMS
        ums_end = len(auto_labels) * samples_per_epoch 
        ums_aligned = ums_signal[offset_sec:ums_end]
        ums_len = len(ums_aligned) // samples_per_epoch * samples_per_epoch
        psg_aligned = psg_signal[:ums_len]
        offset_labels = labels[:ums_len // samples_per_epoch]
        offset_auto_labels = auto_labels[offset_sec//samples_per_epoch:]
        ums_start = offset_sec
        offset_sec_adjusted = offset_sec
    else:
        # UMS 晚启动，丢掉前面 offset 秒的 PSG
        offset_sec = abs(offset_sec)
        if offset_sec % samples_per_epoch == 0:
            psg_aligned = psg_signal[offset_sec:]
            ums_aligned = ums_signal[:len(psg_aligned)]
            offset_labels = labels[offset_sec // samples_per_epoch:]
            ums_start = 0
            offset_auto_labels = auto_labels[:len(psg_aligned)//samples_per_epoch]
            offset_sec_adjusted = -offset_sec
        else:
            offset_epochs = offset_sec // samples_per_epoch + 1
            psg_trim_start = offset_epochs * samples_per_epoch

            psg_aligned = psg_signal[psg_trim_start:]

            ums_start = psg_trim_start - offset_sec
            ums_epoch_start = ums_start//samples_per_epoch
            ums_epoch_end = (ums_start + len(psg_aligned))//samples_per_epoch
            
            ums_aligned = ums_signal[ums_start:ums_start + len(psg_aligned)]
            offset_labels = labels[offset_epochs:]
            offset_auto_labels = auto_labels[ums_epoch_start:ums_epoch_end]
            offset_sec_adjusted = -psg_trim_start
    length = min(len(psg_aligned), len(ums_aligned))
    final_len = min(len(offset_labels), len(offset_auto_labels), (length // samples_per_epoch)) * samples_per_epoch
    return (
        psg_aligned[:final_len],
        ums_aligned[:final_len],
        offset_sec_adjusted,
        [ums_start, ums_start + final_len],
        offset_labels[:final_len // samples_per_epoch],
        offset_auto_labels[:final_len // samples_per_epoch]
    )


def bandpass_filter(signal, low=0.5, high=30, fs=256):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal, fs):
    f, Pxx = welch(signal, fs=fs, nperseg=fs*5)
    def band_power(low, high):
        return np.sum(Pxx[(f >= low) & (f < high)])

    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 12)
    beta = band_power(12, 30)
    std = np.std(signal)
    zcr = np.sum(np.diff(np.sign(signal)) != 0)
    zcr_log = np.log1p(zcr)
    return np.array([delta, theta, alpha, beta, std, zcr_log])

def multi_feature_align(psg_template, ums_signal, fs=256, step_sec=1, filter_before=True, visualize=True):
    """
    使用多特征滑动匹配将 ums_signal 对齐 PSG 模板信号

    参数:
        psg_template: PSG 中的 N3 模板 (1D array, 单通道)
        ums_signal: UMS 信号 (1D array)
        fs: 采样率（Hz）
        step_sec: 滑动窗口步长（秒）
        filter_before: 是否先进行 0.5–30Hz 带通滤波
        visualize: 是否绘图查看匹配过程

    返回:
        best_lag (int): 最佳匹配起点（采样点）
        best_offset_sec (float): 最佳偏移时间（秒）
        matched_segment (1D array): 与模板匹配的 UMS 片段
    """
    if filter_before:
        psg_template = bandpass_filter(psg_template, 0.5, 30, fs)
        ums_signal = bandpass_filter(ums_signal, 0.5, 30, fs)
        psg_template = (psg_template - np.mean(psg_template)) / np.std(psg_template)
        ums_signal = (ums_signal - np.mean(ums_signal)) / np.std(ums_signal)

    template_feat = extract_features(psg_template, fs)
    window_len = len(psg_template)
    step = int(step_sec * fs)

    scores = []
    lags = []
    feat_list = []
    for i in range(0, len(ums_signal) - window_len + 1, step):
        window = ums_signal[i:i + window_len]
        feat = extract_features(window, fs)
        feat_list.append(feat)
        lags.append(i)
    feat_list = np.array(feat_list)
    scaler = StandardScaler()
    features = scaler.fit_transform(feat_list)  # 每一维都变成均值为0，方差为1
    for i, feat in enumerate(features):
        score = 1 - cosine(template_feat, feat)
        scores.append(score)
    scores = np.array(scores)
    print(scores)

    lags = np.array(lags)
    best_idx = np.argmax(scores)
    best_lag = lags[best_idx]
    best_offset_sec = best_lag / fs
    matched_segment = ums_signal[best_lag:best_lag + window_len]

    if visualize:
        plt.figure(figsize=(15, 4))
        plt.plot(lags / fs, scores, label='Cosine Similarity')
        plt.title("Multi-feature Matching Similarity Curve")
        plt.xlabel("Lag (seconds)")
        plt.ylabel("Similarity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.plot(psg_template, label="PSG Template", alpha=0.8)
        plt.subplot(2, 1, 2)
        plt.plot(matched_segment, label="UMS Matched Segment", alpha=0.8)
        plt.title("PSG Template vs UMS Matched Segment")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_lag, best_offset_sec, matched_segment

def plot_score(scores, best_idx, filename, save_root, sub_folder='figures', plot=False, save=False):
    if plot is True:
        # 可视化
        plt.plot(scores)
        plt.axvline(best_idx, color='r', linestyle='--', label='Best Match')
        plt.title("Cross-correlation with PSG SpO2")
        plt.grid(True)
        plt.legend()
        if save is True:
            save_figure(filename=filename, save_root=save_root, subfolder=sub_folder)
        else:
            plt.show()

def save_figure(filename, save_root, subfolder='figures'):
    """
    保存当前 Matplotlib 图像并关闭
    """
    fig_dir = os.path.join(save_root, subfolder)
    os.makedirs(fig_dir, exist_ok=True)
    full_path = os.path.join(fig_dir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()