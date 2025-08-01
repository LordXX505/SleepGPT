import os
import glob
import re
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tools import multi_feature_align, align_spo2_signals_with_offset, plot_manual_vs_auto_hypnogram, \
    plot_original_hypnogram, plot_score
from spo2 import align_ums_to_psg_by_spo2, plot_alignment_with_events
from joint_matching import joint_matching
import warnings
import logging
import mne
import h5py
import sys
from datetime import datetime
import json
from scipy.signal import butter, filtfilt, resample, resample_poly
from scipy.signal import welch

manual_dict = {
    '035-2020-2975': 510,
    '040-2020-2816': -1290,#
    '044-2020-2853': 630,
    '045-2020-2874': 90,
    '050-2020-2916': -90,
    '054-2020-2979': -90,
    '061-2020-3052': -150, #
    '064-2020-3086': -120, #
    '088-2021-152': 1380, #
    '099-2021-319': 210,
    'CAZ021-2020-2674': -360, #
    'CTT023-2020-2694': -900,
    'MYZ002': -1080,
    'WKC014-2020-2435': -2760, #
    'ZJP022-2020-2693': -1590, #
    'ZM005-2020-2373': -300, #
    '108-2021-815': 450,
    '110-2021-836': -150, #
    '130-2021-1013': -150,
    '131-2021-1012': 510,
    '134-2021-1056': -570,
    '148-2021-1187': 900,
    '186-2021-1527': 120,
    '198-2021-1617': 120, #
    '140-2021-1099': 60,
    '149-2021-1186': -30,
    '170-2021-1386': -60,
    "182-2021-1478": -60,
    "185-2021-1516": 180,
    '202-2021-1632': 270,
    "060-2020-3053": -60,
    "157-2021-1266": 120,
    "081-2021-60": 990,
    "094-2021-263": 0,
    'ZZJ008-2020-2389': -570,
    "080-2021-55": 60,
    "200-2021-1618": 660,
    '049-2020-2919': 510,
    '137-2021-1077': -90,
    '106-2021-805': 480,
}


def sort_edf_files(file_list):
    edf_dict = {}
    pattern = re.compile(r'\[(\d{3})\](\-T)?\.edf$')  # 匹配 [001].edf 或 [001]-T.edf
    for file in file_list:
        match = pattern.search(file)
        if match:
            index = int(match.group(1))
            is_t_version = match.group(2) == '-T'
            # 保留主版本（非 -T）优先
            if index not in edf_dict or (edf_dict[index]['is_t'] and not is_t_version):
                edf_dict[index] = {'filename': file, 'is_t': is_t_version}

    # 按编号排序并返回文件名
    sorted_files = [edf_dict[k]['filename'] for k in sorted(edf_dict.keys())]
    return sorted_files


def read_and_concatenate_edf_with_mne(sorted_files, data_dir, preload=True, verbose=False):
    raws = []
    for fname in sorted_files:
        path = os.path.join(data_dir, fname)
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=verbose)
        raws.append(raw)

    concatenated_raw = mne.concatenate_raws(raws)
    return concatenated_raw


import xml.etree.ElementTree as ET


def parse_rml_stages(rml_path, epoch_duration=30, total_duration_sec=None, map_to_int=True):
    """
    从 RML 文件中解析 UserStaging → NeuroAdultAASMStaging → Stage 睡眠分期信息，
    并展开为每 30 秒一个标签的列表
    """
    tree = ET.parse(rml_path)
    root = tree.getroot()

    # 提取命名空间
    ns = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    # 使用带命名空间的 XPath
    user_staging = root.find('.//ns:UserStaging', ns)
    if user_staging is None:
        raise ValueError("找不到 <UserStaging> 节点")

    # 然后进入 NeuroAdultAASMStaging 节点
    staging_block = user_staging.find('.//ns:NeuroAdultAASMStaging', ns)
    if staging_block is None:
        raise ValueError("找不到 <NeuroAdultAASMStaging> 节点")

    # 获取所有 <Stage> 标签
    stage_nodes = staging_block.findall('.//ns:Stage', ns)

    stage_times = []
    stage_labels = []

    for node in stage_nodes:
        label = node.attrib.get('Type')
        start = int(node.attrib.get('Start'))
        stage_times.append(start)
        stage_labels.append(label)

    # 推断总时长
    if total_duration_sec is None:
        total_duration_sec = stage_times[-1] + epoch_duration

    total_epochs = total_duration_sec // epoch_duration
    labels = [None] * int(total_epochs)

    # 展开阶段标签
    for i in range(len(stage_times)):
        start_sec = stage_times[i]
        start_idx = start_sec // epoch_duration

        end_sec = stage_times[i + 1] if i + 1 < len(stage_times) else total_duration_sec
        end_idx = end_sec // epoch_duration

        for j in range(start_idx, end_idx):
            labels[j] = stage_labels[i]
        # 标签映射
    if map_to_int:
        stage_map = {
            'Wake': 0,
            'NonREM1': 1,
            'NonREM2': 2,
            'NonREM3': 3,
            'REM': 4
        }
        labels = [stage_map.get(label, -1) for label in labels]  # -1 表示未识别或 NotScored
    return labels


from scipy.signal import correlate


def align_psg_ums(psg_signal, ums_signal, labels, auto_labels, save_root, sub_name, subject_name, fs=256, stage=3,
                  n_epochs=1,
                  epoch_duration=30, using_multi_f='', segment_length_sec=9000,
                  plot=False, save=False):
    """
    用 PSG 的某个 N3 段与 UMS 做滑动互相关匹配，实现自动对齐

    参数:
        psg_signal: 1D numpy array，PSG 信号（已重采样到 256Hz）
        ums_signal: 1D numpy array，UMS 信号（256Hz）
        labels: list[int]，每个 epoch 的 PSG 标签（0~4）
        fs: 采样率（默认 256Hz）
        stage: 匹配的目标阶段（默认 N3，对应标签 3）
        n_epochs: 用多少个连续 epoch 做模板（建议 3~5）

    返回:
        offset_samples: ums 相对于 psg 的样本偏移
        offset_seconds: ums 相对于 psg 的秒偏移
        aligned_psg: 与 ums 对齐后的 PSG 信号（截断后）
        aligned_ums: 对齐后的 ums 信号
    """
    epoch_len = 30 * fs

    # # 找到第一个连续的 N3 区段（>= n_epochs）

    if using_multi_f == 'mf':
        stage_indices = [i for i, label in enumerate(labels) if label == stage]
        if not stage_indices:
            raise ValueError(f"找不到标签为 {stage} 的 PSG 阶段")
        for i in range(len(stage_indices) - n_epochs + 1):
            if stage_indices[i + n_epochs - 1] - stage_indices[i] == n_epochs - 1:
                template_start = stage_indices[i] * epoch_len
                template_end = template_start + n_epochs * epoch_len
                break
        else:
            raise ValueError(f"没有找到连续 {n_epochs} 个 {stage} 阶段的片段")
        # 提取 PSG 模板信号
        psg_template = psg_signal[template_start:template_end]
        psg_n3_start_sec = template_start // fs
        best_lag, best_offset_sec, matched_segment = multi_feature_align(psg_template, ums_signal[:2000000])
        lag = best_lag
    elif using_multi_f == 'spo2':
        ums_aligned, psg_aligned, best_idx, psg_start, template_events = align_ums_to_psg_by_spo2(ums_signal,
                                                                                                  psg_signal,
                                                                                                  segment_length_sec=segment_length_sec,
                                                                                                  plot=plot)
        if plot is True:
            plot_alignment_with_events(ums_aligned, psg_aligned, template_events, sub_folder=sub_name,
                                       save_root=save_root, filename='alignment_with_events', save=save)
        offset_sec = best_idx - psg_start
        logger.info(f'offset_set: {offset_sec}')
        psg_aligned, ums_aligned, offset_sec_adjusted, ums_st_ed, offset_labels, offset_auto_labels = align_spo2_signals_with_offset(
            psg_signal, ums_signal, offset_sec=offset_sec, labels=labels, auto_labels=auto_labels, fs=1
        )
        if plot is True:
            plot_original_hypnogram(labels, auto_labels)
            plot_manual_vs_auto_hypnogram(offset_labels, offset_auto_labels)
        return psg_aligned, ums_aligned, offset_sec_adjusted, ums_st_ed, offset_labels
    elif using_multi_f == 'joint_matching':
        ums_aligned, psg_aligned, best_idx, psg_start, template_events, scores, stage_sim = joint_matching(psg_signal,
                                                                                                           ums_signal,
                                                                                                           labels,
                                                                                                           auto_labels,
                                                                                                           segment_length_sec=segment_length_sec,
                                                                                                           plot=plot,
                                                                                                           save=save)
        # best_idx = best_idx - 150
        if subject_name in manual_dict.keys():
            best_idx = psg_start + manual_dict[subject_name]
            logger.info(f'{subject_name}: manual - {manual_dict[subject_name]}, best_idx: {best_idx}')
        # best_idx = psg_start + 510

        if plot is True:
            plot_score(scores=scores, best_idx=best_idx, sub_folder=sub_name, save_root=save_root,
                       filename='plot_scores', plot=plot, save=save)
            plot_alignment_with_events(ums_aligned, psg_aligned, template_events, sub_folder=sub_name,
                                       save_root=save_root, filename='alignment_with_events', save=save)
        logger.info(f'bestidx: {best_idx}, psg_start: {psg_start}')
        offset_sec = best_idx - psg_start
        logger.info(f'offset_set: {offset_sec}')
        psg_aligned, ums_aligned, offset_sec_adjusted, ums_st_ed, offset_labels, offset_auto_labels = align_spo2_signals_with_offset(
            psg_signal, ums_signal, offset_sec=offset_sec, labels=labels, auto_labels=auto_labels, fs=1
        )
        if plot is True:
            plot_original_hypnogram(labels[psg_start // 30:(psg_start + segment_length_sec) // 30],
                                    auto_labels[best_idx // 30:(best_idx + segment_length_sec) // 30],
                                    sub_folder=sub_name, save_root=save_root, filename='orig_hyp_match_labels',
                                    save=save)
            plot_original_hypnogram(labels, auto_labels, sub_folder=sub_name, save_root=save_root, filename='orig_hyp',
                                    save=save)
            plot_manual_vs_auto_hypnogram(offset_labels, offset_auto_labels, sub_folder=sub_name, save_root=save_root,
                                          filename='manual_vs_auto_hypnogram', save=save)
        np.savez(file=f"{os.path.join(save_root, 'compare_label.npz')}", labels=offset_labels,
                 auto_labels=offset_auto_labels)
        return psg_aligned, ums_aligned, offset_sec_adjusted, ums_st_ed, offset_labels, stage_sim

    else:
        logger.info(f'using_multi_f: {using_multi_f}')

        def bandpass_filter(data, low=0.5, high=4.5, fs=256):
            b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
            return filtfilt(b, a, data)

        # 滤波
        psg_filtered = bandpass_filter(psg_template)
        ums_filtered = bandpass_filter(ums_signal)

        # 标准化
        template_norm = (psg_filtered - np.mean(psg_filtered)) / np.std(psg_filtered)
        ums_norm = (ums_filtered - np.mean(ums_filtered)) / np.std(ums_filtered)

        ums_norm = ums_norm[:100000]
        # 滑动互相关匹配
        corr = correlate(ums_norm, template_norm, mode='valid')
        logger.info(len(corr))

        plt.plot(corr)
        plt.title("Cross-correlation with PSG N3")
        plt.xlabel("Lag (samples)")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.show()

        lag = np.argmax(corr)

        # 假设 psg_template, ums_signal, lag, fs 已定义
        window_len = len(psg_template)
        matched_ums = ums_norm[lag:lag + window_len]

        plt.figure(figsize=(14, 4))
        plt.subplot(2, 1, 1)
        plt.plot(template_norm, label="PSG N3 Template", alpha=0.8)
        plt.subplot(2, 1, 2)
        plt.plot(matched_ums, label="UMS Matched Segment", alpha=0.8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    logger.info('lag: {lag}')
    offset_samples = lag - psg_n3_start_sec * fs
    samples_per_epoch = epoch_duration * fs

    if offset_samples > 0:
        # UMS 晚启动，需要丢掉前面 offset 秒的 UMS，使其对齐 PSG t=0
        ums_aligned = ums_signal[offset_samples:]
        ums_aligned_len = len(ums_aligned) // samples_per_epoch * samples_per_epoch
        psg_aligned = psg_signal[:ums_aligned_len]
        offset_sec_adjusted = offset_samples / fs
        num_epochs = len(ums_aligned) // samples_per_epoch
        offset_labels = labels[:num_epochs]
    else:
        # UMS 比 PSG 早启动，需要将 PSG 向后对齐到 epoch 起点
        offset_samples = abs(offset_samples)
        if offset_samples % samples_per_epoch == 0:
            # PSG offset 刚好落在 epoch 边界
            psg_aligned = psg_signal[offset_samples:]
            ums_aligned = ums_signal[:len(psg_aligned)]
            offset_sec_adjusted = -offset_samples / fs
            offset_labels = labels[offset_samples:]
        else:
            # PSG offset 落在 epoch 中间，向上对齐到下一个 epoch 起点
            offset_epochs = offset_samples // samples_per_epoch + 1
            psg_trim_start = offset_epochs * samples_per_epoch
            start_offset_in_ums = psg_trim_start - offset_samples

            psg_aligned = psg_signal[psg_trim_start:]
            ums_aligned = ums_signal[start_offset_in_ums:start_offset_in_ums + len(psg_aligned)]
            offset_sec_adjusted = -psg_trim_start / fs
            offset_labels = labels[offset_epochs:]

    # 截断为整 epoch 长度
    final_len = (len(psg_aligned) // samples_per_epoch) * samples_per_epoch

    return psg_aligned[:final_len], ums_aligned[:final_len], offset_sec_adjusted, offset_labels[:(
            len(psg_aligned) // samples_per_epoch)]


def plot_aligned_signals(psg_signal, ums_signal, fs=256, title="Aligned Signals"):
    """
    可视化对齐后的 PSG 和 UMS 信号（默认采样率为 256 Hz）

    参数：
        psg_signal: 对齐后的 PSG 信号（1D array）
        ums_signal: 对齐后的 UMS 信号（1D array）
        fs: 采样率
    """
    assert len(psg_signal) == len(ums_signal), "信号长度不一致！"

    time = np.arange(len(psg_signal)) / fs

    plt.figure(figsize=(15, 4))
    plt.subplot(2, 1, 1)
    plt.plot(time, psg_signal, label="PSG", color='blue')
    plt.title("PSG Signal")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 下图：UMS
    plt.subplot(2, 1, 2)
    plt.plot(time, ums_signal, label="UMS", color='orange')
    plt.title("UMS Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main(items, root_path, psg_root_path, ums_root_path, logger):
    stage_sims = []

    records = []
    for item in items:
        sub_name = item.split('.')[0]
        if sub_name.startswith('S1'):
            psg_path = glob.glob(os.path.join(psg_root_path, sub_name[3:] + '*'))
        elif sub_name.startswith('028'):
            psg_sub_name = sub_name[:8] + '-' + sub_name[8:]
            psg_path = glob.glob(os.path.join(psg_root_path, psg_sub_name + '*'))
        elif sub_name.startswith('102-2021-361'):
            psg_path = glob.glob(os.path.join(psg_root_path, '102-2021-359' + '*'))
        elif sub_name.startswith('079-2121-34'):
            psg_path = glob.glob(os.path.join(psg_root_path, '079-2021-34' + '*'))
        elif sub_name.startswith('077-2121-35'):
            psg_path = glob.glob(os.path.join(psg_root_path, '077-2021-35' + '*'))
        elif sub_name.startswith('183-2021-1492'):
            psg_path = glob.glob(os.path.join(psg_root_path, '183-2021-1470' + '*'))
        elif sub_name.startswith('LRY') and sub_name.endswith('2336'):
            sub_name = 'LRY001'
            psg_sub_name = 'LRY001 -2020-2336'
            psg_path = glob.glob(os.path.join(psg_root_path, psg_sub_name + '*'))
        elif sub_name.startswith('145-2021-1165spo2'):
            sub_name = '165-2021-1341'
            psg_path = glob.glob(os.path.join(psg_root_path, sub_name + '*'))
        else:
            psg_path = glob.glob(os.path.join(psg_root_path, sub_name + '*'))
        ums_path = os.path.join(ums_root_path, sub_name + 'spo2.npy')
        ums_rawdata_path = os.path.join(ums_root_path, sub_name + '.npy')
        save_root = os.path.join(root_path, 'data_new', sub_name)
        ods_path = os.path.join(ums_root_path, sub_name + '_ods.npy')
        # if not sub_name.startswith('145-2021-1165'):
        #     continue
        # if sub_name != "035-2020-2975":
        #     continue
        # if sub_name not in manual_dict.keys():
        #     continue
        if psg_path and sub_name:
            psg_path = psg_path[0]
            logger.info(f'psg_path: {psg_path}')
            psg_file_paths = os.listdir(psg_path)
            for pfp in psg_file_paths:
                if not pfp.endswith("Store"):
                    psg_file_path = pfp
                    break
            files = os.listdir(os.path.join(psg_path, psg_file_path))
            sorted_files = sort_edf_files(files)
            logger.info(sorted_files)
            psg_data = read_and_concatenate_edf_with_mne(sorted_files, os.path.join(psg_path, psg_file_path))
            psg_len = float(psg_data.n_times / psg_data.info["sfreq"])
            try:
                ums_data = np.load(ums_path)
            except:
                continue
            # ums_len = float(len(ums_data)/1)
            # overlapping_path = os.path.join(root_path, 'data', sub_name)
            # try:
            #     with h5py.File(os.path.join(overlapping_path, 'data.h5'), 'r') as hf:
            #         overlapping_len = hf['signal'].shape[0]
            #     records.append({
            #         "name": sub_name,
            #         "psg_len": psg_len,
            #         "ums_len": ums_len,
            #         'overlapping':overlapping_len,
            #     })
            # except:
            #     continue
            logger.info(psg_data.info)
            # 优先通道
            preferred_channels = ['SpO2']
            psg_channel = None
            for ch in preferred_channels:
                if ch in psg_data.ch_names:
                    psg_channel = ch
                    break
            if psg_channel is None:
                raise ValueError("未找到合适的 PSG 通道进行与 UMS 对齐")
            logger.info(f"将使用通道 {psg_channel} 进行对齐")
            # raw_psg 是 mne.io.Raw 对象，当前采样率为 1000 Hz
            raw_psg_resampled = psg_data.copy().resample(1, npad='auto')
            logger.info(f"原始采样率: {psg_data.info['sfreq']} Hz")
            logger.info(f"重采样后采样率: {raw_psg_resampled.info['sfreq']} Hz")

            psg_signal = raw_psg_resampled.get_data(picks=psg_channel)[0]
            rml_path = glob.glob(os.path.join(psg_path, psg_file_path, '*.rml'))[0]
            try:
                labels = parse_rml_stages(rml_path, total_duration_sec=len(psg_signal))
                ums_data = np.load(os.path.join(ums_path))
                ums_labels = np.load(os.path.join(ums_root_path, sub_name + '_stage.npy'))
                logger.info(f"{len(ums_data)}, {len(psg_signal)}")
                psg_aligned, ums_aligned, offset_sec_adjusted, ums_st_ed, offset_labels, stage_sim = align_psg_ums(
                    psg_signal, ums_data, labels, ums_labels,
                    using_multi_f='joint_matching', plot=True, save_root=save_root, sub_name='plot_figures', save=True,
                    subject_name=sub_name)
                offset_labels = np.array(offset_labels)
                stage_sims.append({'sub_name': sub_name, 'scores': stage_sim})
                # 原始采样率 & 目标采样率
                fs_orig = 256
                fs_target = 100
                epoch_sec = 30
                target_len = fs_target * epoch_sec  # 100 * 30 = 3000

                # 1. 读取并裁剪原始 raw data（256Hz）
                ods_label = np.load(ods_path)

                ums_rawdata = np.load(ums_rawdata_path)  # shape: (total_points,)
                ums_rawdata_start = ums_st_ed[0] * fs_orig
                ums_rawdata_end = ums_st_ed[1] * fs_orig
                ums_rawdata = ums_rawdata[ums_rawdata_start:ums_rawdata_end]  # 1D array
                ods_label = ods_label[ums_st_ed[0]//30: ums_st_ed[1]//30]
                def bandpass_filter(sig, fs, lowcut=0.1, highcut=35.0, order=4):
                    nyq = 0.5 * fs
                    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
                    return filtfilt(b, a, sig, padlen=3 * max(len(b), len(a)))

                ums_filt = bandpass_filter(ums_rawdata, fs_orig)
                upsample = fs_orig
                downsample = fs_target
                ums_filt_ds = resample_poly(ums_filt, up=downsample, down=upsample)
                # # 1) Welch PSD —— 原始 256 Hz
                # f_raw, pxx_raw = welch(ums_rawdata, fs=fs_orig, nperseg=4 * fs_orig)
                # # 2) Welch PSD —— 滤波 + 下采 100 Hz
                # f_ds, pxx_ds = welch(ums_filt_ds, fs=fs_target, nperseg=4 * fs_target)
                #
                # plt.figure(figsize=(7, 4))
                #
                # # 原始 256 Hz
                # plt.semilogy(f_raw, pxx_raw, label="Raw 256 Hz")
                #
                # # 滤波 + 下采 100 Hz
                # plt.semilogy(f_ds, pxx_ds, label="Filtered & DS 100 Hz")
                #
                # plt.title("PSD before vs after filtering & down-sampling")
                # plt.xlabel("Frequency (Hz)")
                # plt.ylabel("Power spectral density")
                # plt.xlim(0, 60)
                # plt.grid(True)
                # plt.legend()
                # plt.tight_layout()
                # plt.show()

                # 3. 划分为 epochs（每段 30s * 100Hz = 3000 点）
                num_epochs = len(ums_filt_ds) // target_len
                ums_epochs = ums_filt_ds[:num_epochs * target_len].reshape(num_epochs, target_len)
                ums_aligned = ums_aligned.reshape(num_epochs, -1)
                # ums_filt_ds = ums_filt_ds[:num_epochs * target_len]
                # ums_epochs = ums_rawdata_ds.reshape(num_epochs, target_len)  # shape: (N, 3000)
                assert num_epochs == len(offset_labels), f'{num_epochs}, {len(offset_labels)}'
                assert num_epochs == len(ods_label), f'{num_epochs}, {len(ods_label)}'
                assert ums_aligned.shape[1] == 30,  f'{ums_aligned.shape[1]}'
                # 4. 筛选有效标签
                valid_idx = offset_labels != -1
                ums_epochs = ums_epochs[valid_idx]
                offset_labels = offset_labels[valid_idx]
                ods_label = ods_label[valid_idx]
                ums_aligned = ums_aligned[valid_idx]
                # 5. 保存到 HDF5 文件
                h5_path = f"{save_root}/data.h5"
                os.makedirs(save_root, exist_ok=True)
                with h5py.File(h5_path, 'w') as f:
                    f.create_dataset('signal', data=ums_epochs, compression='gzip')
                    f.create_dataset('stage', data=offset_labels.astype(np.int32))
                    f.create_dataset('spo2',  data=ums_aligned, compression='gzip')
                    f.create_dataset('ods',  data=ods_label.astype(np.int32))
                    f.create_dataset('best_idx', data=ums_st_ed)

            except Exception as e:
                logger.exception(f'Exception: {e}, {psg_path}')
    print(stage_sims)

    out_json = os.path.join(root_path, "length_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ 写入完成：{out_json}  共 {len(records)} 条")


if __name__ == '__main__':
    for name in ['第一批', '第二批']:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        mne.set_log_level('ERROR')
        logging.getLogger('mne').setLevel(logging.CRITICAL)

        root_path = os.path.join('/Users/hwx_admin/Downloads', name)
        check_root_path = os.path.join('/Users/hwx_admin/Downloads', name, 'prep')
        psg_root_path = os.path.join(root_path, 'PSG')
        ums_root_path = os.path.join(root_path, 'prep')
        items = os.listdir(check_root_path)

        log_dir = os.path.join(root_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # 日志文件名（包含时间戳）
        log_filename = datetime.now().strftime('preprocessing_%Y%m%d_%H%M%S.log')
        log_path = os.path.join(log_dir, log_filename)

        # 配置 logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode='w', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)  # 同时输出到控制台
            ]
        )

        logger = logging.getLogger(__name__)

        main(items=items, root_path=root_path, psg_root_path=psg_root_path,
             ums_root_path=ums_root_path, logger=logger)
