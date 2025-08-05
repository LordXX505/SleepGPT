import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import maximum_filter1d, binary_closing

def detect_desats(spo2, fs=1, drop_thr=3, min_dur=10, merge_gap=30):
    """
    spo2      : 1-D array (%)
    fs        : 采样率 (Hz)；家用夹式探头通常 1 Hz
    drop_thr  : 判定阈值 (≥3 % or 4 %)
    min_dur   : 事件最短持续时间 (s)
    merge_gap : 相邻事件间 <merge_gap s 即合并
    return    : ODI, [(start_idx, end_idx), ...]
    """
    # Step-1  : 中值滤波去高频抖动
    win = 20 * fs
    s = maximum_filter1d(spo2, size=win,
                                 origin=-(win//2),  # 左对齐
                                 mode='nearest') - spo2 >= drop_thr
    # --- 1) 填掉小于 1 秒的“空洞” ---
    s = binary_closing(s, structure=2*np.ones(fs))  # 1 秒闭运算
    # 找区段边界
    diff = np.diff(np.concatenate([[0], s.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    # --- 3) 先合并 gap < merge_gap ---
    merged = []
    gap_pts = merge_gap * fs
    for st, ed in zip(starts, ends):
        if merged and st - merged[-1][1] <= gap_pts:
            merged[-1][1] = ed
        else:
            merged.append([st, ed])

    # --- 4) 再做 min_dur 过滤 ---
    min_len = min_dur * fs
    events = [(st, ed) for st, ed in merged if ed - st >= min_len]

    # --- 5) 计算 ODI ---
    odi = len(events) / (len(spo2) / fs / 3600)
    return odi, events


if __name__ == '__main__':
    spo2 = np.load("/Users/hwx_admin/Downloads/第一批/prep/WKC014-2020-2435spo2.npy")  # 或从 HDF5 读取
    odi, events = detect_desats(spo2, fs=1)
    print(f"ODI = {odi:.1f}  events = {len(events)}")
