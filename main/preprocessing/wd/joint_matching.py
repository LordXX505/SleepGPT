import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt

# ------------------------
# Desaturation detection
# ------------------------
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

# ------------------------
# Best SpO2 segment extractor
# ------------------------
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

# ------------------------
# Sleep stage similarity (cosine distance)
# ------------------------
# def sleep_stage_similarity(seg1, seg2):
#     scaler = StandardScaler()
#     seg1 = scaler.fit_transform(np.array(seg1).reshape(-1, 1)).flatten()
#     seg2 = scaler.fit_transform(np.array(seg2).reshape(-1, 1)).flatten()
#     score = 1 - cosine(seg1, seg2)
#     return score

# ------------------------
# Sleep stage similarity (F1-score)
# ------------------------

def sleep_stage_similarity(seg1, seg2, method='kappa'):
    """
    计算两个睡眠分期序列的相似度

    参数:
    - seg1, seg2: 输入的两个标签序列
    - method: 'kappa', 'f1', 或 'both'

    返回:
    - 一个相似度分数（或组合分数）
    """
    labels = [-1, 2, 3, 4, 5]
    if len(seg1) != len(seg2):
        raise ValueError("Segment lengths must match")

    if method == 'kappa':
        return cohen_kappa_score(seg1, seg2, labels=labels)
    elif method == 'f1':
        return f1_score(seg1, seg2, average='macro', labels=labels)
    elif method == 'both':
        kappa = cohen_kappa_score(seg1, seg2)
        f1 = f1_score(seg1, seg2, average='macro', labels=labels)
        return 0.5 * kappa + 0.5 * f1
    else:
        raise ValueError("Invalid method. Choose from 'kappa', 'f1', or 'both'.")

# ------------------------
# Joint alignment scoring
# ------------------------
def sliding_joint_alignment_score(ref_spo2, ref_stage, target_spo2, target_stage, fs=1):
    L = len(ref_spo2)  # 秒数
    ref_epoch_len = len(ref_stage)
    max_idx = len(target_spo2) - L 
    best_score = -np.inf
    best_idx = 0
    scores = []

    for i in range(0, max_idx):
        seg_spo2 = target_spo2[i:i+L]

        # 计算与seg_spo2对应的stage索引
        stage_start = i // (30 * fs)
        stage_end = stage_start + ref_epoch_len
        if stage_end > len(target_stage):
            continue

        seg_stage = target_stage[stage_start:stage_end]
        spo2_corr = np.corrcoef(ref_spo2, seg_spo2)[0, 1] if np.std(seg_spo2) > 1e-5 else 0
        stage_sim = sleep_stage_similarity(ref_stage, seg_stage, method='kappa')
        score = 0.1 * spo2_corr + 0.9 * stage_sim
        scores.append(score)

        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx, best_score, scores

def map_psg_labels_to_auto(psg_labels):
    """
    将PSG分期标签映射到Auto标签定义：
    0: Wake     -> 5
    1: N1       -> 3
    2: N2       -> 3
    3: N3       -> 2
    4: REM      -> 4
    """
    mapping = {0: 4, 1: 3, 2: 3, 3: 2, 4: 5}
    return np.array([mapping.get(label, -1) for label in psg_labels])




def joint_matching(psg_spo2, ums_spo2, psg_stage, ums_stage, segment_length_sec, fs=1, plot=False, max_delay_sec=90000, save=False):
    psg_start, psg_end = select_most_eventful_segment(psg_spo2, fs=fs, segment_length_sec=segment_length_sec)
    psg_spo2_seg = psg_spo2[psg_start:psg_end]
     # 对应PSG阶段
    epoch_start = psg_start // 30
    epoch_end = psg_end // 30
    psg_stage_seg = map_psg_labels_to_auto(psg_stage[epoch_start:epoch_end])

    best_idx, best_score, scores = sliding_joint_alignment_score(
        psg_spo2_seg, psg_stage_seg, ums_spo2, ums_stage
        )
    max_offset = int(max_delay_sec * fs)
    valid_range_start = max(psg_start - max_offset, 0)
    valid_range_end = min(psg_start + max_offset, len(scores))
    search_scores= scores[valid_range_start:valid_range_end]
    relative_best_idx = np.argmax(search_scores)
    best_idx = valid_range_start + relative_best_idx

    stage_start = best_idx // (30 * fs)
    stage_end = stage_start + len(psg_stage_seg)

    ums_aligned = ums_spo2[best_idx:best_idx + len(psg_spo2_seg)]
    psg_aligned = psg_spo2[psg_start:psg_end]
    stage_sim = sleep_stage_similarity(psg_stage_seg, ums_stage[stage_start:stage_end], method='kappa')
    template_events = detect_desaturation_events(psg_aligned, fs=fs)

    return ums_aligned, psg_aligned, best_idx, psg_start, template_events, scores, stage_sim
    