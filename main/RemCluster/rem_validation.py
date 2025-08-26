#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pyarrow import ipc

import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
from tqdm import tqdm
from collections import defaultdict


# -----------------------------
# Arrow 读取：返回 (C, T) 的 float32
# -----------------------------
def _to_1d_numpy_from_col(col):
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if isinstance(col, pa.Array):
        t = col.type
        if pa.types.is_floating(t) or pa.types.is_integer(t):
            return np.asarray(col.to_numpy(), dtype=np.float32)
        first = col[0].as_py()
        return np.asarray(first, dtype=np.float32)
    return np.asarray(col, dtype=np.float32)

def _fix_len_1d(x: np.ndarray, T: int = 3000, pad_value: float = np.nan):
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == T:
        return x
    if x.size > T:
        return x[:T]
    out = np.full(T, pad_value, dtype=np.float32)
    out[:x.size] = x
    return out

def read_arrow_matrix(path: str, T: int = 3000) -> np.ndarray:
    try:
        table = feather.read_table(path)
    except Exception:
        with pa.memory_map(path, "r") as source:
            table = ipc.open_file(source).read_all()
    if "x" in table.schema.names:
        col = table["x"]
        arr = _to_1d_numpy_from_col(col)
        arr = np.asarray(arr)
        if arr.ndim == 2:
            rows = [_fix_len_1d(arr[i], T=T) for i in range(arr.shape[0])]
            return np.stack(rows, axis=0)
        return _fix_len_1d(arr, T=T)[None, :]
    # 多列即多通道
    rows = []
    for i in range(table.num_columns):
        col = table.column(i)
        one = _to_1d_numpy_from_col(col)
        rows.append(_fix_len_1d(one, T=T))
    return np.stack(rows, axis=0)


# -----------------------------
# PSD 统计（Welch）
# -----------------------------
def psd_mean_over_epochs(mats, fs, nperseg=512, noverlap=256, fmax=None):
    """
    mats: list of (C,T)
    返回：
      f: (F,)
      psd_mean: (C,F)
    """
    assert len(mats) > 0
    C, T = mats[0].shape
    acc = None
    cnt = 0
    for mat in mats:
        assert mat.shape == (C, T)
        for ch in range(C):
            f, pxx = welch(mat[ch], fs=fs, nperseg=min(nperseg, T), noverlap=min(noverlap, max(0, T//2)))
            if fmax is not None:
                mask = (f <= fmax)
                f_use, pxx = f[mask], pxx[mask]
            else:
                f_use = f
            if acc is None:
                acc = np.zeros((C, len(f_use)), dtype=np.float64)
            acc[ch] += pxx
        cnt += 1
    psd_mean = acc / max(cnt, 1)
    return f_use, psd_mean


# -----------------------------
# 快速眼动（EOG）启发式检测
# 思路：0.3–5 Hz 带通 -> 一阶差分绝对值 -> z-score 阈值+不应期
# 返回该 epoch 的“眼动事件数”
# -----------------------------
def count_eog_saccades(x, fs, band=(0.3, 5.0), z_th=3.0, refractory=0.15):
    """
    x: 1D 信号
    fs: 采样率
    """
    # 带通
    b, a = butter(2, [band[0] / (fs/2), band[1] / (fs/2)], btype="bandpass")
    xf = filtfilt(b, a, x)
    # 一阶差分幅度
    d = np.abs(np.diff(xf))
    if d.std() < 1e-8:
        return 0
    z = (d - d.mean()) / (d.std() + 1e-8)
    peaks = np.where(z > z_th)[0]
    # 不应期合并
    if len(peaks) == 0:
        return 0
    events = [peaks[0]]
    min_gap = int(refractory * fs)
    for p in peaks[1:]:
        if p - events[-1] >= min_gap:
            events.append(p)
    return len(events)


# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录，如 /data/shhs_new")
    ap.add_argument("--result_dir", required=True, help="聚类输出目录（含 index.csv, labels.npy）")
    ap.add_argument("--subject", default="", help="可选：指定单个 subject；为空表示跨全体抽样")
    ap.add_argument("--fs", type=float, default=100.0, help="采样率 Hz（默认 100）")
    ap.add_argument("--per_label", type=int, default=200, help="每个标签抽取多少个 epoch 用于验证（单 subject 时会取 min）")
    ap.add_argument("--fmax", type=float, default=50.0, help="PSD 展示最高频（默认 50Hz）")
    ap.add_argument("--eog_ch", type=str, default="", help="EOG 通道索引(逗号分隔)，如 '2,3'；不填则对所有通道尝试并取最大计数")
    ap.add_argument("--save_dir", default="validation_out", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 载入 labels + index
    index_csv = os.path.join(args.result_dir, "index.csv")
    labels_npy = os.path.join(args.result_dir, "labels.npy")
    if not (os.path.exists(index_csv) and os.path.exists(labels_npy)):
        raise FileNotFoundError("result_dir 下找不到 index.csv 或 labels.npy")
    df = pd.read_csv(index_csv)
    labels = np.load(labels_npy)
    assert len(df) == len(labels), "index.csv 与 labels.npy 长度不一致"
    df["label"] = labels

    # 过滤 subject / 抽样
    if args.subject:
        df = df[df["subject"].astype(str) == str(args.subject)].copy()
        if df.empty:
            print(f"[warn] 找不到 subject={args.subject}")
            return
        # 每个标签取前 N 个（或随机抽样）
        picks = []
        for lab, sub in df.groupby("label"):
            n = min(len(sub), args.per_label)
            picks.append(sub.sort_values("epoch").head(n))
        df_pick = pd.concat(picks).sort_values(["label", "epoch"]).reset_index(drop=True)
    else:
        # 全体受试者随机抽样
        picks = []
        for lab, sub in df.groupby("label"):
            n = min(len(sub), args.per_label)
            picks.append(sub.sample(n=n, random_state=42))
        df_pick = pd.concat(picks).sort_values(["label", "subject", "epoch"]).reset_index(drop=True)

    # EOG 通道解析
    eog_idx = None
    if args.eog_ch.strip():
        eog_idx = [int(x) for x in args.eog_ch.split(",") if x.strip() != ""]

    # 读取 epoch -> 分 label 聚合
    mats_by_label = defaultdict(list)
    eog_counts_by_label = defaultdict(list)
    missing = 0

    # 推断 epoch 文件名的宽度（默认5）
    def infer_zero_width(subj_dir, default=5):
        try:
            files = [f for f in os.listdir(subj_dir) if f.endswith(".arrow")]
            if not files:
                return default
            w = max(len(os.path.splitext(f)[0]) for f in files)
            return max(w, default)
        except Exception:
            return default

    subj_cache_width = {}

    print(f"[info] 总计将读取 {len(df_pick)} 个 epoch 用于验证")
    for _, row in tqdm(df_pick.iterrows(), total=len(df_pick), ncols=100):
        sid = str(row["subject"])
        eid = int(row["epoch"])
        lab = int(row["label"])
        subj_dir = os.path.join(args.root, sid)
        if sid not in subj_cache_width:
            subj_cache_width[sid] = infer_zero_width(subj_dir, default=5)
        width = subj_cache_width[sid]
        arrow_path = os.path.join(subj_dir, f"{eid:0{width}d}.arrow")
        if not os.path.exists(arrow_path):
            missing += 1
            continue
        try:
            mat = read_arrow_matrix(arrow_path, T=int(args.fs*30))  # 30s * fs
        except Exception as e:
            print(f"[read err] {arrow_path}: {e}")
            continue

        mats_by_label[lab].append(mat)

        # 眼动计数（如果未指定 EOG，则对所有通道计算并取最大值作为该 epoch 的“EOG 活动”）
        if eog_idx is None:
            counts = [count_eog_saccades(mat[ch], args.fs) for ch in range(mat.shape[0])]
            eog_counts_by_label[lab].append(int(np.max(counts)))
        else:
            counts = [count_eog_saccades(mat[ch], args.fs) for ch in eog_idx if ch < mat.shape[0]]
            eog_counts_by_label[lab].append(int(np.max(counts) if counts else 0))

    if missing:
        print(f"[warn] 缺失 {missing} 个 .arrow 文件")

    # 至少要有两个标签的数据
    labs_present = sorted(mats_by_label.keys())
    if len(labs_present) < 2:
        print("[warn] 只拿到一个标签的数据，无法对比")
        return

    # -------- 平均 PSD 对比（每通道一张子图）--------
    # 为了作图整齐，只取两个标签中通道数的最小值
    C_min = min(min(m.shape[0] for m in mats_by_label[labs_present[0]]),
                min(m.shape[0] for m in mats_by_label[labs_present[1]]))

    f0, psd0 = psd_mean_over_epochs([m[:C_min] for m in mats_by_label[labs_present[0]]],
                                    fs=args.fs, fmax=args.fmax)
    f1, psd1 = psd_mean_over_epochs([m[:C_min] for m in mats_by_label[labs_present[1]]],
                                    fs=args.fs, fmax=args.fmax)

    fig, axes = plt.subplots(nrows=C_min, ncols=1, sharex=True, figsize=(10, 2.2*C_min))
    if C_min == 1:
        axes = [axes]
    for ch in range(C_min):
        ax = axes[ch]
        ax.semilogy(f0, psd0[ch] + 1e-20, label=f"label={labs_present[0]}")
        ax.semilogy(f1, psd1[ch] + 1e-20, label=f"label={labs_present[1]}")
        ax.set_ylabel(f"ch{ch}")
        ax.grid(True, which="both", alpha=0.3)
        if ch == 0:
            ax.legend()
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Mean PSD by label", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_psd = os.path.join(args.save_dir, "mean_psd_compare.png")
    fig.savefig(out_psd, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_psd}")

    # -------- EOG 眼动次数统计（箱线图 / 直方图）--------
    # 做个简单的箱线图 + 打印均值
    data0 = np.array(eog_counts_by_label[labs_present[0]], dtype=int)
    data1 = np.array(eog_counts_by_label[labs_present[1]], dtype=int)

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].boxplot([data0, data1], labels=[f"label={labs_present[0]}", f"label={labs_present[1]}"])
    ax[0].set_title("EOG saccade counts (boxplot)")
    ax[0].grid(True, alpha=0.3)

    bins = np.arange(0, max(data0.max() if len(data0) else 1, data1.max() if len(data1) else 1) + 2) - 0.5
    ax[1].hist(data0, bins=bins, alpha=0.6, label=f"label={labs_present[0]}")
    ax[1].hist(data1, bins=bins, alpha=0.6, label=f"label={labs_present[1]}")
    ax[1].set_title("EOG saccade counts (hist)")
    ax[1].set_xlabel("count / 30s")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_eog = os.path.join(args.save_dir, "eog_saccade_counts.png")
    fig.savefig(out_eog, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_eog}")

    # 保存一些统计摘要
    summary = {
        "n_used_label0": len(mats_by_label[labs_present[0]]),
        "n_used_label1": len(mats_by_label[labs_present[1]]),
        "eog_mean_label0": float(np.mean(data0)) if len(data0) else None,
        "eog_mean_label1": float(np.mean(data1)) if len(data1) else None,
        "eog_median_label0": float(np.median(data0)) if len(data0) else None,
        "eog_median_label1": float(np.median(data1)) if len(data1) else None,
        "subject_mode": bool(args.subject),
        "subject": args.subject if args.subject else None,
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {os.path.join(args.save_dir, 'summary.json')}")


if __name__ == "__main__":
    main()