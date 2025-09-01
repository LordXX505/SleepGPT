# rem_patch_psd_diag.py
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyarrow as pa
from scipy.signal import welch
from scipy.stats import ttest_ind
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ----------------------------
# I/O helpers
# ----------------------------
def infer_zero_width(subj_dir: str, default: int = 5) -> int:
    """根据目录中已存在的 .arrow 文件推断 epoch 文件名的零填充宽度（如 00001.arrow → 5）"""
    try:
        p = Path(subj_dir)
        cands = list(p.glob("*.arrow"))
        if not cands:
            return default
        name = cands[0].stem  # e.g., "00001"
        return len(name)
    except Exception:
        return default


def read_arrow_matrix(path: str) -> np.ndarray:
    """
    读取单个 epoch 的原始矩阵，期望返回形状 (C, T)。
    你的数据结构之前描述为 channel x 3000。
    """
    try:
        reader = pa.ipc.RecordBatchFileReader(pa.memory_map(path, "r"))
        tbl = reader.read_all()
    except Exception as e:
        raise RuntimeError(f"Error reading PyArrow file {path}: {e}")

    # 尝试列名 'x'，否则退化为第 0 列
    try:
        col = tbl.column("x")
    except KeyError:
        col = tbl.columns[0]

    # 转 numpy (C, T)
    if isinstance(col, pa.ChunkedArray):
        arrs = []
        for chunk in col.chunks:
            arrs.extend(chunk.to_pylist())
        mat = np.asarray(arrs, dtype=np.float32)
    else:
        mat = np.asarray(col.to_pylist(), dtype=np.float32)

    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    return mat  # (C, T)


# ----------------------------
# 抽样策略
# ----------------------------
def stratified_by_label(df_idx: pd.DataFrame, per_cluster: int, seed: int = 42) -> pd.DataFrame:
    """按 label 分层抽样，每个簇最多取 per_cluster 个 patch"""
    rng = np.random.default_rng(seed)
    parts = []
    for lab, g in df_idx.groupby("label"):
        n = min(per_cluster, len(g))
        parts.append(g.sample(n=n, random_state=int(rng.integers(1_000_000_000))))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_by_label_pid(df_idx: pd.DataFrame, per_bucket: int, seed: int = 42) -> pd.DataFrame:
    """按 (label, pid) 分层抽样，每个桶最多取 per_bucket 个，覆盖 15 个 patch 位置"""
    rng = np.random.default_rng(seed)
    parts = []
    for (lab, pid), g in df_idx.groupby(["label", "pid"]):
        n = min(per_bucket, len(g))
        parts.append(g.sample(n=n, random_state=int(rng.integers(1_000_000_000))))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ----------------------------
# PSD 计算
# ----------------------------
def slice_patch(signal_1d: np.ndarray, pid: int, fs: int, patch_sec: float = 2.0) -> np.ndarray:
    """
    从 30s epoch 中切出第 pid 个 patch（0-based），每个 patch = 2s。
    假设 epoch 总长度 T=3000（fs=100）→ 15 个 patch × 200 点。
    """
    seg_len = int(round(patch_sec * fs))  # e.g., 200
    start = pid * seg_len
    end = start + seg_len
    if end > signal_1d.shape[-1]:
        end = signal_1d.shape[-1]
    return signal_1d[start:end]


def compute_psd_batch(batch_signals: np.ndarray, fs: int, nperseg: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    对一批 1D 信号（形状 [N, T]）计算 Welch PSD，返回 (freqs, psd_matrix[dB])
    """
    psd_list = []
    freqs_ref = None
    for sig in batch_signals:
        nps = min(nperseg, len(sig))
        if nps < 8:  # 信号太短则跳过
            continue
        f, Pxx = welch(sig, fs=fs, nperseg=nps, detrend="constant", scaling="density")
        if freqs_ref is None:
            freqs_ref = f
        else:
            # 若不同（极少见），就按最短长度对齐
            m = min(len(freqs_ref), len(f))
            freqs_ref = freqs_ref[:m]; Pxx = Pxx[:m]
        Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20))
        psd_list.append(Pxx_db)
    if not psd_list:
        return None, None
    psd = np.stack(psd_list, axis=0)  # [N, F]
    return freqs_ref, psd


# ----------------------------
# 聚合（只对抽样后的 df_pick）
# ----------------------------
def lazy_arrow_path(root: str, sid: str, eid: int, width_cache: dict) -> Optional[str]:
    """
    懒计算 .arrow 路径：优先使用 width_cache 中该 subject 的零填充宽度；
    若没有则现场推断并缓存。
    """
    if sid not in width_cache:
        subj_dir = os.path.join(root, sid)
        width_cache[sid] = infer_zero_width(subj_dir, default=5)
    width = width_cache[sid]
    apath = os.path.join(root, sid, f"{eid:0{width}d}.arrow")
    return apath if os.path.exists(apath) else None


def aggregate_modal_psd(
    root: str,
    df_pick: pd.DataFrame,  # 仅抽样后的数据：含 columns [subject, epoch, pid, label]，可选 [arrow_path]
    fs: int,
    eeg_idx: List[int],
    eog_idx: List[int],
    emg_idx: List[int],
    nperseg: int = 128,
    patch_sec: float = 2.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    从抽样的 patch 列表读取原始 epoch，切出 patch，按模态求 PSD。
    - EEG/EOG/EMG：多通道时先通道平均为 1D，再切 patch。
    返回：{"EEG": (f, psd0, psd1), ...}
    """
    modal_signals = {
        "EEG": {0: [], 1: []},
        "EOG": {0: [], 1: []},
        "EMG": {0: [], 1: []},
    }
    width_cache = {}

    for _, row in tqdm(df_pick.iterrows(), total=len(df_pick), desc="Load patches"):
        sid = str(row["subject"])
        eid = int(row["epoch"])
        pid = int(row["pid"])
        lab = int(row["label"])

        apath = row["arrow_path"] if "arrow_path" in row and isinstance(row["arrow_path"], str) else None
        if not apath or not os.path.exists(apath):
            apath = lazy_arrow_path(root, sid, eid, width_cache)
            if apath is None:
                # 找不到就跳过
                continue

        try:
            mat = read_arrow_matrix(apath)  # (C, T)
        except Exception:
            continue

        # EEG
        if eeg_idx:
            try:
                eeg_avg = mat[eeg_idx, :].mean(axis=0)
                patch = slice_patch(eeg_avg, pid, fs, patch_sec=patch_sec)
                modal_signals["EEG"][lab].append(patch)
            except Exception:
                pass

        # EOG
        if eog_idx:
            try:
                eog_avg = mat[eog_idx, :].mean(axis=0)
                patch = slice_patch(eog_avg, pid, fs, patch_sec=patch_sec)
                modal_signals["EOG"][lab].append(patch)
            except Exception:
                pass

        # EMG
        if emg_idx:
            try:
                emg_avg = mat[emg_idx, :].mean(axis=0)
                patch = slice_patch(emg_avg, pid, fs, patch_sec=patch_sec)
                modal_signals["EMG"][lab].append(patch)
            except Exception:
                pass

    out = {}
    for key in ["EEG", "EOG", "EMG"]:
        lst0 = modal_signals[key][0]
        lst1 = modal_signals[key][1]
        if len(lst0) == 0 or len(lst1) == 0:
            continue
        sig0 = np.stack(lst0, axis=0)  # [N0, T]
        sig1 = np.stack(lst1, axis=0)  # [N1, T]
        f, psd0 = compute_psd_batch(sig0, fs, nperseg=nperseg)
        f2, psd1 = compute_psd_batch(sig1, fs, nperseg=nperseg)
        if f is None or f2 is None:
            continue
        if not np.allclose(f, f2):
            m = min(len(f), len(f2))
            f, psd0, psd1 = f[:m], psd0[:, :m], psd1[:, :m]
        out[key] = (f, psd0, psd1)
    return out


# ----------------------------
# 绘图（均值±std + p 值）
# ----------------------------
def plot_modal_psd_with_pvalue(f, psd0, psd1, title, out_png):
    """
    画 2×1：上面是两组 PSD 的均值±标准差；下面是 -log10(p)，并用阴影标出 p<0.05 的区域。
    """
    m0, s0 = psd0.mean(axis=0), psd0.std(axis=0)
    m1, s1 = psd1.mean(axis=0), psd1.std(axis=0)

    # 逐频点 Welch t-test
    pvals = np.array([ttest_ind(psd0[:, i], psd1[:, i], equal_var=False).pvalue for i in range(len(f))])
    sig = pvals < 0.05

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图：均值±std
    axes[0].plot(f, m0, label=f"Cluster 0 (n={psd0.shape[0]})")
    axes[0].fill_between(f, m0 - s0, m0 + s0, alpha=0.2)
    axes[0].plot(f, m1, label=f"Cluster 1 (n={psd1.shape[0]})")
    axes[0].fill_between(f, m1 - s1, m1 + s1, alpha=0.2)
    axes[0].set_ylabel("PSD (dB/Hz)")
    axes[0].set_title(title)
    axes[0].legend()

    # 下图：-log10(p) + 显著性阴影
    y = -np.log10(np.maximum(pvals, 1e-300))
    axes[1].plot(f, y, label="-log10(p)")
    axes[1].fill_between(f, 0, y, where=sig, color="red", alpha=0.25, label="p<0.05")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("-log10(p)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return pvals, m0, m1, s0, s1


# ----------------------------
# 主流程
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Patch-level PSD diagnosis with per-frequency p-values (sample-then-read)")
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录（<root>/<subject>/<epoch>.arrow）")
    ap.add_argument("--result_dir", required=True, help="包含 patch_labels.npy 与 patch_index.csv 的目录")
    ap.add_argument("--fs", type=int, default=100, help="采样率 Hz（默认 100）")
    ap.add_argument("--nperseg", type=int, default=128, help="Welch nperseg（2s patch 建议 ≤ 200）")

    # 抽样相关
    ap.add_argument("--sample_mode", type=str, default="label", choices=["label", "label_pid"],
                    help="'label'：每簇抽样；'label_pid'：对每个(label,pid)桶抽样")
    ap.add_argument("--per_cluster", type=int, default=5000, help="sample_mode=label 时，每簇抽样上限")
    ap.add_argument("--per_bucket", type=int, default=300, help="sample_mode=label_pid 时，每个(label,pid)桶抽样上限")
    ap.add_argument("--seed", type=int, default=42)

    # 通道索引（基于 mat 的第 0 维）；可传多个
    ap.add_argument("--eeg_idx", type=int, nargs="*", default=[], help="EEG 通道索引，例如 --eeg_idx 0 1")
    ap.add_argument("--eog_idx", type=int, nargs="*", default=[], help="EOG 通道索引，例如 --eog_idx 2 3")
    ap.add_argument("--emg_idx", type=int, nargs="*", default=[], help="EMG 通道索引，例如 --emg_idx 4")

    args = ap.parse_args()
    out_dir = os.path.join(args.result_dir, "diag_patch_psd")
    os.makedirs(out_dir, exist_ok=True)

    # 读取 patch-level 标签与索引
    labels_path = os.path.join(args.result_dir, "patch_labels.npy")
    index_path = os.path.join(args.result_dir, "patch_index.csv")
    if not (os.path.exists(labels_path) and os.path.exists(index_path)):
        raise FileNotFoundError("需要 patch_labels.npy 与 patch_index.csv")

    labels = np.load(labels_path)
    df_idx = pd.read_csv(index_path)  # 期望列：subject,epoch,pid
    must_cols = {"subject", "epoch", "pid"}
    if not must_cols.issubset(df_idx.columns):
        raise ValueError("patch_index.csv 需包含列：subject,epoch,pid")

    if len(labels) != len(df_idx):
        raise ValueError(f"labels({len(labels)}) 与 index({len(df_idx)}) 行数不一致")

    df_idx["label"] = labels.astype(int)

    # —— 先抽样 ——（此时不去构造所有行的 arrow_path）
    if args.sample_mode == "label":
        df_pick = stratified_by_label(df_idx, per_cluster=args.per_cluster, seed=args.seed)
    else:  # "label_pid"
        df_pick = stratified_by_label_pid(df_idx, per_bucket=args.per_bucket, seed=args.seed)

    # 仅对抽样后的 df_pick 构造 arrow_path（可选；若不构造，聚合时会懒计算）
    # 这里采用“按 subject 分组一次性推断宽度，再拼路径”，减少文件系统访问
    width_cache = {}
    arrow_paths = []
    for _, r in df_pick.iterrows():
        sid = str(r["subject"]); eid = int(r["epoch"])
        if sid not in width_cache:
            subj_dir = os.path.join(args.root, sid)
            width_cache[sid] = infer_zero_width(subj_dir, default=5)
        apath = os.path.join(args.root, sid, f"{eid:0{width_cache[sid]}d}.arrow")
        arrow_paths.append(apath)
    df_pick = df_pick.copy()
    df_pick["arrow_path"] = arrow_paths

    # —— 计算 PSD（仅基于抽样子集） ——
    modal_out = aggregate_modal_psd(
        root=args.root,
        df_pick=df_pick,
        fs=args.fs,
        eeg_idx=args.eeg_idx,
        eog_idx=args.eog_idx,
        emg_idx=args.emg_idx,
        nperseg=args.nperseg,
        patch_sec=2.0,
    )

    # —— 绘图与落盘 ——
    summary = {
        "sample_mode": args.sample_mode,
        "per_cluster": args.per_cluster,
        "per_bucket": args.per_bucket,
        "seed": args.seed,
        "fs": args.fs,
        "nperseg": args.nperseg,
    }

    for modal in ["EEG", "EOG", "EMG"]:
        if modal not in modal_out:
            continue
        f, psd0, psd1 = modal_out[modal]
        title = f"{modal} PSD comparison (n0={psd0.shape[0]}, n1={psd1.shape[0]})"
        out_png = os.path.join(out_dir, f"{modal}_psd_pvalue.png")
        pvals, m0, m1, s0, s1 = plot_modal_psd_with_pvalue(f, psd0, psd1, title, out_png)

        # 数据落盘
        np.save(os.path.join(out_dir, f"{modal}_freqs.npy"), f)
        np.save(os.path.join(out_dir, f"{modal}_pvals.npy"), pvals)
        np.save(os.path.join(out_dir, f"{modal}_psd_cluster0.npy"), psd0)
        np.save(os.path.join(out_dir, f"{modal}_psd_cluster1.npy"), psd1)

        df_csv = pd.DataFrame({
            "freq_hz": f,
            "mean_psd_c0_db": m0,
            "std_psd_c0_db": s0,
            "mean_psd_c1_db": m1,
            "std_psd_c1_db": s1,
            "p_value": pvals,
            "-log10_p": -np.log10(np.maximum(pvals, 1e-300)),
        })
        df_csv.to_csv(os.path.join(out_dir, f"{modal}_psd_stats.csv"), index=False)

        summary[modal] = {
            "n_cluster0": int(psd0.shape[0]),
            "n_cluster1": int(psd1.shape[0]),
            "n_freqs": int(len(f)),
            "png": out_png,
        }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("\n[Done] Outputs saved under:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()