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

    # 假设表里有列名 'x'，其中每条为一个通道的向量；具体按你存储结构微调
    # 之前你提到过：tables['x'][0] 就是 data，这里做更稳健一些
    try:
        col = tbl.column("x")
    except KeyError:
        # 如果列名不是 'x'，尝试第 0 列
        col = tbl.columns[0]

    # 将箭头数组转换为 numpy (C, T)
    # 适配几种常见存法：List<fixed-size> / ChunkedArray / LargeList 等
    if isinstance(col, pa.ChunkedArray):
        arrs = []
        for chunk in col.chunks:
            arrs.extend(chunk.to_pylist())
        mat = np.asarray(arrs, dtype=np.float32)
    else:
        mat = np.asarray(col.to_pylist(), dtype=np.float32)

    # 若是一维，补一维
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    return mat  # (C, T)

# ----------------------------
# 数据抽样与 PSD 计算
# ----------------------------
def slice_patch(signal_1d: np.ndarray, pid: int, fs: int, patch_sec: float = 2.0) -> np.ndarray:
    """
    从 30s epoch 中切出第 pid 个 patch（0-based），每个 patch = 2s。
    假设 epoch 总长度 T=3000（fs=100）→ 15 个 patch × 200 点。
    """
    seg_len = int(round(patch_sec * fs))  # 2s*100Hz=200
    start = pid * seg_len
    end = start + seg_len
    if end > signal_1d.shape[-1]:
        # 越界则截断
        end = signal_1d.shape[-1]
    return signal_1d[start:end]

def compute_psd_batch(batch_signals: np.ndarray, fs: int, nperseg: int = 128) -> (np.ndarray, np.ndarray):
    """
    对一批 1D 信号（形状 [N, T]）计算 Welch PSD，返回 (freqs, psd_matrix[dB])
    """
    psd_list = []
    for sig in batch_signals:
        # nperseg 不宜超过长度
        nps = min(nperseg, len(sig))
        if nps < 8:  # 太短跳过
            continue
        f, Pxx = welch(sig, fs=fs, nperseg=nps, detrend="constant", scaling="density")
        # 转 dB：为避免 log(0) → 加微小量
        Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20))
        psd_list.append(Pxx_db)
    if not psd_list:
        return None, None
    psd = np.stack(psd_list, axis=0)  # [N, F]
    return f, psd

def aggregate_modal_psd(root: str,
                        df_pick: pd.DataFrame,
                        fs: int,
                        eeg_idx: list,
                        eog_idx: list,
                        emg_idx: list,
                        nperseg: int = 128,
                        patch_sec: float = 2.0) -> dict:
    """
    从采样到的 patch 列表（df_pick: subject,epoch,pid,label,arrow_path）
    读取原始 epoch，切出对应 patch，按模态求 PSD。
    - EEG/EOG 用“多个通道平均”方式：先在通道维做平均 → 得到一个 1D patch 信号
    - EMG 亦可多通道平均（如果你只有一个 EMG 通道，就传一个索引）
    返回：{"EEG": (f, psd0, psd1), "EOG":(...), "EMG":(...)}；若某模态通道缺失则不返回该键
    """
    # 收集三个模态的两组补丁
    modal_signals = {
        "EEG": {0: [], 1: []},
        "EOG": {0: [], 1: []},
        "EMG": {0: [], 1: []}
    }
    # 逐条 patch 读取
    for _, row in tqdm(df_pick.iterrows(), total=len(df_pick), desc="Load patches"):
        sid = str(row["subject"])
        eid = int(row["epoch"])
        pid = int(row["patch"])
        lab = int(row["label"])
        apath = row["arrow_path"]
        if not os.path.exists(apath):
            # 自动回退：根据 root 拼一下（防止 index.csv 里没写路径）
            subj_dir = os.path.join(root, sid)
            width = infer_zero_width(subj_dir, default=5)
            apath2 = os.path.join(subj_dir, f"{eid:0{width}d}.arrow")
            if os.path.exists(apath2):
                apath = apath2
            else:
                print(f"[miss] {apath}")
                continue

        try:
            mat = read_arrow_matrix(apath)  # (C, T)
        except Exception as e:
            print(f"[read err] {apath}: {e}")
            continue

        # EEG
        if eeg_idx:
            try:
                eeg_sig = mat[eeg_idx, :]  # 可能多通道
                eeg_avg = eeg_sig.mean(axis=0)
                patch_eeg = slice_patch(eeg_avg, pid, fs, patch_sec=patch_sec)
                modal_signals["EEG"][lab].append(patch_eeg)
            except Exception:
                pass

        # EOG
        if eog_idx:
            try:
                eog_sig = mat[eog_idx, :]
                eog_avg = eog_sig.mean(axis=0)
                patch_eog = slice_patch(eog_avg, pid, fs, patch_sec=patch_sec)
                modal_signals["EOG"][lab].append(patch_eog)
            except Exception:
                pass

        # EMG
        if emg_idx:
            try:
                emg_sig = mat[emg_idx, :]
                emg_avg = emg_sig.mean(axis=0)
                patch_emg = slice_patch(emg_avg, pid, fs, patch_sec=patch_sec)
                modal_signals["EMG"][lab].append(patch_emg)
            except Exception:
                pass

    # 计算每个模态的 PSD
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
        if f is None or psd0 is None or f2 is None or psd1 is None:
            continue
        # 对齐（理论上 f 与 f2 一样）
        if not np.allclose(f, f2):
            m = min(len(f), len(f2))
            f, psd0, psd1 = f[:m], psd0[:, :m], psd1[:, :m]
        out[key] = (f, psd0, psd1)
    return out

# ----------------------------
# 统计与绘图
# ----------------------------
def plot_modal_psd_with_pvalue(f, psd0, psd1, title, out_png):
    """
    画 2×1：上面是两组 PSD 的均值±标准差；下面是 -log10(p)，并用阴影标出 p<0.05 的区域。
    """
    # 均值/标准差
    m0, s0 = psd0.mean(axis=0), psd0.std(axis=0)
    m1, s1 = psd1.mean(axis=0), psd1.std(axis=0)

    # 逐频点 t 检验
    pvals = np.array([ttest_ind(psd0[:, i], psd1[:, i], equal_var=False).pvalue for i in range(len(f))])
    sig = pvals < 0.05

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图
    axes[0].plot(f, m0, label=f"Cluster 0 (n={psd0.shape[0]})")
    axes[0].fill_between(f, m0 - s0, m0 + s0, alpha=0.2)
    axes[0].plot(f, m1, label=f"Cluster 1 (n={psd1.shape[0]})")
    axes[0].fill_between(f, m1 - s1, m1 + s1, alpha=0.2)
    axes[0].set_ylabel("PSD (dB/Hz)")
    axes[0].set_title(title)
    axes[0].legend()

    # 下图
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
    ap = argparse.ArgumentParser("Patch-level PSD diagnosis with per-frequency p-values")
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录（<root>/<subject>/<epoch>.arrow）")
    ap.add_argument("--result_dir", required=True, help="包含 patch_labels.npy 与 patch_index.csv 的目录")
    ap.add_argument("--fs", type=int, default=100, help="采样率 Hz（默认 100）")
    ap.add_argument("--per_cluster", type=int, default=5000, help="每个簇抽样 patch 数量上限")
    ap.add_argument("--nperseg", type=int, default=128, help="Welch nperseg（2s patch 建议 ≤ 200）")

    # 通道索引（基于 mat 的第 0 维）；可传多个
    ap.add_argument("--eeg_idx", type=int, nargs="*", default=[], help="EEG 通道索引列表，例如 --eeg_idx 0 1")
    ap.add_argument("--eog_idx", type=int, nargs="*", default=[], help="EOG 通道索引列表，例如 --eog_idx 2")
    ap.add_argument("--emg_idx", type=int, nargs="*", default=[], help="EMG 通道索引列表，例如 --emg_idx 3")

    args = ap.parse_args()
    os.makedirs(os.path.join(args.result_dir, "diag_patch_psd"), exist_ok=True)
    out_dir = os.path.join(args.result_dir, "diag_patch_psd")

    # 读取 patch-level 标签与索引
    labels_path = os.path.join(args.result_dir, "patch_labels.npy")
    index_path  = os.path.join(args.result_dir, "patch_index.csv")
    if not (os.path.exists(labels_path) and os.path.exists(index_path)):
        raise FileNotFoundError("需要 patch_labels.npy 与 patch_index.csv")

    labels = np.load(labels_path)
    df_idx = pd.read_csv(index_path)  # 期望列：subject,epoch,pid
    if not set(["subject", "epoch", "patch"]).issubset(df_idx.columns):
        raise ValueError("patch_index.csv 需包含列：subject,epoch,pid")

    if len(labels) != len(df_idx):
        raise ValueError(f"labels({len(labels)}) 与 index({len(df_idx)}) 行数不一致")

    df_idx["label"] = labels.astype(int)
    print('统一生成并缓存，避免反复推断')
    # 构造 .arrow 路径（若 index 已含有可用路径，也可跳过）
    # 统一生成并缓存，避免反复推断
    arrow_paths = []
    last_sid, last_width = None, 5
    df_idx["arrow_path"] = { os.path.join(args.root, str(sid), f"{int(eid):05d}.arrow")
                             for sid,eid in zip(df_idx["subject"], df_idx["epoch"])}
    print('分簇抽样')

    # 分簇抽样
    g0 = df_idx[df_idx["label"] == 0].sample(n=min(args.per_cluster, (df_idx["label"]==0).sum()), random_state=42)
    g1 = df_idx[df_idx["label"] == 1].sample(n=min(args.per_cluster, (df_idx["label"]==1).sum()), random_state=42)
    df_pick = pd.concat([g0, g1], ignore_index=True)
    print('聚合并计算 PSD（按模态）')

    # 聚合并计算 PSD（按模态）
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

    # 逐模态绘图与导出
    summary = {}
    for modal in ["EEG", "EOG", "EMG"]:
        if modal not in modal_out:
            continue
        f, psd0, psd1 = modal_out[modal]
        title = f"{modal} PSD comparison (n0={psd0.shape[0]}, n1={psd1.shape[0]})"
        out_png = os.path.join(out_dir, f"{modal}_psd_pvalue.png")
        pvals, m0, m1, s0, s1 = plot_modal_psd_with_pvalue(f, psd0, psd1, title, out_png)

        # 保存数值
        np.save(os.path.join(out_dir, f"{modal}_freqs.npy"), f)
        np.save(os.path.join(out_dir, f"{modal}_pvals.npy"), pvals)
        np.save(os.path.join(out_dir, f"{modal}_psd_cluster0.npy"), psd0)
        np.save(os.path.join(out_dir, f"{modal}_psd_cluster1.npy"), psd1)

        # CSV 摘要（均值与 p 值）
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
            "png": out_png
        }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("\n[Done] Outputs saved under:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()