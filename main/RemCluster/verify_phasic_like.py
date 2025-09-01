import os, argparse, json, warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------- I/O -------------
def read_arrow_matrix(path: str) -> np.ndarray:
    """读取 .arrow -> ndarray (C, T)"""
    with pa.memory_map(path, "r") as source:
        reader = pa_ipc.RecordBatchFileReader(source)
        table = reader.read_all()

    # 兼容你之前的存法：data 在 'x' 列
    if "x" in table.column_names:
        col = table["x"][0]
        if isinstance(col, pa.ChunkedArray):
            arr = np.array(col.to_pylist())
        elif isinstance(col, (pa.Array, pa.Scalar)):
            arr = np.array(col.as_py())
        else:
            arr = np.array(col)
    else:
        # 如果没有 'x' 列，试着把表转成 numpy
        # 你可以改成适配你自己的 schema
        cols = [np.array(table[c].to_pylist()) for c in table.column_names]
        arr = np.stack(cols, axis=0)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # 期望 (C,T)
    if arr.shape[0] > arr.shape[1]:
        # 有些保存会是 (T,C)；若明显更像 (T,C)，则转置
        if arr.shape[0] > 10 and arr.shape[1] < 10:
            arr = arr.T
    return arr


def infer_zero_width(subj_dir: str, default=5) -> int:
    """推断 00001.arrow 的零填充宽度。"""
    try:
        names = [n for n in os.listdir(subj_dir) if n.endswith(".arrow")]
        if not names:
            return default
        stem = os.path.splitext(names[0])[0]
        return len(stem)
    except Exception:
        return default


# ------------- PSD & 统计 -------------
def compute_psd(sig: np.ndarray, fs: float, nperseg: int = 256, scaling: str = "density"):
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, scaling=scaling)
    return f, Pxx


def group_psd_by_modality(mat: np.ndarray,
                          fs: float,
                          patch_sec: float,
                          epoch_sec: float,
                          eid: int,
                          maybe_pid: int,
                          eeg_idx: np.ndarray,
                          eog_idx: np.ndarray,
                          emg_idx: np.ndarray,
                          db_scale: bool):
    """
    返回 dict: {'EEG': Pxx, 'EOG': Pxx, 'EMG': Pxx} 每类是 1×F（通道内均值）
    - mat: (C,T)
    - 如果 maybe_pid is not None: 按 pid 截取 patch；否则整个 epoch。
    """
    T = mat.shape[1]
    fs = float(fs)

    if maybe_pid is not None:
        # patch 切片
        start = int(maybe_pid * patch_sec * fs)
        end   = start + int(patch_sec * fs)
        if end > T:
            return None, None  # 越界，略过
        seg = mat[:, start:end]
    else:
        # 整个 epoch
        need = int(epoch_sec * fs)
        if T < need:
            return None, None
        seg = mat[:, :need]

    # 各类通道：对通道求 PSD，再在通道维取均值（避免通道数不等）
    out = {}
    freqs = None
    for name, idx in [("EEG", eeg_idx), ("EOG", eog_idx), ("EMG", emg_idx)]:
        if idx.size == 0:
            continue
        idx = idx[(idx >= 0) & (idx < seg.shape[0])]
        if idx.size == 0:
            continue
        P_list = []
        for ch in idx:
            f, Pxx = compute_psd(seg[ch], fs=fs, nperseg=min(256, seg.shape[1]))
            P_list.append(Pxx)
            if freqs is None: freqs = f
        P_arr = np.stack(P_list, 0).mean(0)  # 通道内均值
        if db_scale:
            # 避免 log(0)
            P_arr = 10.0 * np.log10(P_arr + 1e-12)
        out[name] = P_arr
    return freqs, out


def balanced_indices_by_label(labels: np.ndarray, max_per_cluster: int):
    """返回两个 cluster 的均衡抽样索引（尽量相等数量）。"""
    uniq = np.unique(labels)
    if len(uniq) != 2:
        # 只处理两类；否则返回全部
        return np.arange(len(labels))
    idx0 = np.where(labels == uniq[0])[0]
    idx1 = np.where(labels == uniq[1])[0]
    n = min(len(idx0), len(idx1), max_per_cluster)
    rng = np.random.default_rng(42)
    return np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])


# ------------- 主流程 -------------
def main():
    ap = argparse.ArgumentParser("Cluster PSD compare on EEG/EOG/EMG with per-frequency tests")
    ap.add_argument("--result_dir", required=True, help="包含 labels.npy / index.csv")
    ap.add_argument("--data_root",  required=True, help="原始 .arrow 根目录")
    ap.add_argument("--out_dir",    required=True, help="输出目录")
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--epoch_sec", type=float, default=30.0)
    ap.add_argument("--patch_sec", type=float, default=2.0)

    ap.add_argument("--eeg_idx", type=str, default="")
    ap.add_argument("--eog_idx", type=str, default="")
    ap.add_argument("--emg_idx", type=str, default="")

    ap.add_argument("--db_scale", action="store_true", help="以 dB 显示/检验 (10*log10)")
    ap.add_argument("--max_per_cluster", type=int, default=5000, help="每个簇的最大样本数（平衡抽样）")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 labels / index
    labels = np.load(os.path.join(args.result_dir, "labels.npy"))
    df = pd.read_csv(os.path.join(args.result_dir, "index.csv"))

    # 支持 patch 粒度或 epoch 粒度
    has_patch = "patch" in df.columns
    if has_patch:
        df = df[["subject", "epoch", "patch"]].copy()
    else:
        df = df[["subject", "epoch"]].copy()
        df["patch"] = np.nan  # 占位

    # 对齐 labels
    if len(labels) != len(df):
        raise ValueError(f"labels({len(labels)}) 与 index.csv({len(df)}) 长度不一致")
    df["label"] = labels

    # 解析通道索引
    def parse_idx(s):
        s = str(s).strip()
        if not s:
            return np.array([], dtype=int)
        return np.array([int(x) for x in s.split(",") if x != ""], dtype=int)

    eeg_idx = parse_idx(args.eeg_idx)
    eog_idx = parse_idx(args.eog_idx)
    emg_idx = parse_idx(args.emg_idx)
    if eog_idx.size == 0:
        print("[warn] 未提供 EOG 索引；强烈建议设置 --eog_idx 先看 EOG 差异。")

    # 平衡抽样（防止极端不平衡）
    keep = balanced_indices_by_label(df["label"].values, args.max_per_cluster)
    df_s = df.iloc[keep].reset_index(drop=True)

    # 收集各模态的 PSD 序列
    store = {"EEG": [], "EOG": [], "EMG": []}
    freqs = None

    # 遍历样本
    # 这里按 subject/epoch 定位 .arrow 文件：/root/subject/00001.arrow
    # 自动推断填零宽度
    prev_subj, zero_w = None, 5

    for i, row in tqdm(df_s.iterrows(), total=len(df_s), desc="Compute PSD"):
        sid = str(row["subject"])
        eid = int(row["epoch"])
        pid = None if np.isnan(row["patch"]) else int(row["patch"])

        subj_dir = os.path.join(args.data_root, sid)
        if prev_subj != sid:
            zero_w = infer_zero_width(subj_dir, default=5)
            prev_subj = sid

        arrow_path = os.path.join(subj_dir, f"{eid:0{zero_w}d}.arrow")
        if not os.path.exists(arrow_path):
            # 缺文件就跳过
            continue

        try:
            mat = read_arrow_matrix(arrow_path)  # (C,T)
        except Exception as e:
            print(f"[read err] {arrow_path}: {e}")
            continue

        f, psd_dict = compute_one(mat, args.fs, args.patch_sec, args.epoch_sec, eid, pid,
                                  eeg_idx, eog_idx, emg_idx, args.db_scale)
        if f is None:
            continue
        if freqs is None:
            freqs = f

        # 保存各模态的一条曲线（通道均值后）
        for k in store.keys():
            if k in psd_dict:
                store[k].append(psd_dict[k])

    if freqs is None:
        raise RuntimeError("未成功读取到任何 PSD。请检查数据路径/索引。")

    # 转 numpy
    for k in store.keys():
        if len(store[k]) > 0:
            store[k] = np.stack(store[k], 0)  # [N, F]
        else:
            store[k] = None

    # 按 labels 同步切分
    y = df_s["label"].values
    # 注意：上面的平衡抽样已做过，y 和 store 的样本数对齐
    # 做统计 & 画图
    results = {}
    for name in ["EOG", "EEG", "EMG"]:
        X = store[name]
        if X is None:
            continue

        # 两类样本
        y0 = X[y == 0] if np.any(y == 0) else None
        y1 = X[y == 1] if np.any(y == 1) else None
        if (y0 is None) or (y1 is None) or (len(y0) == 0) or (len(y1) == 0):
            print(f"[warn] {name}: 某个 cluster 无样本，跳过统计")
            continue

        # 逐频率 Mann–Whitney U
        pvals = []
        for fi in range(X.shape[1]):
            p = mannwhitneyu(y0[:, fi], y1[:, fi], alternative="two-sided")[1]
            pvals.append(p)
        pvals = np.array(pvals)
        _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")

        # 均值/方差
        m0, s0 = y0.mean(0), y0.std(0)
        m1, s1 = y1.mean(0), y1.std(0)

        # 保存数值
        np.save(os.path.join(args.out_dir, f"pvals_{name}.npy"), pvals_fdr)
        np.save(os.path.join(args.out_dir, f"mean0_{name}.npy"), m0)
        np.save(os.path.join(args.out_dir, f"mean1_{name}.npy"), m1)
        np.save(os.path.join(args.out_dir, f"freqs.npy"), freqs)

        # 画图
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, m0, label="Cluster 0", linewidth=1.5)
        plt.fill_between(freqs, m0 - s0, m0 + s0, alpha=0.15)
        plt.plot(freqs, m1, label="Cluster 1", linewidth=1.5)
        plt.fill_between(freqs, m1 - s1, m1 + s1, alpha=0.15)

        sig = pvals_fdr < 0.05
        if np.any(sig):
            plt.scatter(freqs[sig], ((m0+m1)/2)[sig], s=6, c="k", alpha=0.6, label="p<0.05 (FDR)")

        ylabel = "PSD (dB/Hz)" if args.db_scale else "PSD (V^2/Hz)"
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        title = f"{name} PSD comparison (n0={len(y0)}, n1={len(y1)})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(args.out_dir, f"psd_mean_std_{name}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        # 简要频段统计（可按需要调整）
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "sigma": (12.0, 15.0),
            "beta":  (15.0, 30.0)
        }
        band_stats = {}
        for bname, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.sum() == 0:
                continue
            # 均值功率（对频带积分或平均都可，这里做平均）
            band_stats[bname] = {
                "mean_cluster0": float(m0[mask].mean()),
                "mean_cluster1": float(m1[mask].mean()),
                "p_fraction_sig": float((pvals_fdr[mask] < 0.05).mean()),
            }

        results[name] = {
            "n0": int(len(y0)),
            "n1": int(len(y1)),
            "n_freq": int(len(freqs)),
            "sig_count": int((pvals_fdr < 0.05).sum()),
            "bands": band_stats
        }

    with open(os.path.join(args.out_dir, "psd_stats_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n== 完成 ==")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def compute_one(mat, fs, patch_sec, epoch_sec, eid, pid,
                eeg_idx, eog_idx, emg_idx, db_scale):
    return group_psd_by_modality(mat, fs, patch_sec, epoch_sec, eid, pid,
                                 eeg_idx, eog_idx, emg_idx, db_scale)


if __name__ == "__main__":
    main()