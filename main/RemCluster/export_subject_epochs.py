import os
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pyarrow import ipc
import matplotlib.pyplot as plt
from scipy.signal import welch

def infer_zero_width(subject_dir: str, default: int = 5) -> int:
    files = [f for f in os.listdir(subject_dir) if f.endswith(".arrow")]
    widths = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            widths.append(len(stem))
    return max(widths) if widths else default

def read_arrow_matrix(path: str) -> np.ndarray:
    try:
        tables = pa.ipc.RecordBatchFileReader(
            pa.memory_map(path, "r")
        ).read_all()
    except Exception as e:
        raise RuntimeError(f"Error reading PyArrow file {file_path}: {e}")
    data = tables['x'][0]

    if isinstance(data, pa.ChunkedArray):
        x = np.array(data.to_pylist())
    elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
        x = np.array(data.as_py())
    else:
        x = np.array(data)
    x = x.astype(np.float32)
    x = x*1e6
    return x[[0,1,3,4]]

def plot_epoch_signal(mat: np.ndarray, fs: float, title: str, out_path: str):
    """单张图绘制多通道堆叠波形"""
    C, T = mat.shape
    t = np.arange(T) / fs
    offset = 0.0
    fig, Axes = plt.subplots(nrows=4, ncols=1, sharex='all', figsize=(30, 32))
    for i, channels in enumerate(range(4)):
        axes = Axes[i]
        axes.plot(range(3000), mat[i][:3000])
        axes.grid(True)
        axes.set_xticks(np.arange(0, 3000, 200))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (shifted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_epoch_signal_time_psd(mat: np.ndarray, fs: float, title: str, out_path: str,
                               nperseg: int = 1024, noverlap: int = 512, fmax: float = None):
    """
    在一张图里同时展示时域与 PSD（4行×2列）
    左列：时域波形；右列：PSD
    """
    C, T = mat.shape
    t = np.arange(T) / fs
    fig, axes = plt.subplots(nrows=C, ncols=2, figsize=(16, 9), gridspec_kw={"wspace": 0.25})
    if C == 1:
        axes = [axes]

    for ch in range(C):
        # 左：时域
        ax_time = axes[ch][0]
        ax_time.plot(t, mat[ch], linewidth=0.8)
        ax_time.set_ylabel(f"ch{ch}")
        ax_time.grid(True, alpha=0.3)
        if ch == C - 1:
            ax_time.set_xlabel("Time (s)")

        # 右：PSD
        ax_psd = axes[ch][1]
        f, pxx = welch(mat[ch], fs=fs, nperseg=min(nperseg, T), noverlap=min(noverlap, max(0, T//2)))
        ax_psd.semilogy(f, pxx, linewidth=0.9)
        if fmax is not None:
            ax_psd.set_xlim(0, fmax)
        if ch == C - 1:
            ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.grid(True, which="both", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录，例如 /data/shhs_new")
    ap.add_argument("--result_dir", required=True, help="聚类输出目录，含 labels.npy 和 index.csv")
    ap.add_argument("--subject", required=True, help="指定 subject，例如 shhs1-200639")
    ap.add_argument("--fs", type=float, default=100.0, help="采样率 (Hz)")
    ap.add_argument("--per_label", type=int, default=5, help="每个标签抽取多少个 epoch")
    args = ap.parse_args()

    out_dir = os.path.join(args.result_dir, "epoch_show")
    os.makedirs(out_dir, exist_ok=True)

    # 读取 labels + index
    index_csv = os.path.join(args.result_dir, "index.csv")
    labels_npy = os.path.join(args.result_dir, "labels.npy")
    df = pd.read_csv(index_csv)
    labels = np.load(labels_npy)
    assert len(df) == len(labels), "index.csv 与 labels.npy 长度不一致"
    df["label"] = labels

    # 只保留该 subject
    df = df[df["subject"].astype(str) == str(args.subject)].sort_values("epoch").reset_index(drop=True)
    if df.empty:
        print(f"[warn] 找不到 subject={args.subject}")
        return

    # —— 按标签各抽 N 条 —— #
    picks = []
    for lab, sub in df.groupby("label"):
        n = min(len(sub), args.per_label)
        if n == 0:
            continue
        # 随机抽样；如果你想“最早的 N 条”，可改为 sub.head(n)
        picks.append(sub.sample(n=n, random_state=42).assign(label=lab))
    if not picks:
        print(f"[warn] 该 subject 下没有任何可用样本")
        return
    df_pick = pd.concat(picks, axis=0).sort_values(["label", "epoch"])

    # 构造 .arrow 路径模板
    subj_dir = os.path.join(args.root, args.subject)
    width = infer_zero_width(subj_dir, default=5)

    # 逐条读取并保存图
    for _, row in df_pick.iterrows():
        eid = int(row["epoch"])
        lab = int(row["label"])
        arrow_path = os.path.join(subj_dir, f"{eid:0{width}d}.arrow")
        if not os.path.exists(arrow_path):
            print(f"[miss] {arrow_path}")
            continue

        try:
            mat = read_arrow_matrix(arrow_path)  # (C, 3000)
        except Exception as e:
            print(f"[read err] {arrow_path}: {e}")
            continue

        title = f"{args.subject} · epoch {eid} · label {lab}"
        out_path = os.path.join(out_dir, f"{args.subject}_e{eid:0{width}d}_L{lab}_time_psd.png")
        plot_epoch_signal_time_psd(mat, args.fs, title, out_path)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()