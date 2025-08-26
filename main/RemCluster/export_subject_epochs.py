import os
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pyarrow import ipc
import matplotlib.pyplot as plt

def infer_zero_width(subject_dir: str, default: int = 5) -> int:
    files = [f for f in os.listdir(subject_dir) if f.endswith(".arrow")]
    widths = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            widths.append(len(stem))
    return max(widths) if widths else default

def read_arrow_matrix(path: str) -> np.ndarray:
    """读取 .arrow -> np.ndarray (C, 3000)"""
    try:
        tbl = feather.read_table(path)
    except Exception:
        with pa.memory_map(path, 'r') as source:
            reader = ipc.open_file(source)
            tbl = reader.read_all()
    df = tbl.to_pandas()

    # 单列嵌套 or 多列直接拼
    if df.shape[1] == 1:
        col = df.iloc[:, 0].to_numpy()
        if col.size == 1 and hasattr(col[0], "__len__"):
            return np.asarray(col[0], dtype=np.float32)
        arrs = [np.asarray(x) for x in col]
        return np.stack(arrs, axis=0).astype(np.float32)
    else:
        arrs = []
        for c in df.columns:
            v = df[c].to_numpy()
            if v.size == 1 and hasattr(v[0], "__len__"):
                arrs.append(np.asarray(v[0]))
            else:
                arrs.append(v.astype(np.float32))
        return np.stack(arrs, axis=0).astype(np.float32)

def plot_epoch_signal(mat: np.ndarray, fs: float, title: str, out_path: str):
    """单张图绘制多通道堆叠波形"""
    C, T = mat.shape
    t = np.arange(T) / fs
    offset = 0.0
    plt.figure(figsize=(12, 6))
    for ch in range(C):
        s = mat[ch]
        plt.plot(t, s + offset, linewidth=0.8)
        plt.text(t[0], offset, f"ch{ch}", va="bottom", fontsize=7)
        amp = (np.nanmax(s) - np.nanmin(s))
        offset += (amp * 1.2 + 1.0)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (shifted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录，例如 /data/shhs_new")
    ap.add_argument("--result_dir", required=True, help="聚类输出目录，含 labels.npy 和 index.csv")
    ap.add_argument("--subject", required=True, help="指定 subject，例如 shhs1-200639")
    ap.add_argument("--out_dir", required=True, help="输出图像保存目录")
    ap.add_argument("--fs", type=float, default=100.0, help="采样率 (Hz)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 labels + index
    index_csv = os.path.join(args.result_dir, "index.csv")
    labels_npy = os.path.join(args.result_dir, "labels.npy")
    df = pd.read_csv(index_csv)
    labels = np.load(labels_npy)
    df["label"] = labels
    df = df[df["subject"].astype(str) == str(args.subject)].sort_values("epoch").reset_index(drop=True)
    if df.empty:
        print(f"[warn] 找不到 subject={args.subject}"); return

    subj_dir = os.path.join(args.root, args.subject)
    width = infer_zero_width(subj_dir, default=5)

    for _, row in df.iterrows():
        eid = int(row["epoch"])
        lab = int(row["label"])
        arrow_path = os.path.join(subj_dir, f"{eid:0{width}d}.arrow")
        if not os.path.exists(arrow_path):
            print(f"[miss] {arrow_path}")
            continue
        try:
            mat = read_arrow_matrix(arrow_path)  # (C,3000)
        except Exception as e:
            print(f"[read err] {arrow_path}: {e}")
            continue
        title = f"{args.subject} · epoch {eid} · label {lab}"
        out_path = os.path.join(args.out_dir, f"{args.subject}_e{eid:0{width}d}_L{lab}.png")
        plot_epoch_signal(mat, args.fs, title, out_path)
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()