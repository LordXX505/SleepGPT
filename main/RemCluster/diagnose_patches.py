# diagnose_patches.py
import os, json, argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import matplotlib.pyplot as plt
from scipy.signal import welch


def infer_zero_width(subj_dir: str, default: int = 5) -> int:
    """推断 epoch 文件名的零填充宽度（如 00001.arrow）"""
    try:
        names = [n for n in os.listdir(subj_dir) if n.endswith(".arrow")]
        if not names: return default
        base = os.path.splitext(sorted(names)[0])[0]
        return len(base)
    except Exception:
        return default


def read_arrow_epoch(path: str) -> np.ndarray:
    """
    读取单个 .arrow 文件；返回 (C, 3000) float32
    需要列 'x' 为每通道的 3000 点。
    """
    with pa.memory_map(path, "r") as source:
        rbfr = pa_ipc.RecordBatchFileReader(source)
        table = rbfr.read_all()
    col = table["x"]
    arr = np.array(col.to_pylist(), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.shape[1] == 3000, f"expect T=3000, got {arr.shape}"
    return arr  # (C,3000)


def plot_epoch_with_highlight(mat: np.ndarray, fs: float, patch_id: int, title: str, out_path: str,
                              chans: list = None):
    """画整段30s并高亮2s的patch"""
    C, T = mat.shape
    if chans is None:
        chans = list(range(C))
    t = np.arange(T) / fs

    plt.figure(figsize=(12, 6))
    offset = 0.0
    for ch in chans:
        s = mat[ch]
        s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-8)
        plt.plot(t, s + offset, linewidth=0.8)
        plt.text(t[0], offset, f"ch{ch}", va="bottom", fontsize=8)
        amp = (np.nanmax(s) - np.nanmin(s))
        offset += (amp * 1.2 + 1.5)

    start = patch_id * 2.0
    end   = (patch_id + 1) * 2.0
    plt.axvspan(start, end, color="orange", alpha=0.2, label=f"patch {patch_id}")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (std-norm, shifted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def psd_one_patch(mat: np.ndarray, fs: float, patch_id: int,
                  nperseg: int = 128, noverlap: int = 64, use_log10: bool = True) -> dict:
    """Welch PSD for a 2s patch."""
    C, T = mat.shape
    st = patch_id * int(2 * fs)
    ed = st + int(2 * fs)
    seg = mat[:, st:ed]  # (C, 200)

    out = {}
    for ch in range(C):
        x = seg[ch].astype(np.float32)
        f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)), noverlap=min(noverlap, max(0, len(x)//2 - 1)))
        if use_log10:
            Pxx = 10.0 * np.log10(Pxx + 1e-12)
        out[ch] = (f, Pxx)
    return out


def summarize_psd(psds: list, target_chs: list) -> dict:
    """对一组 PSD 结果做均值/方差统计。psds: [ {ch:(f,Pxx)}, ... ]"""
    stat = {}
    for ch in target_chs:
        stacks = []
        f_ref = None
        for d in psds:
            f, p = d[ch]
            if f_ref is None:
                f_ref = f
            stacks.append(p)
        if not stacks:
            continue
        P = np.stack(stacks, 0)
        stat[ch] = {"f": f_ref, "mean": P.mean(axis=0), "std": P.std(axis=0)}
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录，如 /data/shhs_new/shhs_new")
    ap.add_argument("--result_dir", required=True, help="包含 patch_index.csv 与 labels_patch.npy 的输出目录")
    ap.add_argument("--subject", default="", help="仅分析某个 subject（可留空为全部）")
    ap.add_argument("--fs", type=float, default=100.0, help="采样率 (Hz)")
    ap.add_argument("--n_per_label", type=int, default=10, help="每个 label 抽样多少个 patch")
    ap.add_argument("--plot_chs", type=str, default="", help="要画的通道，如 '0,1,2,3'（留空=全部）")
    ap.add_argument("--eog_chs", type=str, default="", help="EOG 通道索引（用于 PSD 统计），如 '0,1'")
    ap.add_argument("--emg_chs", type=str, default="", help="EMG 通道索引（用于 PSD 统计），如 '3'")
    ap.add_argument("--out_dir", type=str, default="diag_patch_viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 载入 patch-level 结果
    idx_csv = os.path.join(args.result_dir, "patch_index.csv")
    lab_npy = os.path.join(args.result_dir, "labels_patch.npy")
    if not (os.path.exists(idx_csv) and os.path.exists(lab_npy)):
        raise FileNotFoundError("need patch_index.csv and labels_patch.npy in result_dir")

    df = pd.read_csv(idx_csv)  # subject,epoch,patch
    labels_patch = np.load(lab_npy)
    if len(labels_patch) != len(df):
        raise RuntimeError("labels_patch length != patch_index rows")

    df["label"] = labels_patch
    if args.subject:
        df = df[df["subject"].astype(str) == str(args.subject)]
        if df.empty:
            print(f"[warn] subject={args.subject} not found in patch_index.csv")
            return

    # 采样各类 patch
    picks = []
    for lab in sorted(df["label"].unique()):
        d = df[df["label"] == lab]
        if len(d) == 0:
            continue
        picks.append(d.sample(n=min(args.n_per_label, len(d)), random_state=42))
    if not picks:
        print("[warn] no patches to plot."); return
    pick_df = pd.concat(picks, axis=0).reset_index(drop=True)

    # 通道选择
    plot_chs = [int(x) for x in args.plot_chs.split(",")] if args.plot_chs else None
    eog_chs  = [int(x) for x in args.eog_chs.split(",")]  if args.eog_chs  else []
    emg_chs  = [int(x) for x in args.emg_chs.split(",")]  if args.emg_chs  else []

    # 逐一画原始波形并收集 PSD
    psd_per_label = {}  # lab -> list of {ch:(f,Pxx)}
    for i, row in pick_df.iterrows():
        sid  = str(row["subject"])
        eid  = int(row["epoch"])
        pid  = int(row["patch"])
        lab  = int(row["label"])

        subj_dir = os.path.join(args.root, sid)
        if not os.path.isdir(subj_dir):
            print(f"[miss subj] {subj_dir}"); continue

        width = infer_zero_width(subj_dir, default=5)
        arrow = os.path.join(subj_dir, f"{eid:0{width}d}.arrow")
        if not os.path.exists(arrow):
            print(f"[miss arrow] {arrow}"); continue

        try:
            mat = read_arrow_epoch(arrow)  # (C,3000)
        except Exception as e:
            print(f"[read err] {arrow}: {e}")
            continue

        # 波形+高亮
        title = f"{sid} · epoch {eid} · patch {pid} · label {lab}"
        out_png = os.path.join(args.out_dir, f"{sid}_e{eid:0{width}d}_p{pid}_L{lab}.png")
        plot_epoch_with_highlight(mat, args.fs, pid, title, out_png, chans=plot_chs)
        print(f"[plot] {out_png}")

        # PSD
        dpsd = psd_one_patch(mat, fs=args.fs, patch_id=pid, nperseg=128, noverlap=64, use_log10=True)
        psd_per_label.setdefault(lab, []).append(dpsd)

    # 汇总 PSD（EOG/EMG 可选）
    summary = {"fs": args.fs, "n_per_label": int(args.n_per_label),
               "eog_chs": eog_chs, "emg_chs": emg_chs}

    def summarize_and_plot(stat_dict, name: str):
        if not stat_dict: return
        # 画均值±std
        for lab, dch in stat_dict.items():
            for ch, dd in dch.items():
                f = dd["f"]; mu = dd["mean"]; sd = dd["std"]
                plt.plot(f, mu, label=f"lab{lab}-ch{ch}")
                plt.fill_between(f, mu-sd, mu+sd, alpha=0.2)
        plt.xlabel("Hz"); plt.ylabel("Power (dB)")
        plt.title(f"{name} PSD (Welch, log10)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"psd_{name}.png")
        plt.savefig(out, dpi=200); plt.close()
        print(f"[plot] {out}")

    # 统计
    if eog_chs:
        summary["EOG"] = summarize_psd(psd_per_label.get(0, []) + psd_per_label.get(1, []), eog_chs)
        # 拆 label 存
        summary["EOG_by_label"] = {}
        for lab in psd_per_label:
            summary["EOG_by_label"][lab] = summarize_psd(psd_per_label[lab], eog_chs)
        summarize_and_plot(summary["EOG_by_label"], "EOG")

    if emg_chs:
        summary["EMG"] = summarize_psd(psd_per_label.get(0, []) + psd_per_label.get(1, []), emg_chs)
        summary["EMG_by_label"] = {}
        for lab in psd_per_label:
            summary["EMG_by_label"][lab] = summarize_psd(psd_per_label[lab], emg_chs)
        summarize_and_plot(summary["EMG_by_label"], "EMG")

    with open(os.path.join(args.out_dir, "patch_psd_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[dump] {os.path.join(args.out_dir, 'patch_psd_summary.json')}")


if __name__ == "__main__":
    main()