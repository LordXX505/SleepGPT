#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validator.py
弱监督伪标签验证器（patch 粒度）
- 加载 patch_index.csv + patch_labels.npy
- 仅验证已标注的 patch（label ∈ {0=tonic, 1=phasic}）
- 大规模数据的抽样、PSD 分析、规则一致性、可视化与汇总

目录约定：
result_dir/
  ├ patch_index.csv        (subject, epoch, pid)
  ├ patch_labels.npy       (int8: 1=phasic, 0=tonic, -1=unknown)
  └ validate_pl/           (本脚本输出)

作者：你
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt


# ---------------- I/O helpers ----------------
def infer_zero_width(subj_dir: str, default: int = 5) -> int:
    """根据 <subj_dir> 中现有 *.arrow 推断零填充宽度（如 00001.arrow → 5）"""
    try:
        cands = list(Path(subj_dir).glob("*.arrow"))
        if not cands:
            return default
        return len(cands[0].stem)
    except Exception:
        return default


def lazy_arrow_path(root: str, subject: str, epoch: int, cache: Dict[str, int]) -> Optional[str]:
    """按需推断并缓存每个 subject 的零宽度，然后拼出 .arrow 路径。不存在返回 None。"""
    subj_dir = os.path.join(root, subject)
    if subject not in cache:
        cache[subject] = infer_zero_width(subj_dir, 5)
    width = cache[subject]
    p = os.path.join(subj_dir, f"{epoch:0{width}d}.arrow")
    return p if os.path.exists(p) else None


def read_arrow_matrix(path: str) -> np.ndarray:
    """读取单个 epoch 的原始矩阵，返回 (C, T) float32"""
    try:
        reader = pa.ipc.RecordBatchFileReader(pa.memory_map(path, "r"))
        tbl = reader.read_all()
    except Exception as e:
        raise RuntimeError(f"Error reading PyArrow file {path}: {e}")

    try:
        col = tbl.column("x")
    except KeyError:
        col = tbl.columns[0]

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


def slice_patch(signal_1d: np.ndarray, pid: int, fs: int, patch_sec: float = 2.0) -> np.ndarray:
    """从 30s epoch 中切出第 pid 个 2s patch（0-based）"""
    seg_len = int(round(patch_sec * fs))
    s = pid * seg_len
    e = min(s + seg_len, signal_1d.shape[-1])
    return signal_1d[s:e]


# ---------------- 抽样器（支持超大样本） ----------------
class LabelWiseReservoir:
    """分簇（水库）抽样：保持每个簇最多 K 条"""
    def __init__(self, k_per_label: int, seed: int = 42):
        self.k = int(k_per_label)
        self.seed = np.random.RandomState(seed)
        self.count = {0: 0, 1: 0}
        self.samples = {0: [], 1: []}  # 存放 (subject, epoch, pid)

    def feed(self, label: int, item: Tuple[str, int, int]):
        if label not in (0, 1):
            return
        self.count[label] += 1
        s = self.samples[label]
        if len(s) < self.k:
            s.append(item)
        else:
            # 水库替换概率 = k / seen
            j = self.seed.randint(0, self.count[label])
            if j < self.k:
                s[j] = item

    def export(self) -> Dict[int, List[Tuple[str, int, int]]]:
        return {0: list(self.samples[0]), 1: list(self.samples[1])}


# ---------------- PSD & 规则相关 ----------------
def compute_psd_batch(batch_signals: np.ndarray, fs: int, nperseg: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """对 [N, T] 信号批量计算 Welch PSD，返回 (freqs, psd_db[N,F])"""
    out = []
    freqs = None
    for sig in batch_signals:
        nps = min(nperseg, len(sig))
        if nps < 8:
            continue
        f, Pxx = welch(sig, fs=fs, nperseg=nps, detrend="constant", scaling="density")
        if freqs is None:
            freqs = f
        Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20))
        out.append(Pxx_db)
    if not out:
        return None, None
    return freqs, np.stack(out, axis=0)


def detect_eog_deflections(sig: np.ndarray, fs: int, amp_th: float = 150.0, max_width_ms: float = 400.0) -> int:
    """返回“突波事件数”：|sig| ≥ 阈值的连续段且宽度 ≤ max_width_ms"""
    x = np.asarray(sig, dtype=np.float32)
    mask = np.abs(x) >= amp_th
    if not mask.any():
        return 0
    max_w = int(round(max_width_ms * fs / 1000.0))
    cnt = 0
    i = 0
    N = len(x)
    while i < N:
        if mask[i]:
            j = i + 1
            while j < N and mask[j]:
                j += 1
            width = j - i
            if width <= max_w:
                cnt += 1
            i = j
        else:
            i += 1
    return cnt


# ---------------- 主类 ----------------
class PseudoLabelValidator:
    def __init__(self,
                 root: str,
                 result_dir: str,
                 eeg_idx: List[int],
                 eog_idx: List[int],
                 emg_idx: List[int],
                 fs: int = 100,
                 nperseg: int = 128,
                 patch_sec: float = 2.0,
                 seed: int = 42):
        self.root = root
        self.result_dir = result_dir
        self.eeg_idx = list(eeg_idx)
        self.eog_idx = list(eog_idx)
        self.emg_idx = list(emg_idx)
        self.fs = fs
        self.nperseg = nperseg
        self.patch_sec = patch_sec
        self.seed = seed

        self.out_dir = os.path.join(self.result_dir, "validate_pl")
        os.makedirs(self.out_dir, exist_ok=True)

        self.df_idx = None  # subject, epoch, pid
        self.labels = None  # int8

    # ---------- 加载与对齐 ----------
    def load_labels(self):
        idx_path = os.path.join(self.result_dir, "patch_index.csv")
        lab_path = os.path.join(self.result_dir, "patch_labels.npy")
        if not (os.path.exists(idx_path) and os.path.exists(lab_path)):
            raise FileNotFoundError("需要 patch_index.csv + patch_labels.npy")

        df_idx = pd.read_csv(idx_path)
        labels = np.load(lab_path)

        if not set(["subject", "epoch", "pid"]).issubset(df_idx.columns):
            raise ValueError("patch_index.csv 需包含列：subject, epoch, pid")
        if len(df_idx) != len(labels):
            raise ValueError(f"index({len(df_idx)}) 与 labels({len(labels)}) 数量不一致")

        # 过滤 unknown=-1
        df_idx["label"] = labels.astype(np.int8)
        df_idx = df_idx[df_idx["label"].isin([0, 1])].reset_index(drop=True)
        labels = df_idx["label"].values.astype(np.int8)

        self.df_idx = df_idx[["subject", "epoch", "pid"]].copy()
        self.labels = labels
        # 统计保存
        uniq, cnts = np.unique(self.labels, return_counts=True)
        stats = {int(u): int(c) for u, c in zip(uniq, cnts)}
        summary = {
            "n_total_labeled": int(len(self.labels)),
            "label_counts": stats
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ---------- 分簇抽样 ----------
    def sample_by_label(self, per_label_n: int) -> Dict[int, List[Tuple[str, int, int]]]:
        res = LabelWiseReservoir(k_per_label=per_label_n, seed=self.seed)
        for (sid, eid, pid), lab in zip(self.df_idx.itertuples(index=False, name=None), self.labels):
            res.feed(lab, (sid, int(eid), int(pid)))
        return res.export()

    # ---------- 原始波形可视化 ----------
    def plot_examples(self, samples: Dict[int, List[Tuple[str, int, int]]], per_label_show: int = 16):
        vis_dir = os.path.join(self.out_dir, "examples")
        os.makedirs(vis_dir, exist_ok=True)
        width_cache: Dict[str, int] = {}

        def _plot_modal(modal: str, idxs: List[int]):
            # 组装 (N, T) 后画网格子图
            mats = []
            titles = []
            cidx = getattr(self, f"{modal.lower()}_idx")
            if not cidx:
                return
            for (sid, eid, pid) in idxs:
                p = lazy_arrow_path(self.root, sid, eid, width_cache)
                if p is None:
                    continue
                try:
                    mat = read_arrow_matrix(p)
                    avg = mat[cidx, :].mean(axis=0)
                    seg = slice_patch(avg, pid, self.fs, self.patch_sec)
                    mats.append(seg)
                    titles.append(f"{sid} e{eid} p{pid}")
                except Exception:
                    continue
            if not mats:
                return
            # 画图（每行 4 张）
            N = min(per_label_show, len(mats))
            cols = 4
            rows = int(np.ceil(N / cols))
            plt.figure(figsize=(4 * cols, 2.2 * rows))
            for i in range(N):
                ax = plt.subplot(rows, cols, i + 1)
                ax.plot(mats[i], linewidth=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(titles[i], fontsize=8)
            plt.suptitle(f"{modal} examples", y=0.99)
            out = os.path.join(vis_dir, f"{modal}_examples.png")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(out, dpi=200)
            plt.close()

        # 为每个簇画 EEG/EOG/EMG
        for lab in (0, 1):
            idxs = samples[lab][:per_label_show]
            tag = f"cluster{lab}"
            for modal in ["EEG", "EOG", "EMG"]:
                cidx = getattr(self, f"{modal.lower()}_idx")
                if cidx:
                    _plot_modal(modal=f"{modal}_{tag}", idxs=idxs)

    # ---------- PSD 分析与 p 值图 ----------
    def psd_compare(self, samples: Dict[int, List[Tuple[str, int, int]]]):
        width_cache: Dict[str, int] = {}
        out = {}
        for modal, cidx in [("EEG", self.eeg_idx), ("EOG", self.eog_idx), ("EMG", self.emg_idx)]:
            if not cidx:
                continue

            data = {0: [], 1: []}
            for lab in (0, 1):
                for sid, eid, pid in tqdm(samples[lab], desc=f"Load {modal} lab={lab}", ncols=100):
                    p = lazy_arrow_path(self.root, sid, eid, width_cache)
                    if p is None:
                        continue
                    try:
                        mat = read_arrow_matrix(p)
                        avg = mat[cidx, :].mean(axis=0)
                        seg = slice_patch(avg, pid, self.fs, self.patch_sec)
                        data[lab].append(seg)
                    except Exception:
                        continue

            if len(data[0]) == 0 or len(data[1]) == 0:
                continue

            sig0 = np.stack(data[0], 0)
            sig1 = np.stack(data[1], 0)
            f, psd0 = compute_psd_batch(sig0, self.fs, nperseg=self.nperseg)
            f2, psd1 = compute_psd_batch(sig1, self.fs, nperseg=self.nperseg)
            if f is None or f2 is None:
                continue
            if not np.allclose(f, f2):
                m = min(len(f), len(f2))
                f, psd0, psd1 = f[:m], psd0[:, :m], psd1[:, :m]

            # 逐频点 t 检验
            pvals = np.array([ttest_ind(psd0[:, i], psd1[:, i], equal_var=False).pvalue for i in range(len(f))])
            sig_mask = pvals < 0.05

            # 绘图
            m0, s0 = psd0.mean(0), psd0.std(0)
            m1, s1 = psd1.mean(0), psd1.std(0)
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            # 上图：PSD 均值±std
            axes[0].plot(f, m0, label=f"Cluster 0 (n={psd0.shape[0]})")
            axes[0].fill_between(f, m0 - s0, m0 + s0, alpha=0.2)
            axes[0].plot(f, m1, label=f"Cluster 1 (n={psd1.shape[0]})")
            axes[0].fill_between(f, m1 - s1, m1 + s1, alpha=0.2)
            axes[0].set_ylabel("PSD (dB/Hz)")
            axes[0].set_title(f"{modal} PSD comparison")
            axes[0].legend()
            # 下图：-log10(p) + 显著区域
            y = -np.log10(np.maximum(pvals, 1e-300))
            axes[1].plot(f, y, label="-log10(p)")
            axes[1].fill_between(f, 0, y, where=sig_mask, color="red", alpha=0.25, label="p<0.05")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("-log10(p)")
            axes[1].legend()
            plt.tight_layout()
            out_png = os.path.join(self.out_dir, f"{modal}_psd_pvalue.png")
            plt.savefig(out_png, dpi=300)
            plt.close()

            # 保存 CSV
            df_csv = pd.DataFrame({
                "freq_hz": f,
                "mean_psd_c0_db": m0,
                "std_psd_c0_db": s0,
                "mean_psd_c1_db": m1,
                "std_psd_c1_db": s1,
                "p_value": pvals,
                "-log10_p": y
            })
            df_csv.to_csv(os.path.join(self.out_dir, f"{modal}_psd_stats.csv"), index=False)

            out[modal] = {
                "n0": int(psd0.shape[0]),
                "n1": int(psd1.shape[0]),
                "png": out_png
            }

        with open(os.path.join(self.out_dir, "psd_summary.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # ---------- 规则一致性（EOG 振幅/事件数） ----------
    def rule_consistency(self, samples: Dict[int, List[Tuple[str, int, int]]],
                         amp_th: float = 150.0, max_width_ms: float = 400.0):
        if not self.eog_idx:
            return
        width_cache: Dict[str, int] = {}
        feats = {0: {"amp": [], "ev": []}, 1: {"amp": [], "ev": []}}
        for lab in (0, 1):
            for sid, eid, pid in tqdm(samples[lab], desc=f"Rule check lab={lab}", ncols=100):
                p = lazy_arrow_path(self.root, sid, eid, width_cache)
                if p is None:
                    continue
                try:
                    mat = read_arrow_matrix(p)
                    eog = mat[self.eog_idx, :].mean(axis=0)
                    seg = slice_patch(eog, pid, self.fs, self.patch_sec)
                    amp = float(np.max(np.abs(seg)))
                    ev = int(detect_eog_deflections(seg, self.fs, amp_th=amp_th, max_width_ms=max_width_ms))
                    feats[lab]["amp"].append(amp)
                    feats[lab]["ev"].append(ev)
                except Exception:
                    continue

        # 可视化与统计
        out = {}
        for key in ["amp", "ev"]:
            vals0 = np.asarray(feats[0][key], dtype=float)
            vals1 = np.asarray(feats[1][key], dtype=float)
            if len(vals0) == 0 or len(vals1) == 0:
                continue
            # KS 检验
            ks_p = ks_2samp(vals0, vals1, alternative="two-sided", mode="auto").pvalue
            # 画直方图
            plt.figure(figsize=(7, 4))
            bins = 50
            plt.hist(vals0, bins=bins, alpha=0.5, label=f"cluster0 (n={len(vals0)})")
            plt.hist(vals1, bins=bins, alpha=0.5, label=f"cluster1 (n={len(vals1)})")
            plt.title(f"EOG {key} distribution (KS p={ks_p:.2e})")
            plt.legend()
            plt.tight_layout()
            out_png = os.path.join(self.out_dir, f"rulecheck_{key}.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            out[key] = {"ks_p": float(ks_p), "png": out_png}
        with open(os.path.join(self.out_dir, "rule_check.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # ---------- 多轮对比（可选） ----------
    @staticmethod
    def compare_rounds(result_dirs: List[str], out_png: str):
        """对比多个 result_dir 的 label 分布"""
        rows = []
        for rd in result_dirs:
            lab_path = os.path.join(rd, "patch_labels.npy")
            if not os.path.exists(lab_path):
                continue
            labs = np.load(lab_path)
            labs = labs[np.isin(labs, [0, 1])]
            n = len(labs)
            if n == 0:
                continue
            p1 = (labs == 1).mean()
            rows.append({"result_dir": rd, "n_labeled": n, "ratio_phasic": p1})
        if not rows:
            return
        df = pd.DataFrame(rows)
        plt.figure(figsize=(7, 4))
        plt.barh(df["result_dir"], df["ratio_phasic"])
        plt.xlabel("Phasic ratio among labeled patches")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    # ---------- 总流程 ----------
    def run_all(self,
                per_label_n: int = 500000,
                per_label_show: int = 16,
                amp_th: float = 150.0,
                max_width_ms: float = 400.0):
        print("[1/5] Load labels & index ...")
        self.load_labels()

        # 简单分布与 per-subject 数量
        print("[2/5] Save distributions ...")
        # 全局分布
        uniq, cnts = np.unique(self.labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(uniq, cnts)}
        with open(os.path.join(self.out_dir, "label_distribution.json"), "w") as f:
            json.dump(dist, f, indent=2)
        # per-subject
        df_tmp = self.df_idx.copy()
        df_tmp["label"] = self.labels
        per_sub = df_tmp.groupby(["subject", "label"]).size().unstack(fill_value=0)
        per_sub.to_csv(os.path.join(self.out_dir, "per_subject_counts.csv"))

        print("[3/5] Reservoir sampling by label ...")
        samples = self.sample_by_label(per_label_n=per_label_n)

        print("[4/5] PSD analysis (+ p-values) ...")
        self.psd_compare(samples)

        print("[5/5] Rule consistency (EOG amplitude / event count) ...")
        self.rule_consistency(samples, amp_th=amp_th, max_width_ms=max_width_ms)

        # 小样例波形图（可选）
        try:
            self.plot_examples(samples, per_label_show=per_label_show)
        except Exception as e:
            print(f"[warn] plot_examples failed: {e}")

        print(f"\n[Done] Reports saved at: {self.out_dir}")


# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser("PseudoLabelValidator")
    ap.add_argument("--root", required=True, help="原始 .arrow 根目录：<root>/<subject>/<epoch>.arrow")
    ap.add_argument("--result_dir", required=True, help="伪标签输出目录（含 patch_index.csv / patch_labels.npy）")
    ap.add_argument("--eeg_idx", type=int, nargs="*", default=[], help="EEG 通道索引（可多个）")
    ap.add_argument("--eog_idx", type=int, nargs="*", default=[], help="EOG 通道索引（可多个）")
    ap.add_argument("--emg_idx", type=int, nargs="*", default=[], help="EMG 通道索引（可多个）")
    ap.add_argument("--fs", type=int, default=100, help="采样率 Hz")
    ap.add_argument("--nperseg", type=int, default=128, help="Welch nperseg（≤ patch 点数）")
    ap.add_argument("--per_label_n", type=int, default=500000, help="每个簇抽样上限（reservoir）")
    ap.add_argument("--per_label_show", type=int, default=16, help="示例波形张数/簇")
    ap.add_argument("--amp_th", type=float, default=150.0, help="规则检测 EOG 振幅阈值（μV）")
    ap.add_argument("--max_width_ms", type=float, default=400.0, help="规则检测 EOG 最大宽度（ms）")
    ap.add_argument("--compare_rounds", type=str, nargs="*", default=[], help="可选：多个 result_dir 对比")
    return ap


def main():
    args = build_argparser().parse_args()
    v = PseudoLabelValidator(
        root=args.root,
        result_dir=args.result_dir,
        eeg_idx=args.eeg_idx,
        eog_idx=args.eog_idx,
        emg_idx=args.emg_idx,
        fs=args.fs,
        nperseg=args.nperseg
    )
    v.run_all(per_label_n=args.per_label_n,
              per_label_show=args.per_label_show,
              amp_th=args.amp_th,
              max_width_ms=args.max_width_ms)

    if args.compare_rounds:
        out_png = os.path.join(v.out_dir, "compare_rounds.png")
        PseudoLabelValidator.compare_rounds(args.compare_rounds, out_png)
        print(f"[Compare] Saved: {out_png}")


if __name__ == "__main__":
    main()