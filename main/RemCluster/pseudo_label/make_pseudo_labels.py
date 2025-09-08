#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_pseudo_labels.py
基于 EOG 规则的弱监督伪标签生成（patch=2s），并且：
  - 只对 REM 期（stage ∈ --rem_values）生成
  - 输出每个 subject 的 phasic/tonic/unknown patch 计数日志

输出文件（在 --out_dir 下）：
  - patch_index.csv   (subject, epoch, pid)   与 patch_labels.npy 严格对齐
  - patch_labels.npy  (int8；1=phasic, 0=tonic, -1=unknown)
  - subject_patch_counts.csv (per subject 汇总)
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm


# ---------------- 基础 I/O ----------------
def list_subjects(root: str) -> List[str]:
    return sorted([p.name for p in Path(root).iterdir() if p.is_dir()])


def list_epochs(subj_dir: str) -> List[int]:
    out = []
    for p in Path(subj_dir).glob("*.arrow"):
        m = re.match(r"^(\d+)\.arrow$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def infer_zero_width(subj_dir: str, default: int = 5) -> int:
    cands = list(Path(subj_dir).glob("*.arrow"))
    if not cands:
        return default
    return len(cands[0].stem)


def _table_column(tbl: pa.Table, name: str) -> Optional[pa.ChunkedArray]:
    try:
        return tbl.column(name)
    except KeyError:
        return None


def read_arrow_epoch(path: str):
    """读取单个 epoch：返回 (signal, stage)，signal shape=(C, T)"""
    reader = pa.ipc.RecordBatchFileReader(pa.memory_map(path, "r"))
    tbl = reader.read_all()
    data = tbl['x'][0]
    if isinstance(data, pa.ChunkedArray):
        x = np.array(data.to_pylist())
    elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
        x = np.array(data.as_py())
    else:
        x = np.array(data)
    x = x.astype(np.float32)
    stage = np.array(tbl["stage"]).astype(np.long)
    return x, stage


# ---------------- Patch / 规则 ----------------
def slice_patch(sig: np.ndarray, pid: int, fs: int, patch_sec: float = 2.0) -> np.ndarray:
    n = int(round(patch_sec * fs))
    s = pid * n
    e = min(s + n, sig.shape[-1])
    return sig[s:e]


def detect_eog_deflections(sig: np.ndarray,
                           fs: int,
                           amp_th: float = 150.0,
                           max_width_ms: float = 400.0) -> int:
    """
    返回“突波事件数”：|sig| >= amp_th 的连续区段，且宽度 <= max_width_ms
    """
    x = np.asarray(sig, dtype=np.float32)
    mask = np.abs(x) >= amp_th
    if not mask.any():
        return 0
    max_w_samples = int(round(max_width_ms * fs / 1000.0))
    cnt = 0
    i, N = 0, len(x)
    while i < N:
        if mask[i]:
            j = i + 1
            while j < N and mask[j]:
                j += 1
            width = j - i
            if width <= max_w_samples:
                cnt += 1
            i = j
        else:
            i += 1
    return cnt


def build_pseudo_labels_for_epoch(mat: np.ndarray,
                                  eog_idx: List[int],
                                  fs: int,
                                  amp_phasic: float,
                                  amp_tonic: float,
                                  max_width_ms: float,
                                  min_consecutive: int = 2,
                                  separation_k: int = 2) -> np.ndarray:
    """
    对一个 epoch（C,T）输出 15 个 patch 的标签（1/0/-1）。
    参见说明：连续事件判 phasic；与事件隔离判 tonic；其余 unknown。
    """
    T = mat.shape[1]
    eog_avg = mat[eog_idx, :].mean(axis=0)
    patch_len = int(round(2.0 * fs))
    n_patch = min(15, max(1, T // patch_len))

    has_event = np.zeros(n_patch, dtype=bool)
    is_tonic  = np.zeros(n_patch, dtype=bool)
    for pid in range(n_patch):
        p = slice_patch(eog_avg, pid, fs, 2.0)
        ev = detect_eog_deflections(p, fs=fs, amp_th=amp_phasic, max_width_ms=max_width_ms)
        has_event[pid] = (ev >= 1)
        is_tonic[pid] = (np.max(np.abs(p)) < amp_tonic)

    labels = np.full(n_patch, -1, dtype=np.int8)
    # phasic by consecutive
    if min_consecutive <= 1:
        labels[has_event] = 1
    else:
        run = 0
        for i in range(n_patch):
            if has_event[i]:
                run += 1
            else:
                if run >= min_consecutive:
                    labels[i-run:i] = 1
                run = 0
        if run >= min_consecutive:
            labels[n_patch-run:n_patch] = 1
    # tonic by isolation
    for i in range(n_patch):
        if labels[i] != -1:
            continue
        if not is_tonic[i]:
            continue
        L = max(0, i - separation_k)
        R = min(n_patch, i + separation_k + 1)
        if not has_event[L:R].any():
            labels[i] = 0
    return labels


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser("Make pseudo labels (phasic/tonic/unknown) on 2s patches, REM-only, with per-subject logs.")
    ap.add_argument("--root", required=True, help="原始数据根目录：<root>/<subject>/<epoch>.arrow")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--fs", type=int, default=100, help="采样率 Hz（默认 100）")

    # EOG 通道
    ap.add_argument("--eog_idx", type=int, nargs="+", required=True, help="EOG 通道下标（可多个），如：--eog_idx 2 3")

    # 规则阈值
    ap.add_argument("--amp_phasic", type=float, default=150.0, help="phasic 振幅阈值（μV）")
    ap.add_argument("--amp_tonic",  type=float, default=25.0,  help="tonic 最大振幅（μV）")
    ap.add_argument("--max_width_ms", type=float, default=400.0, help="phasic 突波最大宽度（ms）")
    ap.add_argument("--min_consecutive", type=int, default=2, help="相邻 2s 连续数达到多少判 phasic（默认2）")
    ap.add_argument("--separation_k", type=int, default=2, help="tonic 需要与任何事件相隔的 patch 数（默认2，即 4s 间隔）")

    # REM 过滤
    ap.add_argument("--rem_values", nargs="+", default=["REM", "R", "5", 5],
                    help="判定为 REM 的 stage 值集合（字符串或整数都可以），默认 ['REM','R','5',5]")
    # 可选：只对清单中的 subject/epoch 生成（例如你已有 REM 清单时）
    ap.add_argument("--restrict_index", type=str, default=None,
                    help="可选 CSV（columns: subject,epoch），仅处理出现在清单内的样本")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 标准化 REM 值为字符串集合，比较时统一 str(stage)
    rem_allow: Set[str] = set(str(v) for v in args.rem_values)

    restrict = None
    if args.restrict_index and os.path.exists(args.restrict_index):
        df = pd.read_csv(args.restrict_index)
        if not set(["subject", "epoch"]).issubset(df.columns):
            raise ValueError("restrict_index 需要包含列：subject,epoch")
        restrict = set((str(s), int(e)) for s, e in zip(df["subject"], df["epoch"]))

    index_rows = []
    labels_all = []

    # per-subject 统计
    rows_subject_log = []  # (subject, n_epochs_used, n_patches_total, n_phasic, n_tonic, n_unknown)

    subs = list_subjects(args.root)
    total_epochs_all = 0
    for sid in tqdm(subs, desc="Subjects"):
        subj_dir = os.path.join(args.root, sid)
        width = infer_zero_width(subj_dir, 5)
        eids = list_epochs(subj_dir)

        n_epochs_used = 0
        n_phasic = 0
        n_tonic = 0
        n_unknown = 0

        for eid in tqdm(eids, leave=False, desc=f"{sid}"):
            if restrict and (sid, eid) not in restrict:
                continue

            apath = os.path.join(subj_dir, f"{eid:0{width}d}.arrow")
            try:
                mat, stage = read_arrow_epoch(apath)
            except Exception:
                continue

            # 仅处理 REM 期
            if str(stage) not in rem_allow:
                continue

            labs = build_pseudo_labels_for_epoch(
                mat=mat,
                eog_idx=args.eog_idx,
                fs=args.fs,
                amp_phasic=args.amp_phasic,
                amp_tonic=args.amp_tonic,
                max_width_ms=args.max_width_ms,
                min_consecutive=args.min_consecutive,
                separation_k=args.separation_k
            )

            # 汇入索引与标签
            for pid, y in enumerate(labs.tolist()):
                index_rows.append((sid, eid, pid))
                labels_all.append(int(y))
                if y == 1:
                    n_phasic += 1
                elif y == 0:
                    n_tonic += 1
                else:
                    n_unknown += 1

            n_epochs_used += 1
            total_epochs_all += 1

        # 记录每个 subject 的计数
        if n_epochs_used > 0:
            rows_subject_log.append((
                sid,
                n_epochs_used,
                n_phasic + n_tonic + n_unknown,
                n_phasic,
                n_tonic,
                n_unknown
            ))

    # 保存 patch 索引与标签
    df_idx = pd.DataFrame(index_rows, columns=["subject", "epoch", "pid"])
    df_idx.to_csv(os.path.join(args.out_dir, "patch_index.csv"), index=False)
    np.save(os.path.join(args.out_dir, "patch_labels.npy"), np.asarray(labels_all, dtype=np.int8))

    # 保存 per-subject 日志
    df_sub = pd.DataFrame(rows_subject_log,
                          columns=["subject", "n_epochs_used", "n_patches_total", "n_phasic", "n_tonic", "n_unknown"])
    df_sub.to_csv(os.path.join(args.out_dir, "subject_patch_counts.csv"), index=False)

    # 汇总打印
    labs = np.asarray(labels_all, dtype=np.int8)
    uniq, cnts = np.unique(labs, return_counts=True)
    stats = dict(zip([int(u) for u in uniq], [int(c) for c in cnts]))

    print("\n[Done] Saved:")
    print("  -", os.path.join(args.out_dir, "patch_index.csv"))
    print("  -", os.path.join(args.out_dir, "patch_labels.npy"))
    print("  -", os.path.join(args.out_dir, "subject_patch_counts.csv"))
    print(f"Total REM epochs used: {total_epochs_all}")
    print("Label counts (1=phasic, 0=tonic, -1=unknown):", stats)


if __name__ == "__main__":
    main()