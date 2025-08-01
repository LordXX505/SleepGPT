#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成:
  └── {tag}_stage.npy   # (N_epoch,)  int16
  └── {tag}_ods_sec.npy # (N_epoch*30,) uint8
  └── {tag}_ods.npy     # (N_epoch,)  uint8
"""
import os, glob, re, json
import numpy as np
import pandas as pd

# ========= 0. 路径配置 =================================================
ROOT_UMS = "/Users/hwx_admin/Downloads/第一批/UMS"  #
SAVE_DIR = os.path.join(ROOT_UMS, "../prep")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 1. 工具函数 =================================================
def clean(line: str) -> str:                       # 去除控制字符
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", line)

def load_stage_final(csv_path: str) -> np.ndarray: # 从 stage_final.csv 读 stage
    df = pd.read_csv(csv_path)
    col = "stage" if "stage" in df.columns else df.columns[0]
    return df[col].astype(np.int16).values         # (N_epoch,)

def parse_stage_od(j_path: str):
    """
    解析 stage_od.json
    返回:
        stage_dict {epoch->stage}      (可能为空)
        events     [(start,len), …]    (秒)
        max_page   int (>=0) or -1
    """
    stage_dict, events, max_page = {}, [], -1
    with open(j_path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(clean(raw))
            except json.JSONDecodeError:
                continue

            page = d.get("page") or d.get("stage") or d.get("epoch")
            if page is not None:
                page = int(page) - 1 if int(page) > 0 else int(page)
                max_page = max(max_page, page)
                if "stage" in d:
                    stage_dict[page] = int(d["stage"])

            for ev in d.get("ods", []):
                try:
                    s, l = int(ev["start"]), int(ev["length"])
                    events.append((s, l))
                except (KeyError, ValueError, TypeError):
                    continue
    return stage_dict, events, max_page

def build_sec_mask(events, total_sec):
    """events=[(start,len)], return (total_sec,) 0/1"""
    m = np.zeros(total_sec, dtype=np.uint8)
    for s, l in events:
        e = min(s + l, total_sec)
        m[s:e] = 1
    return m

# ========= 2. 收集记录目录 =============================================
record_dirs = set()
for fp in glob.glob(os.path.join(ROOT_UMS, "**", "*stage_*.*"), recursive=True):
    record_dirs.add(os.path.dirname(fp))

# ========= 3. 主流程 ===================================================
errors = []
valid_dict = {}
for rec_dir in sorted(record_dirs):
    tag = os.path.basename(rec_dir).split(" ")[0]          # 记录名
    jlist = glob.glob(os.path.join(rec_dir, "*stage_od.json"))
    clist = glob.glob(os.path.join(rec_dir, "*stage_final.csv"))

    if not jlist and not clist:
        continue

    try:
        # ---------- 3.1 解析 stage_od (如果有) -------------------------
        stage_dict, events, max_page = {}, [], -1
        if jlist:
            stage_dict, events, max_page = parse_stage_od(jlist[0])

        # ---------- 3.2 解析/补全 stage_final ------------------------
        if clist:
            stage_final = load_stage_final(clist[0])           # (N_csv,)
        else:
            stage_final = np.zeros(0, dtype=np.int16)

        # --------- 3.3 确定 N_epoch & stage array -------------------
        if stage_dict:
            n_epoch = max(max_page + 1, len(stage_final))
            stage_arr = np.full(n_epoch, -1, np.int16)
            for p, s in stage_dict.items():
                stage_arr[p] = s
            # 仍为 -1 的槽位，如有 CSV 则填充
            if stage_final.size and (-1 in stage_arr):
                stage_arr[stage_arr == -1] = stage_final[stage_arr == -1]
        elif stage_final.size:
            n_epoch  = len(stage_final)
            stage_arr = stage_final.copy()
        else:                                  # 没有任何 stage 信息
            print(f"[Warn] no stage label in {rec_dir}")
            continue

        total_sec = n_epoch * 30

        # ---------- 3.4 秒级 ODS 掩码 -------------------------------
        if events:
            ods_sec = build_sec_mask(events, total_sec)       # (S,)
            ods_valid = 1
        else:
            ods_sec = np.zeros(total_sec, dtype=np.uint8)
            ods_valid = 0
        # ---------- 3.5 epoch-level ODS ----------------------------
        ods_ep = ods_sec.reshape(n_epoch, 30).max(axis=1)     # (N,)
        valid_dict[tag] = ods_valid
        # ---------- 3.6 保存 ---------------------------------------
        np.save(os.path.join(SAVE_DIR, f"{tag}_stage.npy"),   stage_arr)
        np.save(os.path.join(SAVE_DIR, f"{tag}_ods_sec.npy"), ods_sec)
        np.save(os.path.join(SAVE_DIR, f"{tag}_ods.npy"),     ods_ep)

        print(f"✔ {tag}: epochs={n_epoch}, ods_ep={ods_ep.sum()}")

    except Exception as e:
        errors.append({"dir": rec_dir, "err": str(e)})
        print(f"[Error] {rec_dir}: {e}")

print("\nDone. Errors:", errors if errors else "None")
np.save(os.path.join(SAVE_DIR, f"valid_ods.npy"), valid_dict)
