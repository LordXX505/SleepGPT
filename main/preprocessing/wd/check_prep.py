#!/usr/bin/env python3
# check_one.py  ——  打印 stage 与 ODS
import os, argparse, textwrap, numpy as np

PREP_DIR = "/Users/hwx_admin/Downloads/第二批/prep"   # 改成你的 prep 目录

def load_npy(tag, kind):
    path = os.path.join(PREP_DIR, f"{tag}_{kind}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} file not found: {path}")
    return np.load(path)

def pretty(arr, maxlen=240):
    return "\n        ".join(textwrap.wrap(" ".join(map(str, arr)), maxlen))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print stage & ODS")
    parser.add_argument("--tag", help="subject tag, e.g., subj001", default='202-2021-1632')
    parser.add_argument("--sec", action="store_true",
                        help="print second-level ods_sec instead of epoch-level ods")
    parser.add_argument("--maxlen", type=int, default=240,
                        help="max chars per printed line")
    args = parser.parse_args()

    stage = load_npy(args.tag, "stage")          # (N_epoch,)
    if args.sec:                                 # 秒级打印
        ods = load_npy(args.tag, "ods_sec")      # (N_epoch*30,)
        print("Stage :\n", pretty(stage, args.maxlen))
        print("ODS_sec :\n", pretty(ods, args.maxlen))
        print(f"\nTotal seconds: {len(ods)},  ODS seconds: {ods.sum()}")
    else:                                        # epoch 级打印
        ods = load_npy(args.tag, "ods")          # (N_epoch,)
        print("Stage :\n", pretty(stage, args.maxlen))
        print("ODS_ep :\n", pretty(ods, args.maxlen))
        pos = np.where(ods == 1)[0]  # 下标数组
        if pos.size == 0:
            print(f"{args.tag}: 该夜无 ODS 阳性 epoch")
        else:
            print(f"{args.tag}: ODS 阳性 epoch 索引 ({len(pos)} 个) →\n{pos.tolist()}")