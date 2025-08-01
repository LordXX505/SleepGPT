import os.path
import numpy as np
import torch
import pandas as pd
def macro_f1(cm: np.ndarray):
    """Macro-F1 for多分类混淆矩阵 (numpy array)."""
    tp  = np.diag(cm)
    fp  = cm.sum(axis=0) - tp
    fn  = cm.sum(axis=1) - tp
    precision = np.where(tp+fp == 0, 0, tp / (tp + fp))
    recall    = np.where(tp+fn == 0, 0, tp / (tp + fn))
    f1 = np.where(precision+recall == 0, 0, 2*precision*recall / (precision+recall))
    return f1.mean()

def cohen_kappa(cm: np.ndarray):
    """Cohen's kappa for多分类."""
    total = cm.sum()
    po = np.trace(cm) / total
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (total**2)
    return (po - pe) / (1 - pe) if (1-pe) else np.nan

def load_json(file_path):
    import json
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res
def main(res_path):
    res = torch.load(res_path, map_location=torch.device('cpu'))
    # res = load_json(os.path.join('/Users/hwx_admin/Sleep/temp_log/ums/Test_F3/res.json'))
    print(res)
    acc_list = []
    overall_cm = np.zeros((4, 4), dtype=int)

    for subj, cm in res.items():
        if isinstance(cm, torch.Tensor):
            cm = cm.numpy()
        correct = np.trace(cm)
        total = np.sum(cm)
        acc = correct / total
        acc_list.append({'Subject': subj, 'Accuracy': round(acc, 4)})
        overall_cm += cm

    # overall_cm = np.array([[21138, 4452, 102, 815],  # Wake (PSG)
    #                [6805, 98618, 4991, 4537],  # Light sleep
    #                [237, 4737, 19402, 47],  # Deep sleep
    #                [950, 4010, 5, 23740]])

    row_tot = overall_cm.sum(axis=1, keepdims=True)
    row_pct = overall_cm / row_tot  # 每行除以该行总计  →  与图中 (%) 一致

    # -------- 3. 生成 “计数 (百分比%)” 字符串 --------
    labels = ["Wake", "Light sleep", "Deep sleep", "REM sleep"]
    tbl = pd.DataFrame(index=["Wake", "Light sleep", "Deep sleep", "REM sleep"],
                       columns=["Wake", "Light sleep", "Deep sleep", "REM sleep"])

    for i, rlab in enumerate(labels):
        for j, clab in enumerate(labels):
            count = overall_cm[i, j]
            pct = row_pct[i, j] * 100
            tbl.loc[rlab, clab] = f"{count:,} ({pct:.1f}%)"

    print(tbl.to_string())

    # 构建DataFrame
    df_acc = pd.DataFrame(acc_list)
    print(df_acc)

    # 输出 overall accuracy
    print(np.trace(overall_cm), overall_cm.sum())
    overall_acc = np.trace(overall_cm) / overall_cm.sum()
    overall_mf1 = macro_f1(overall_cm)
    overall_kappa = cohen_kappa(overall_cm)

    print(f"\nOverall Acc  : {overall_acc:.4f}")
    print(f"Overall MF1  : {overall_mf1:.4f}")
    print(f"Overall Kappa: {overall_kappa:.4f}")


if __name__ == '__main__':

    root_path = '/Users/hwx_admin/Sleep/temp_log'
    kfold = '11'
    sub_name = 'ModelCheckpoint-epoch=24-val_acc=0.7900-val_macro=0.7387-val_score=6.5287.ckpt'
    res_path = os.path.join(root_path, kfold, sub_name)
    main(res_path)