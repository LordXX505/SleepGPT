import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# ==============================
# 把你提供的矩阵放到一个列表里
# ==============================
matrices = [
    np.array([[742, 51, 70, 1, 76],
              [150, 168, 311, 5, 250],
              [17, 66, 5298, 238, 158],
              [0, 0, 541, 940, 0],
              [14, 52, 81, 9, 1162]]),
np.array([[ 658,  185,  224,    3,  107],
        [  78,  251,  343,    0,  250],
        [  66,   98, 4537,  157,  124],
        [   6,   12,  499,  677,   13],
        [  24,   27,  169,    1, 1251]]),
    np.array([[742, 51, 70, 1, 76],
              [150, 168, 311, 5, 250],
              [17, 66, 5298, 238, 158],
              [0, 0, 541, 940, 0],
              [14, 52, 81, 9, 1162]]),
np.array([[ 505,  202,  219,   13,  101],
        [  73,  205,  340,   21,  233],
        [  54,   77, 4629,  218,  149],
        [  33,    0,  293,  936,    1],
        [  43,   26,  202,   24, 1243]]),
np.array([[ 806,  185,   45,    7,   68],
        [ 108,  254,  448,   10,  333],
        [  13,  121, 4201,  177,  154],
        [   4,    0,  341,  717,    6],
        [  14,   75,  137,    3, 1533]]),
np.array([[1025,  152,   99,    0,   73],
        [  85,  179,  377,    0,  317],
        [   5,   31, 4381,  221,   88],
        [   0,    0,  339,  573,    3],
        [  11,   60,  119,    0, 1542]]),

]

# ==============================
# 累加总体混淆矩阵
# ==============================
total_cm = np.zeros_like(matrices[0])
for m in matrices:
    total_cm += m

print("===== 总体混淆矩阵 =====")
print(total_cm)

# ==============================
# 计算总体指标
# ==============================
y_true, y_pred = [], []
n_classes = total_cm.shape[0]
for i in range(n_classes):
    for j in range(n_classes):
        count = total_cm[i, j]
        if count > 0:
            y_true.extend([i] * count)
            y_pred.extend([j] * count)

overall_acc = accuracy_score(y_true, y_pred)
overall_mf1 = f1_score(y_true, y_pred, average='macro')
overall_kappa = cohen_kappa_score(y_true, y_pred)

print("\n===== 总体指标 =====")
print(f"Overall Acc  : {overall_acc:.4f}")
print(f"Overall MF1  : {overall_mf1:.4f}")
print(f"Overall Kappa: {overall_kappa:.4f}")

# ==============================
# 每类 F1
# ==============================
per_class_f1 = f1_score(y_true, y_pred, average=None)
# labels = ["Wake", "Light sleep", "Deep sleep", "REM sleep"]
labels = ["Wake", "N1", "N2", "N3", "REM sleep"]


print("\n===== 每类 F1 分数 =====")
for lbl, score in zip(labels, per_class_f1):
    print(f"{lbl:12s}: {score:.4f}")