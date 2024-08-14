import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

conf = [0] * 20
TP = 'TP'
FN = 'FN'
FP = 'FP'
conf[19] = { TP: 221.0, FN: 85.0, FP: 120.0}
conf[18] = {TP: 735.0, FN: 253.0, FP: 223.0,}
conf[15] = { TP: 39.0, FN: 27.0, FP: 26.0}
conf[12] = {TP: 489.0, FN: 160.0, FP: 214.0,}
conf[13] = { TP: 573.0, FN: 115.0, FP: 456.0, }
conf[11] = { TP: 424.0, FN: 151.0, FP: 336.0}
conf[6] = { TP: 98.0, FN: 52.0, FP: 115.0}
conf[5] = { TP: 127.0, FN: 66.0, FP: 63.0,}
conf[17] = {TP: 379.0, FN: 86.0, FP: 200.0}
conf[14] = {TP: 617.0, FN: 91.0, FP: 355.0,}
conf[8] = {TP: 281.0, FN: 83.0, FP: 127.0}
conf[2] = {TP: 916.0, FN: 224.0, FP: 555.0,}
conf[10] = { TP: 644.0, FN: 151.0, FP: 192.0}
conf[9] = {TP: 647.0, FN: 112.0, FP: 214.0}
conf[3] = { TP: 98.0, FN: 37.0, FP: 77.0,}
conf[1] = { TP: 605.0, FN: 107.0, FP: 199.0}
conf[16] = {TP: 282.0, FN: 99.0, FP: 151.0,}
conf[7] = { TP: 695.0, FN: 201.0, FP: 308.0}
conf[4] = { TP: 171.0, FN: 81.0, FP: 90.0, }
# conf[19] = {TP: 884.0, FN: 142.0, FP: 449.0}
# conf[18] = {TP: 1139.0, FN: 278.0, FP: 495.0}
# conf[12] = {TP: 918.0, FN: 159.0, FP: 321.0}
# conf[11] = {TP: 1102.0, FN: 337.0, FP: 516.0}
# conf[6] = {TP: 446.0, FN: 390.0, FP: 229.0}
# conf[5] = {TP: 451.0, FN: 189.0, FP: 231.0}
# conf[17] = {TP: 982.0, FN: 204.0, FP: 347.0}
# conf[13] = {TP: 1145.0, FN: 251.0, FP: 304.0}
# conf[2] = {TP: 1927.0, FN: 278.0, FP: 580.0}
# conf[14] = {TP: 1373.0, FN: 235.0, FP: 516.0}
# conf[10] = {TP: 1609.0, FN: 328.0, FP: 200.0}
# conf[3] = {TP: 434.0, FN: 106.0, FP: 206.0}
# conf[9] = {TP: 1219.0, FN: 281.0, FP: 224.0}
# conf[7] = {TP: 1235.0, FN: 341.0, FP: 230.0}
# conf[1] = {TP: 1319.0, FN: 314.0, FP: 306.0}


# 计算每个类别的指标
def calculate_metrics(conf):
    total_tp = total_fn = total_fp = 0
    metrics_results = []

    for i, metrics in enumerate(conf):
        if metrics != 0:
            tp = metrics[TP]
            fn = metrics[FN]
            fp = metrics[FP]

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            total = tp + fn  # 总个数

            metrics_results.append((i, f1_score, sensitivity, ppv, total))

            total_tp += tp
            total_fn += fn
            total_fp += fp
        else:
            metrics_results.append((i, 0, 0, 0, 0))

    # 计算总体指标
    overall_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1_score = 2 * (overall_ppv * overall_sensitivity) / (overall_ppv + overall_sensitivity) if (
                                                                                                                    overall_ppv + overall_sensitivity) > 0 else 0

    metrics_results.append(('Overall', overall_f1_score, overall_sensitivity, overall_ppv, total_tp + total_fn))

    return metrics_results


metrics_results = calculate_metrics(conf)
print(metrics_results)
valid_metrics = [(res[1], res[4]) for res in metrics_results[:-1] if res[1] > 0 and res[4] > 0]
f1_scores, totals = zip(*valid_metrics)


# Calculate the Pearson correlation coefficient

# 提取各个指标
categories = [str(result[0]) for result in metrics_results]
sensitivities = [result[2] for result in metrics_results]
ppvs = [result[3] for result in metrics_results]
correlation_coef, p_value = pearsonr(totals, f1_scores)

print(f"Pearson Correlation Coefficient: {correlation_coef:.4f}")
print(f"P-Value: {p_value:.4f}")
# 绘制条形图
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# F1 Score
ax1 = axs[0]
ax2 = ax1.twinx()
ax1.bar(categories[1:], f1_scores[1:], color='blue')
ax2.plot(categories[1:-1], [totals[i] for i in range(len(totals)-1)], color='red', marker='o')
ax1.set_ylim(0.5, 0.9)  # 设置左边y轴范围
ax2.set_ylim(50, 1500)  # 设置右边y轴范围

ax1.set_xlabel('Category')
ax1.set_ylabel('F1 Score')
ax2.set_ylabel('Total Count')
ax1.set_title('F1 Score by Category')

# Sensitivity
ax1 = axs[1]
ax2 = ax1.twinx()
ax1.bar(categories[1:], sensitivities[1:], color='green')
ax2.plot(categories[1:-1], [totals[i] for i in range(len(totals)-1)], color='red', marker='o')
ax1.set_ylim(0.5, 0.9)  # 设置左边y轴范围
ax2.set_ylim(50, 1500)  # 设置右边y轴范围

ax1.set_xlabel('Category')
ax1.set_ylabel('Sensitivity')
ax2.set_ylabel('Total Count')
ax1.set_title('Sensitivity by Category')

# PPV
ax1 = axs[2]
ax2 = ax1.twinx()
ax1.bar(categories[1:], ppvs[1:], color='orange')
ax2.plot(categories[1:-1], [totals[i] for i in range(len(totals)-1)], color='red', marker='o')
ax1.set_ylim(0.5, 0.9)  # 设置左边y轴范围
ax2.set_ylim(50, 1500)  # 设置右边y轴范围


ax1.set_xlabel('Category')
ax1.set_ylabel('PPV')
ax2.set_ylabel('Total Count')
ax1.set_title('PPV by Category')

plt.tight_layout()
os.makedirs('./result/spindle_results/', exist_ok=True)
plt.savefig('./result/spindle_results/expert1_aug1.svg')
plt.show()