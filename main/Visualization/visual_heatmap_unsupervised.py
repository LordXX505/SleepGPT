import matplotlib.pyplot as plt
import numpy as np

# 数据
systems = ['SSL-ECG', 'CPC', 'SimCLR', 'TS2VEC', 'TST', 'FEAT', 'C-MAE', 'TS-TCC', 'MTS-LOF', 'SleepGPT']
acc_means = np.array([74.56, 82.82, 78.91, 83.33, 83.43, 81.78, 82.11, 83.00, 84.35, 85.93])
mf1_means = np.array([65.44, 73.94, 68.60, 73.23, 73.23, 72.58, 71.86, 73.57, 73.52, 80.6])
acc_errors = np.array([0.60, 1.68, 3.11, 1.54, 0.24, 0.28, 0.31, 0.71, 0.31, 0.22])
mf1_errors = np.array([0.97, 1.75, 2.71, 2.17, 2.17, 0.31, 0.15, 0.74, 0.58, 0.57])

# 柱宽和位置
bar_width = 0.35
index = np.arange(len(systems))

# 创建图形和轴
fig, ax = plt.subplots(figsize=(14, 8))
acc_bars = ax.bar(index - bar_width/2, acc_means, bar_width, yerr=acc_errors,
                  capsize=5, color='#E4E45F', label='ACC')

# 绘制MF1柱状图
mf1_bars = ax.bar(index + bar_width/2, mf1_means, bar_width, yerr=mf1_errors,
                  capsize=5, color='#AB84B6', label='MF1')


ax.bar(index[-1] - bar_width/2, acc_means[-1], bar_width,  color='#E4E45F', linewidth=2, alpha=1.0, hatch='/')
ax.bar(index[-1] + bar_width/2,  mf1_means[-1], bar_width, color='#AB84B6', linewidth=2, alpha=1.0,  hatch='/')


ax.set_xlabel('System')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison of Different Systems')
ax.set_xticks(index)
ax.set_xticklabels(systems, rotation=45, ha="right")
ax.legend()
ax.set_ylim(63, 87)
# 显示图形
plt.savefig('../result/heatmap/unsupervised.svg')
plt.tight_layout()
plt.show()