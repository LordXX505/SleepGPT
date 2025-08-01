import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 假设已经计算了Precision、Recall和F1-Score
# 在你的数据中，每个样本或每个实验都有相应的 TP、FN 和 FP 值
precisions = [0.9, 0.8, 0.75, 0.85, 0.88, 0.76, 0.92, 0.81, 0.79, 0.84]  # 示例数据
recalls = [0.7, 0.65, 0.78, 0.8, 0.82, 0.68, 0.89, 0.77, 0.75, 0.85]    # 示例数据
f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]

# 将数据放入DataFrame以便于绘图
data = {
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1_scores
}
df = pd.DataFrame(data)

# 创建分布的box plot
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Box plot for Precision
plt.subplot(1, 3, 1)
sns.boxplot(data=df['Precision'])
plt.title('Precision Distribution')

# Box plot for Recall
plt.subplot(1, 3, 2)
sns.boxplot(data=df['Recall'])
plt.title('Recall Distribution')

# Box plot for F1-Score
plt.subplot(1, 3, 3)
sns.boxplot(data=df['F1-Score'])
plt.title('F1-Score Distribution')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Histogram for Precision
plt.subplot(1, 3, 1)
sns.histplot(df['Precision'], bins=10, kde=True)
plt.title('Precision Distribution')

# Histogram for Recall
plt.subplot(1, 3, 2)
sns.histplot(df['Recall'], bins=10, kde=True)
plt.title('Recall Distribution')

# Histogram for F1-Score
plt.subplot(1, 3, 3)
sns.histplot(df['F1-Score'], bins=10, kde=True)
plt.title('F1-Score Distribution')

plt.tight_layout()
plt.show()