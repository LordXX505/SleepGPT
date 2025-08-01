import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Data
conf = {
    19: {'TP': 221.0, 'FN': 85.0, 'FP': 120.0},
    18: {'TP': 735.0, 'FN': 253.0, 'FP': 223.0},
    15: {'TP': 39.0, 'FN': 27.0, 'FP': 26.0},
    12: {'TP': 489.0, 'FN': 160.0, 'FP': 214.0},
    13: {'TP': 573.0, 'FN': 115.0, 'FP': 456.0},
    11: {'TP': 424.0, 'FN': 151.0, 'FP': 336.0},
    6: {'TP': 98.0, 'FN': 52.0, 'FP': 115.0},
    5: {'TP': 127.0, 'FN': 66.0, 'FP': 63.0},
    17: {'TP': 379.0, 'FN': 86.0, 'FP': 200.0},
    14: {'TP': 617.0, 'FN': 91.0, 'FP': 355.0},
    8: {'TP': 281.0, 'FN': 83.0, 'FP': 127.0},
    2: {'TP': 916.0, 'FN': 224.0, 'FP': 555.0},
    10: {'TP': 644.0, 'FN': 151.0, 'FP': 192.0},
    9: {'TP': 647.0, 'FN': 112.0, 'FP': 214.0},
    3: {'TP': 98.0, 'FN': 37.0, 'FP': 77.0},
    1: {'TP': 605.0, 'FN': 107.0, 'FP': 199.0},
    16: {'TP': 282.0, 'FN': 99.0, 'FP': 151.0},
    7: {'TP': 695.0, 'FN': 201.0, 'FP': 308.0},
    4: {'TP': 171.0, 'FN': 81.0, 'FP': 90.0}
}

# Calculate precision, recall, f1_score, and total true labels (TP + FN)
data = {'Precision': [], 'Recall': [], 'F1-Score': [], 'Total Labels': [], 'Group': []}

for key, val in conf.items():
    TP = val['TP']
    FN = val['FN']
    FP = val['FP']

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    total_labels = TP + FN

    data['Precision'].append(precision)
    data['Recall'].append(recall)
    data['F1-Score'].append(f1)
    data['Total Labels'].append(total_labels)
    data['Group'].append(key)

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate Pearson correlations
r_prec, p_prec = stats.pearsonr(df['Total Labels'], df['Precision'])
r_rec, p_rec = stats.pearsonr(df['Total Labels'], df['Recall'])
r_f1, p_f1 = stats.pearsonr(df['Total Labels'], df['F1-Score'])

# Set up subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

# Precision subplot
sns.regplot(x='Total Labels', y='Precision', data=df, ax=axs[0], color='blue', scatter_kws={'s': 50},
            line_kws={'color': 'blue'})
axs[0].set_title(f'Precision (r={r_prec:.2f}, p={p_prec:.4f})')
axs[0].set_xlabel('Total Labels')
axs[0].set_ylabel('Precision')

# Recall subplot
sns.regplot(x='Total Labels', y='Recall', data=df, ax=axs[1], color='green', scatter_kws={'s': 50},
            line_kws={'color': 'green'})
axs[1].set_title(f'Recall (r={r_rec:.2f}, p={p_rec:.4f})')
axs[1].set_xlabel('Total Labels')
axs[1].set_ylabel('Recall')

# F1-Score subplot
sns.regplot(x='Total Labels', y='F1-Score', data=df, ax=axs[2], color='red', scatter_kws={'s': 50},
            line_kws={'color': 'red'})
axs[2].set_title(f'F1-Score (r={r_f1:.2f}, p={p_f1:.4f})')
axs[2].set_xlabel('Total Labels')
axs[2].set_ylabel('F1-Score')

# Adjust layout for better visibility
plt.tight_layout()
plt.savefig('./result/spindle_results/expert1_p.svg')
# Show the plot
plt.show()