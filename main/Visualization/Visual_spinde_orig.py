import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conf = [0] * 20
TP = 'TP'
FN = 'FN'
FP = 'FP'
conf[13] = {TP: 546.0, FN: 142.0, FP: 391.0}
conf[11] = {TP: 392.0, FN: 183.0, FP: 319.0}
conf[6] = {TP: 85.0, FN: 65.0, FP: 77.0}
conf[5] = {TP: 141.0, FN: 52.0, FP: 90.0}
conf[17] = {TP: 369.0, FN: 96.0, FP: 182.0}
conf[14] = {TP: 573.0, FN: 135.0, FP: 310.0}
conf[8] = {TP: 306.0, FN: 58.0, FP: 158.0}
conf[2] = {TP: 969.0, FN: 171.0, FP: 776.0}
conf[10] = {TP: 648.0, FN: 147.0, FP: 222.0}
conf[9] = {TP: 626.0, FN: 133.0, FP: 224.0}
conf[3] = {TP: 97.0, FN: 38.0, FP: 71.0}
conf[1] = {TP: 587.0, FN: 125.0, FP: 202.0}
conf[16] = {TP: 296.0, FN: 85.0, FP: 197.0}
conf[7] = {TP: 706.0, FN: 190.0, FP: 339.0}
conf[4] = {TP: 187.0, FN: 65.0, FP: 121.0}
conf[19] = {TP: 195.0, FN: 111.0, FP: 89.0}
conf[18] = {TP: 707.0, FN: 281.0, FP: 218.0}
conf[15] = {TP: 48.0, FN: 18.0, FP: 48.0}
conf[12] = {TP: 529.0, FN: 120.0, FP: 268.0}



# Initialize lists to store metrics
TP_total = 0
FN_total = 0
FP_total = 0

for subject in conf:
    if subject != 0:
        TP_total += subject['TP']
        FN_total += subject['FN']
        FP_total += subject['FP']

# Calculate overall metrics
overall_precision = TP_total / (TP_total + FP_total)
overall_recall = TP_total / (TP_total + FN_total)
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

# Plot the overall metrics
metrics = ['Precision', 'Recall', 'F1-Score']
values = [overall_precision, overall_recall, overall_f1]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Overall Metrics')
plt.ylim(0, 1)
import os
os.makedirs('./result/spindle_results/', exist_ok=True)
plt.savefig('./result/spindle_results/expert2_aug0.svg')
plt.show()

overall_metrics = {
    'Overall Precision': overall_precision,
    'Overall Recall': overall_recall,
    'Overall F1-Score': overall_f1
}
