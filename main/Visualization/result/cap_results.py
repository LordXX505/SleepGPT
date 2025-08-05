import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
def get_stage0():
    cm_list = [
        np.array([[ 7,  1,  1], [ 0, 39,  1], [ 2,  9, 11]]),
        np.array([[ 6,  1,  2], [ 0, 36,  4], [ 3,  9, 10]]),
        np.array([[ 6,  1,  2], [ 0, 35,  5], [ 2, 10, 10]]),
        np.array([[ 6,  1,  2], [ 0, 38,  2], [ 3, 10,  9]]),
        np.array([[ 6,  1,  2], [ 0, 37,  3], [ 2, 13,  7]]),
        np.array([[ 7,  1,  1], [ 0, 36,  4], [ 2, 10, 10]]),
        np.array([[ 6,  1,  2], [ 0, 39,  1], [ 3, 11,  8]]),
        np.array([[ 6,  1,  2], [ 0, 38,  2], [ 2, 11,  9]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 2, 12,  8]]),
        np.array([[ 6,  1,  2], [ 0, 39,  1], [ 2, 12,  8]])
    ]
    return cm_list
def get_stage1():
    cm_list = [
        np.array([[ 7,  1,  1], [ 0, 36,  4], [ 4,  5, 13]]),
        np.array([[ 7,  1,  1], [ 0, 36,  4], [ 5,  6, 11]]),
        np.array([[ 7,  1,  1], [ 0, 35,  5], [ 5,  8,  9]]),
        np.array([[ 7,  1,  1], [ 0, 37,  3], [ 6,  8,  8]]),
        np.array([[ 7,  1,  1], [ 0, 36,  4], [ 6,  7,  9]]),
        np.array([[ 7,  1,  1], [ 0, 37,  3], [ 5,  8,  9]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 6,  6, 10]]),
        np.array([[ 7,  1,  1], [ 0, 37,  3], [ 6,  9,  7]]),
        np.array([[ 7,  1,  1], [ 0, 37,  3], [ 4,  6, 12]]),
        np.array([[ 7,  1,  1], [ 0, 37,  3], [ 6,  7,  9]])
    ]
    return cm_list

def get_stage2():
    cm_list = [
        np.array([[ 7,  1,  1], [ 0, 37,  3], [ 4,  3, 15]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 5,  3, 14]]),
        np.array([[ 6,  1,  2], [ 0, 38,  2], [ 5,  4, 13]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 6,  3, 13]]),
        np.array([[ 6,  1,  2], [ 0, 37,  3], [ 5,  4, 13]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 5,  5, 12]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 5,  3, 14]]),
        np.array([[ 6,  1,  2], [ 0, 38,  2], [ 5,  6, 11]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 5,  3, 14]]),
        np.array([[ 7,  1,  1], [ 0, 36,  4], [ 5,  4, 13]])
    ]
    return cm_list

# Function to calculate class-wise accuracy, precision, recall, and F1 score
def calculate_classwise_metrics(cm):
    total = np.sum(cm)  # Total number of samples
    accuracy = np.trace(cm) / total  # Accuracy

    class_metrics = {}

    # Calculate metrics for each class
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP  # False positives for class i
        FN = np.sum(cm[i, :]) - TP  # False negatives for class i
        TN = total - (TP + FP + FN)  # True negatives for class i

        # Precision, Recall, and F1 for class i
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # Sensitivity (Recall for class i)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Specificity (True Negative Rate for class i)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Class-wise accuracy calculation
        total_for_class = TP + FP + FN + TN
        class_accuracy = (TP + TN) / total_for_class if total_for_class > 0 else 0

        class_metrics[f'Class {i}'] = {
            'Accuracy': class_accuracy,
            'Precision': precision,
            'Recall': recall,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1 Score': f1
        }

    return accuracy, class_metrics
def get_stage3():
    cm_list = [
        np.array([[ 6,  1,  2], [ 0, 38,  2], [ 6,  2, 14]]),
        np.array([[ 5,  1,  3], [ 0, 37,  3], [ 6,  3, 13]]),
        np.array([[ 6,  1,  2], [ 0, 37,  3], [ 6,  5, 11]]),
        np.array([[ 6,  1,  2], [ 0, 37,  3], [ 6,  4, 12]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 6,  5, 11]]),
        np.array([[ 7,  0,  2], [ 0, 37,  3], [ 5,  4, 13]]),
        np.array([[ 7,  1,  1], [ 0, 38,  2], [ 6,  4, 12]]),
        np.array([[ 5,  1,  3], [ 0, 36,  4], [ 6,  6, 10]]),
        np.array([[ 5,  1,  3], [ 0, 38,  2], [ 5,  4, 13]]),
        np.array([[ 6,  1,  2], [ 0, 35,  5], [ 6,  4, 12]])
    ]
    return cm_list

# 你提供的混淆矩阵数据
def get_stage4():
    cm_list = [
        np.array([[ 6,  1,  1], [ 0, 35,  5], [ 6,  2, 14]]),
        np.array([[ 5,  1,  2], [ 0, 37,  3], [ 4,  1, 17]]),
        np.array([[ 5,  1,  2], [ 0, 36,  4], [ 5,  5, 12]]),
        np.array([[ 6,  1,  1], [ 0, 36,  4], [ 6,  3, 13]]),
        np.array([[ 4,  1,  3], [ 0, 35,  5], [ 6,  3, 13]]),
        np.array([[ 6,  1,  1], [ 0, 36,  4], [ 6,  3, 13]]),
        np.array([[ 5,  1,  2], [ 0, 36,  4], [ 6,  4, 12]]),
        np.array([[ 5,  1,  2], [ 0, 36,  4], [ 4,  4, 14]]),
        np.array([[ 4,  1,  3], [ 0, 36,  4], [ 5,  1, 16]]),
        np.array([[ 5,  1,  2], [ 0, 35,  5], [ 6,  3, 13]])
    ]
    return cm_list



cm_list=get_stage4()
# Calculate metrics for each matrix
metrics = [calculate_classwise_metrics(cm) for cm in cm_list]

# Print the metrics for each confusion matrix
for i, (acc, class_metrics) in enumerate(metrics):
    print(f"Matrix {i + 1} Metrics:")
    print(f"  Accuracy (Overall): {acc:.4f}")

    for class_name, ms in class_metrics.items():
        print(f"  {class_name}:")
        print(f"    Accuracy: {ms['Accuracy']:.4f}")
        print(f"    Precision: {ms['Precision']:.4f}")
        print(f"    Recall: {ms['Recall']:.4f}")
        print(f"    Sensitivity: {ms['Sensitivity']:.4f}")
        print(f"    Specificity: {ms['Specificity']:.4f}")
        print(f"    F1 Score: {ms['F1 Score']:.4f}")

    print()
class_acc = {i: [] for i in range(3)}
class_precision = {i: [] for i in range(3)}
class_recall = {i: [] for i in range(3)}
class_sensitivity = {i: [] for i in range(3)}
class_specificity = {i: [] for i in range(3)}
class_f1 = {i: [] for i in range(3)}

# Collect metrics
for _, class_metrics in metrics:
    for i in range(3):
        class_acc[i].append(class_metrics[f'Class {i}']['Accuracy'])
        class_precision[i].append(class_metrics[f'Class {i}']['Precision'])
        class_recall[i].append(class_metrics[f'Class {i}']['Recall'])
        class_sensitivity[i].append(class_metrics[f'Class {i}']['Sensitivity'])
        class_specificity[i].append(class_metrics[f'Class {i}']['Specificity'])
        class_f1[i].append(class_metrics[f'Class {i}']['F1 Score'])

# Calculate the mean and std for each class
for i in range(3):
    print(f"\nClass {i} Metrics:")
    print(f"  Accuracy - Mean: {np.mean(class_acc[i]):.4f}, Std: {np.std(class_acc[i]):.4f}")
    print(f"  Precision - Mean: {np.mean(class_precision[i]):.4f}, Std: {np.std(class_precision[i]):.4f}")
    print(f"  Recall - Mean: {np.mean(class_recall[i]):.4f}, Std: {np.std(class_recall[i]):.4f}")
    print(f"  Sensitivity - Mean: {np.mean(class_sensitivity[i]):.4f}, Std: {np.std(class_sensitivity[i]):.4f}")
    print(f"  Specificity - Mean: {np.mean(class_specificity[i]):.4f}, Std: {np.std(class_specificity[i]):.4f}")
    print(f"  F1 Score - Mean: {np.mean(class_f1[i]):.4f}, Std: {np.std(class_f1[i]):.4f}")

# 获取所有的 confusion matrices
cm_list = get_all().values()

# 绘制平均混淆矩阵
plot_average_confusion_matrix(cm_list)
