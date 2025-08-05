import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from scipy.stats import ttest_ind
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def plot_f1score_per_subject(individual_values, subject):
    """
    绘制每个被试的 per-class F1-score，展示原始与增强模型之间的对比
    """
    original_f1_scores = individual_values['f1_per_class'][0]
    augmented_f1_scores = individual_values['f1_per_class'][1]
    subjects = [f'Sub{i+1}' for i in range(len(original_f1_scores))]  # 自动生成被试名
    # 分组，用不同颜色区分 (Improved: F1增强, Unimproved: F1下降)
    for i, label in enumerate(['W', 'N1', 'N2', 'N3', 'REM']):
        colors = ['magenta' if augmented_f1_scores[j][i] >= original_f1_scores[j][i] else 'blue' for j in range(len(subjects))]

        # 创建图表
        plt.figure(figsize=(6, 8))

        # 遍历每个subject，绘制线条
        for j in range(len(subjects)):
            plt.plot([0, 1], [original_f1_scores[j][i], augmented_f1_scores[j][i]], color=colors[j], marker='o', lw=2)

        # 设置x轴刻度标签
        plt.xticks([0, 1], ['Original', 'Augmented'])

        # 添加图例
        plt.legend(['Improved', 'Unimproved'], loc='upper right')

        # 设置标题和轴标签
        plt.title('Per-Class F1-score Comparison for Each Subject')
        plt.savefig(f'../../result/portion_edf_bar/{subject}-PerF1-score-{label}.svg')
        plt.ylim([0, 1])

        # 显示图表
        plt.grid(True)
        plt.show()


def get_radar_f1(metrics_per_subject, subject, num_vars = 5):
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    values = metrics_per_subject['f1_per_class'][0]
    values1 = metrics_per_subject['f1_per_class'][1]
    values = np.append(values, values[0])  # 数据闭合
    values1 = np.append(values1, values1[0])  # 数据闭合
    # values = np.where(values<0.9, 0.901, values)
    # values1 = np.where(values1<0.9, 0.901, values1)
    ax.plot(angles, values, '#4DBADB', 'ro-', linewidth=2, label='0')  # 添加Model 0数据线
    ax.fill(angles, values,  '#4DBADB', alpha=0.25)
    ax.scatter(angles, values, color='#4DBADB', marker='o')
    ax.plot(angles, values1, '#E44A33', 'ro-', linewidth=2, label='1')  # 添加Model 1数据线
    ax.fill(angles, values1,  '#E44A33', alpha=0.25)
    ax.scatter(angles, values1, color='#E44A33', marker='o')

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_rlabel_position(90)
    ax.set_rticks([0, 0.2, 0.4, 0.6, 0.8,  1.0])  # 设置网格线
    ax.set_ylim([0, 1.1])
    ax.set_yticklabels([])  # 隐藏刻度值
    plt.savefig(f'../../result/portion_edf_bar/{subject}_radar.svg')
    plt.show()



# 计算 Accuracy、Macro-F1、Kappa 指标
def compute_metrics(confusion_matrix):
    acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    f1_per_class = []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_per_class.append(f1)
    mf1 = np.mean(f1_per_class)

    y_true = []
    y_pred = []
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            y_true += [i] * confusion_matrix[i, j]
            y_pred += [j] * confusion_matrix[i, j]
    kappa = cohen_kappa_score(y_true, y_pred)

    return acc, mf1, kappa, f1_per_class


# 读取所有被试的混淆矩阵，并计算指标，返回每个被试的分布值
# def get_metrics_per_subject(data_path, subject, modes):
#     metrics_results = {'acc': [], 'mf1': [], 'kappa': [], 'f1_per_class': []}
#     individual_values = {'acc': [], 'mf1': [], 'kappa': [], 'f1_per_class': []}  # 存储每个被试的实际值
#     confusion_res = [[0 for _ in range(5)] for _ in range(5)]
#     for mode in modes:
#         p = os.path.join(data_path, mode + subject + '.ckpt')
#         ckpt = torch.load(p, map_location=torch.device('cpu'))

#         acc_vals, mf1_vals, kappa_vals, f1_per_class_vals = [], [], [], []
#         for subject_name, confusion_matrix in ckpt.items():
#             confusion_matrix = confusion_matrix.numpy()
#             confusion_res += confusion_matrix
#             acc, mf1, kappa, f1_per_class = compute_metrics(confusion_matrix)
#             acc_vals.append(acc)
#             mf1_vals.append(mf1)
#             kappa_vals.append(kappa)
#             f1_per_class_vals.append(f1_per_class)
#         f1_per_class_vals = np.array(f1_per_class_vals)
#         metrics_results['acc'].append(np.mean(acc_vals))
#         metrics_results['mf1'].append(np.mean(mf1_vals))
#         metrics_results['kappa'].append(np.mean(kappa_vals))
#         metrics_results['f1_per_class'].append(np.mean(f1_per_class_vals, axis=0))

#         individual_values['acc'].append(acc_vals)
#         individual_values['mf1'].append(mf1_vals)
#         individual_values['kappa'].append(kappa_vals)
#         individual_values['f1_per_class'].append(f1_per_class_vals)
#     return metrics_results, individual_values,
# 读取所有被试的混淆矩阵，并计算指标，返回每个被试的分布值
def get_metrics_per_subject(data_path, subject, modes):
    metrics_results = {'acc': [], 'mf1': [], 'kappa': [], 'f1_per_class': []}
    individual_values = {'acc': [], 'mf1': [], 'kappa': [], 'f1_per_class': []}  # 存储每个被试的实际值
    confusion_res = [[0 for _ in range(5)] for _ in range(5)]
    for mode in modes:
        p = os.path.join(data_path, mode + '_' + subject + '.json')
        with open(p, 'r') as js:
            ckpt = json.load(js)

        acc_vals, mf1_vals, kappa_vals, f1_per_class_vals = [], [], [], []
        for subject_name, confusion_matrix in ckpt.items():
            confusion_matrix = np.array(confusion_matrix)
            confusion_res += confusion_matrix
            acc, mf1, kappa, f1_per_class = compute_metrics(confusion_matrix)
            acc_vals.append(acc)
            mf1_vals.append(mf1)
            kappa_vals.append(kappa)
            f1_per_class_vals.append(f1_per_class)
        f1_per_class_vals = np.array(f1_per_class_vals)
        metrics_results['acc'].append(np.mean(acc_vals))
        metrics_results['mf1'].append(np.mean(mf1_vals))
        metrics_results['kappa'].append(np.mean(kappa_vals))
        metrics_results['f1_per_class'].append(np.mean(f1_per_class_vals, axis=0))

        individual_values['acc'].append(acc_vals)
        individual_values['mf1'].append(mf1_vals)
        individual_values['kappa'].append(kappa_vals)
        individual_values['f1_per_class'].append(f1_per_class_vals)
    return metrics_results, individual_values,


def plot_per_subject(metrics_per_subject, individual_values, subject, modes, start, end):
    metrics = ['acc', 'mf1', 'kappa']  # 要绘制的metrics
    metric_names = ['Accuracy', 'Macro F1', 'Kappa']  # 用于显示的名称
    width = 0.2  # 柱状图的宽度

    # 假设误差条是每个模式下各个被试的标准差，可以根据你的需求调整
    error_bars = {metric: [np.std(individual_values[metric][i]) for i in range(len(modes))] for metric in metrics}
    circle_size=50
    # 为每个metric创建一个单独的figure
    for metric_idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(4, 6))  # 创建单独的figure
        x = np.arange(len(modes))  # x轴位置

        # 绘制每个metric的柱状图
        for i, mode in enumerate(modes):
            ax.bar(x[i], metrics_per_subject[metric][i], width,
                   yerr=error_bars[metric][i], label=f'{mode}', capsize=5)

            # 添加小圆圈表示个体的分布
            for val in individual_values[metric][i]:  # 这里用每个被试的值绘制小圆圈
                jitter = (np.random.rand() - 0.5) * width / 2  # 添加一些随机抖动，避免重叠
                ax.scatter(x[i] + jitter, val, color='black', s=circle_size, alpha=0.7)

        # 设置figure的标题和标签
        ax.set_title(f'{metric_names[metric_idx]} Comparison for Subject {subject}')
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_ylabel(metric_names[metric_idx])
        ax.set_ylim(start, end)  # 设置y轴范围为0-1
        # 保存figure并显示
        plt.legend()
        plt.savefig(f'/Users/hwx_admin/Sleep/result/aug_portion_edf_res/portion_edf_bar/{metric}_box.svg')
        plt.show()
        
def plot_per_metrics(metrics_per_subject, individual_values, modes, metric, start=0, end=1, name='edf'):
    width = 0.4
    subjects = ['1', '2', '5', '12']
    error_bars = {metric: [[np.std(individual_values[metric][j, i]) for i in range(len(modes))] for j in range(len(subjects))] }
    # metrics, subjects, modes: 10, 5x5 or 10, 5x1
    circle_size=30
    # print(error_bars)
    # 为每个metric创建一个单独的figure
    fig, ax = plt.subplots(figsize=(6, 6))  # 创建单独的figure
    x = np.arange(len(subjects))  # x轴位置
    p_values = []
    # 绘制每个metric的柱状图
    for i, subject in enumerate(subjects):
        # 原始和增强的对比，绘制并排的柱状图
        ax.bar(x[i] - width / 2, metrics_per_subject[metric][0][i], width,
               yerr=error_bars[metric][i][0],
               label='Original' if i == 0 else "", color='b', capsize=5)
        ax.bar(x[i] + width / 2, metrics_per_subject[metric][1][i], width,
               yerr=error_bars[metric][i][1],
               label='Augmented' if i == 0 else "", color='g', capsize=5)
        p_value = ttest_ind(individual_values[metric][i][0], individual_values[metric][i][1], equal_var=False)

        print((f'{metric} subject: {subject}, p_value: {p_value}, {np.mean(individual_values[metric][i][0])-np.mean(individual_values[metric][i][1])}'))
        for val in individual_values[metric][i][0]:  # 这里用每个被试的值绘制小圆圈
            jitter = (np.random.rand() - 0.5) * width / 2
            ax.scatter(x[i] + jitter - width / 2, val, color='black', s=circle_size, alpha=0.7)
        for val in individual_values[metric][i][1]:  # 这里用每个被试的值绘制小圆圈
            jitter = (np.random.rand() - 0.5) * width / 2
            ax.scatter(x[i] + jitter + width / 2, val, color='black', s=circle_size, alpha=0.7)
    # 设置figure的标题和标签

    # ax.set_title(f'{metric} Comparison ')
    # ax.set_xticks(x)
    # ax.set_xticklabels(subjects)
    # ax.set_ylabel(metric)
    # ax.set_ylim(0, 1)  # 设置y轴范围为0-1
    # # 保存figure并显示
    # plt.legend()
    # plt.savefig(f'/Users/hwx_admin/Sleep/result/aug_portion_edf_res/portion_edf_bar/{metric}_{name}.svg')
    # plt.show()


def plot_per_metrics_box(metrics_per_subject, individual_values, metric):
    subjects = ['1', '2', '5', '12']
    circle_size = 30
    fig, ax = plt.subplots(figsize=(6, 6))  # 创建单独的figure
    x = np.arange(len(subjects))  # x轴位置
    width = 0.4  # 调整宽度以便于箱线图之间的间距
    # 准备数据以供boxplot绘制
    box_data_original = [individual_values[metric][i][0] for i in range(len(subjects))]  # 原始数据
    box_data_augmented = [individual_values[metric][i][1] for i in range(len(subjects))]  # 增强数据

    # 绘制每个subject的原始和增强数据的boxplot
    for i, subject in enumerate(subjects):
        # 原始数据boxplot
        bp_orig = ax.boxplot(box_data_original[i], positions=[x[i] - width / 2], widths=0.3, patch_artist=True,
                             boxprops=dict(facecolor='b', color='b'), medianprops=dict(color='black'))
        # 增强数据boxplot
        bp_aug = ax.boxplot(box_data_augmented[i], positions=[x[i] + width / 2], widths=0.3, patch_artist=True,
                            boxprops=dict(facecolor='g', color='g'), medianprops=dict(color='black'))

        # 绘制散点
        for val in box_data_original[i]:  # 原始数据的小圆圈
            jitter = (np.random.rand() - 0.5) * width / 2
            ax.scatter(x[i] + jitter - width / 2, val, color='black', s=circle_size, alpha=0.7)
        for val in box_data_augmented[i]:  # 增强数据的小圆圈
            jitter = (np.random.rand() - 0.5) * width / 2
            ax.scatter(x[i] + jitter + width / 2, val, color='black', s=circle_size, alpha=0.7)

    # 设置figure的标题和标签
    ax.set_title(f'{metric} Comparison ')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)  # 设置y轴范围为0-1

    # 添加图例
    # orig_patch = mlines.Line2D([], [], color='b', marker='s', linestyle='None', markersize=10, label='Original')
    # aug_patch = mlines.Line2D([], [], color='g', marker='s', linestyle='None', markersize=10, label='Augmented')
    # plt.legend(handles=[orig_patch, aug_patch])

    plt.savefig(f'/Users/hwx_admin/Sleep/result/aug_portion_edf_res/portion_edf_bar/{metric}_box.svg')

    plt.show()
def plot_1x8_individual_metrics(individual_values, subjects):
    """
    绘制8个折线图，显示每个subject的orig和aug的metric变化
    横坐标为1, 2, 5, 12的subject ID
    """
    metrics = ['acc', 'mf1', 'kappa']
    metric_names = ['Accuracy', 'Macro F1', 'Kappa']
    x_labels = [1, 2, 5, 12]  # x-axis is the subject IDs

    for metric_idx, metric in enumerate(metrics):
        fig, axes = plt.subplots(1, 8, figsize=(24, 6))  # Create 1x8 grid of subplots

        # Plot lines for orig and aug across all subjects
        orig_values = individual_values[metric][:, 0]  # Original values for all subjects
        aug_values = individual_values[metric][:, 1]  # Augmented values for all subjects

        # Plot orig and aug as separate lines
        for subj_idx, ax in enumerate(axes):
            ax.plot(x_labels, orig_values[:, subj_idx], marker='o', color='b', label='Original', linewidth=2)
            ax.plot(x_labels, aug_values[:, subj_idx], marker='o', color='g', label='Augmented', linewidth=2)

            # Add titles, labels, and legends
            ax.set_title(f'{metric_names[metric_idx]} Comparison (Original vs. Augmented)', fontsize=14)
            ax.set_xlabel('Subjects', fontsize=12)
            ax.set_ylabel(metric_names[metric_idx], fontsize=12)
            ax.set_xticks(x_labels)
            ax.set_ylim(0, 1)  # Set y-axis limits for consistency

            # Add a grid for better readability
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(f'{metric_names[metric_idx]} Comparison (Original vs. Augmented)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'../../result/portion_edf_bar/{metric_names[metric_idx]}_line_comparison.svg')
        plt.show()

# 主函数调用
if __name__ == "__main__":
    data_path = '/Users/hwx_admin/Sleep/result/aug_portion_edf_res'
    subject_names = ['1', '2', '5', '12']  # 被试名
    modes = ['orig', 'aug']
    # modes = ['', 'aug_']  # 模式（无数据增强和有数据增强）

    end = [0.7, 0.7, 0.9, 0.9]
    start = [0.0, 0.0, 0.4, 0.4]
    metrics = {'acc': [[], []], 'mf1': [[], []], 'kappa': [[], []], 'f1_per_class': [[], []]}
    individual= {'acc': [], 'mf1': [], 'kappa': [], 'f1_per_class': []}
    for i, subject in enumerate(subject_names):
        metrics_per_subject, individual_values = get_metrics_per_subject(data_path, subject, modes)
        # plot_per_subject(metrics_per_subject, individual_values, subject, modes, start[i], end[i])
        # plot_per_metrics_box(metrics_per_subject, individual_values, subject, modes, start[i], end[i])
        for metric in ['acc', 'mf1', 'kappa', 'f1_per_class']:
            metrics[metric][0].append(metrics_per_subject[metric][0])
            metrics[metric][1].append(metrics_per_subject[metric][1])
            individual[metric].append(np.array(individual_values[metric]))
    print(individual)
    # metrics, subjects, modes: 10, 5x5 or 10, 5x1
    # f1_score_perclass = individual['f1_per_class']
    # np.save('./result/f1_score',f1_score_perclass, allow_pickle='True')
    for metric in ['acc', 'mf1', 'kappa']:
        individual[metric] = np.array(individual[metric])
    #     # plot_per_metrics_box(metrics, individual,  metric)
        plot_per_metrics(metrics, individual,  modes, metric, name='edf')
            # plot_per_metrics(metrics, individual, modes, start, end, metric)
    # plot_1x8_individual_metrics(individual, subject_names)