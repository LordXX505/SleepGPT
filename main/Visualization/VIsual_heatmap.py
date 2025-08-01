import copy

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import tensor
import pandas as pd
import seaborn as sns
from math import pi
from scipy.stats import ttest_ind, chi2_contingency

cmap = LinearSegmentedColormap.from_list("mycmap",
                                         [(0, "#282A62"), (0.2, "#692F7C"), (0.4, "#B43970"),
                                          (0.6, "#d96558"), (0.8, "#efa143"), (1, "#F6C63C")])
cmap2 = LinearSegmentedColormap.from_list("mycmap",
                                         [(0, "#9ec1d4"), (0.2, "#ddf1f3"), (0.4, "#ecf4dd"),
                                          (0.6, "#fff7ac"), (0.8, "#ecb477"), (1, "#e87651")])
def confusion(cm: torch.Tensor):
    sum0 = cm.sum(axis=0)
    sum1 = cm.sum(axis=1)
    all_sum = cm.sum()
    p0 = torch.diag(cm).sum() / all_sum
    FP = sum0 - torch.diag(cm)
    FN = sum1 - torch.diag(cm)
    TP = torch.diag(cm)
    acc = TP.sum()/all_sum

    TN = all_sum - FP - FN - TP

    precision = TP / (TP + FP + 1e-6)

    recall = TP / (TP + FN+ 1e-6)

    pe = (sum0 * sum1).sum() / (all_sum ** 2)

    kappa = (p0 - pe) / (1 - pe)
    sensitivity = TP/ (TP + FN)
    specificity = TN / (FP + TN)
    macro_f1_sub = 2 * precision * recall / (precision + recall+ 1e-6)
    macro_f1 = torch.mean(macro_f1_sub, dim=-1)
    return precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub
def main():
    root_path = '/Users/hwx_admin/Sleep/DATA_result/result'
    dataset_path = glob.glob(os.path.join(root_path, '*'))
    ds_res = {}

    # F6C63C
    # efa143 # d96558 #B43970 #692F7C #282A62
    vmin, vmax = 100, 0
    for dataset in dataset_path:
        ds_res[os.path.basename(dataset)] = {}
        ckpt_path = glob.glob(os.path.join(dataset, 'attn/*'))
        res = {0: torch.zeros(8), 1: torch.zeros(8), 2: torch.zeros(8), 3: torch.zeros(8), 4: torch.zeros(8)}
        for ckpts in ckpt_path:
            ckpt = torch.load(os.path.join(ckpts, 'res.ckpt'), map_location='cpu')
            for k, v in ckpt.items():
                res[k] += v
        ds_res[os.path.basename(dataset)] = res
    heatmap_data_list = []
    for dataset_name in [1, 2, 3, 5]:
        data = ds_res[f'MASS{dataset_name}']
        # 将所有tensor转换为NumPy数组
        heatmap_data = np.array([data[i].numpy() for i in data.keys()], dtype=np.float32)
        sum = np.array([heatmap_data[i].sum() for i in range(heatmap_data.shape[0])]).reshape(-1, 1)
        heatmap_data = heatmap_data/sum  # 20 fold
        heatmap_data_list.append(heatmap_data)
        vmin = min(vmin, heatmap_data.min())
        vmax = max(vmax, heatmap_data.max())
    print(vmin, vmax)
    for i, dataset_name in enumerate([1, 2, 3, 5]):
        heatmap_data = heatmap_data_list[i]
        # 使用Matplotlib创建热图
        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(heatmap_data, cmap=cmap2, vmin=vmin, vmax=vmax)
        # plt.colorbar()  # 显示颜色条
        # plt.title('Heatmap of EEG Electrodes by Sleep Stage')
        plt.xlabel('Electrode Channels')
        plt.ylabel('Sleep Stage Label')
        electrode_channels = ['C3', 'C4', 'EOG', 'EMG', 'F3', 'Fpz', 'O1', 'Pz']
        sleep_stages = ['W', 'Stage 1', 'Stage 2', 'Stage 3', 'REM']
        ax.set_xticks(np.arange(len(electrode_channels)) + 0.5)
        ax.set_yticks(np.arange(len(sleep_stages)) + 0.5)
        ax.set_xticklabels(electrode_channels, rotation=0, ha="center")  # 水平居中对齐
        ax.set_yticklabels(sleep_stages, rotation=0, va="center")  # 垂直居中对齐
        plt.subplots_adjust(bottom=0.15, top=0.85)
        plt.savefig(f'/Users/hwx_admin/Sleep/result/heatmap/{dataset_name}_heatmap.svg')
        plt.show()

def visual_radar_f1():
    mass = [{0: None, 1: None} for _ in range(6)]
    mass[3][0] = tensor([[1964, 156, 64, 4, 137],
                         [105, 473, 484, 2, 505],
                         [33, 103, 8043, 360, 238],
                         [1, 0, 407, 2124, 6],
                         [11, 93, 151, 0, 2816]])
    mass[3][1] = tensor([[2064, 114, 44, 3, 100],
                         [77, 503, 500, 0, 489],
                         [49, 98, 8047, 401, 182],
                         [3, 0, 545, 1990, 0],
                         [17, 72, 185, 2, 2795]])
    mass[5][0] = tensor([[978, 3, 104, 8, 91],
                         [125, 18, 233, 0, 319],
                         [22, 3, 5343, 136, 95],
                         [0, 0, 307, 1562, 0],
                         [23, 0, 123, 0, 2307]])
    mass[5][1] = tensor([[1001, 49, 65, 7, 62],
                         [65, 89, 210, 0, 331],
                         [26, 26, 5337, 93, 117],
                         [4, 0, 399, 1466, 0],
                         [28, 4, 172, 1, 2248]])
    mass[1][0] = tensor([[2744,   38,  256,    0,   41],
                    [ 666,   49, 1217,    0,  309],
                    [ 136,   23, 6950,    0,  167],
                    [   0,    0, 1081,    0,    0],
                    [ 419,   20,  500,    0,  904]])
    mass[1][1] = tensor([[2917,  102,   14,    0,   46],
                            [413,  594,  784,    1,  449],
                            [67,  337, 6455,  186,  231],
                            [1,    0,  455,  625,    0],
                            [172,   77,  193,    0, 1401]])
    mass[2][1] = tensor([[388, 0, 29, 4, 464],
                         [6, 0, 103, 1, 346],
                         [9, 0, 3398, 436, 260],
                         [1, 0, 103, 1212, 0],
                         [7, 0, 99, 0, 1614]])
    mass[2][0] = tensor([[0, 0, 191, 9, 685],
                         [0, 0, 88, 1, 367],
                         [0, 0, 3039, 637, 427],
                         [0, 0, 90, 1226, 0],
                         [0, 0, 125, 1, 1594]])
    res = []
    plt.style.use('ggplot')

    res_overall = [mass[1][0].detach().clone(), mass[1][1].detach().clone()]
    for i in [2, 3, 5]:
        res_overall[0] += mass[i][0]
        res_overall[1] += mass[i][1]
    ans = [0 for _ in range(2)]
    overall_metric = [0, 0]
    for i in range(2):
        recision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub = confusion(res_overall[i])
        ans[i] = macro_f1_sub
        overall_metric[i] = (kappa, acc, macro_f1)
    print(f'ans: {ans}')
    overall_metric = [0, 0]  # 存储两个模型的整体指标
    overall_kappa, overall_acc, overall_macro_f1 = [[], []], [[], []], [[], []]  # 分别存储每个数据集的指标

    # 遍历数据集，分别计算每个数据集的 kappa, acc, macro_f1
    for i in [1, 2, 3, 5]:
        for model in range(2):  # 模型 0 和模型 1
            recision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub = confusion(mass[i][model])
            overall_kappa[model].append(kappa.item())
            overall_acc[model].append(acc.item())
            overall_macro_f1[model].append(macro_f1.item())

    # 计算每个模型的平均值
    for model in range(2):
        overall_metric[model] = (
            np.mean(overall_kappa[model]),
            np.mean(overall_acc[model]),
            np.mean(overall_macro_f1[model])
        )
    stat, p, dof, expected = chi2_contingency(res_overall)

    print(f"overall_metric: {overall_metric}")
    # 计算 p-value
    kappa_p_value = ttest_ind(overall_kappa[0], overall_kappa[1], equal_var=False).pvalue
    acc_p_value = ttest_ind(overall_acc[0], overall_acc[1], equal_var=False).pvalue
    macro_f1_p_value = ttest_ind(overall_macro_f1[0], overall_macro_f1[1], equal_var=False).pvalue
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(overall_acc[0], overall_acc[1])
    print(kappa_p_value, acc_p_value, macro_f1_p_value)
    max_per_class = np.max(np.stack([ans[0], ans[1]]), axis=0)
    ans[0] = np.where(max_per_class != 0, ans[0] / max_per_class, 1)
    ans[1] = np.where(max_per_class != 0, ans[1] / max_per_class, 1)
    for index in [1, 2, 3, 5]:
        res_temp = {}
        precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub = confusion(mass[index][0])
        res_temp[0] = macro_f1_sub.numpy()
        precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub = confusion(mass[index][1])
        res_temp[1] = macro_f1_sub.numpy()
        print(f"index: {index}  res_temp: {res_temp}")
        max_per_class = np.max(np.stack([res_temp[0], res_temp[1]]), axis=0)
        res_temp[0] = np.where(max_per_class!=0, res_temp[0]/max_per_class, 0)
        res_temp[1] = np.where(max_per_class!=0, res_temp[1]/max_per_class, 0)
        res.append(res_temp)
    print(res)
    res.append({0: ans[0], 1: ans[1]})

    df = pd.DataFrame(res)
    df.index = ['1', '2', '3', '4', '5']
    num_vars = 5
    labels = ['W', 'N1', 'N2', 'N3', 'REM']  # 类别标签

    # 创建角度，这里确保图形是闭合的
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    for i, row in df.iterrows():
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        values = row[0]  # Model 0
        values1 = row[1]  # Model 1
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
        plt.savefig(f'/Users/hwx_admin/Sleep/result/heatmap/{i}.svg')
        plt.show()
    categories = ["k", "ACC", "MF1"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width / 2, overall_metric[0], width, label='Model 1', color='#4DBADB', alpha=0.75)
    rects2 = ax.barh(x + width / 2, overall_metric[1], width, label='Model 2', color='#E44A33', alpha=0.75)
    # for rect in rects1 + rects2:
    #     width = rect.get_width()
    #     ax.text(width, rect.get_y() + rect.get_height() / 2,
    #             f'{width:.3f}', ha='left', va='center')
    ax.set_yticks(x)
    ax.set_xlim([0.55, 0.85])
    ax.set_yticklabels(categories)
    plt.savefig(f'/Users/hwx_admin/Sleep/result/heatmap/overall_metric.svg')

    plt.show()

#plt
if __name__ == '__main__':
    # main()
    visual_radar_f1()