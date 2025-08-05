import os

import matplotlib.pyplot as plt
import numpy as np

def bar_supervised():
    datasets_supervised = ['SleepEDF-20', 'SHHS', 'MASS', 'Physio2018']
    models_supervised = ['SGMPT', 'XSleepNet', 'DeepSleepNet']

    supervised_acc = {
        'SGMPT': [87.8, 91.3, 88.6, 81.1],
        'XSleepNet': [86.4, 89.1, 87.5, 81.1],
        'DeepSleepNet': [85.2, 87.2, 85.2, 80.5]  # 修改为准确数据
    }

    supervised_f1 = {
        'SGMPT': [81.6, 84.5, 84.1, 79.3],
        'XSleepNet': [80.5, 82.2, 83.2, 79.5],
        'DeepSleepNet': [78.0, 80.5, 82.0, 78.3]  # 修改为准确数据
    }

    x = np.arange(len(datasets_supervised))
    width = 0.2

    # 创建有监督学习图表
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 定义颜色
    colors_acc = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_f1 = ['#aec7e8', '#ffbb78', '#98df8a']

    # 绘制每个模型的柱状图
    for i, model in enumerate(models_supervised):
        ax1.bar(x + (i - 1) * width, supervised_acc[model], width, label=f'{model} ACC', color=colors_acc[i])
        ax1.bar(x + (i - 1) * width, supervised_f1[model], width, label=f'{model} F1', color=colors_f1[i], alpha=0.7)

    # 添加标签和标题
    ax1.set_xlabel('Datasets')
    ax1.set_ylabel('Accuracy/F1 Score')
    ax1.set_title('Supervised Learning Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_supervised)
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1), title='Legend')

    # 显示图表
    plt.tight_layout()
    plt.show()

def radar():
    import matplotlib.pyplot as plt
    import numpy as np

    # 数据准备
    labels = ['SleepEDF-20', 'SHHS', 'MASS', 'Physio2018', 'SleepEDF-78']
    num_vars = len(labels)

    # 有监督学习的准确率
    supervised_acc = {
        'SGMPT': [87.8, 89.1, 88.6, 81.11, 84.7],
        'XSleepNet': [86.4, 89.1, 87.6, 81.1, 84],
        'SeqSleepNet': [85.6, 88.4, 87.0, 79.2, 83.8]
    }

    # 有监督学习的F1分数
    supervised_f1 = {
        'SGMPT': [81.6, 82.4, 85.1, 79.3, 77.5],
        'XSleepNet': [80.9, 82.3, 83.8, 79.5, 78.7],
        'SeqSleepNet': [78.6, 80.1, 83.3, 79.2, 78.2]
    }
    # 有监督学习的Kappa Score
    supervised_kappa = {
        'SGMPT': [83.4, 84.5, 83.7, 74.2, 78.3],
        'XSleepNet': [81.3, 84.7, 82.3, 74.2, 78.7],
        'SeqSleepNet': [80.3, 83.8, 81.5, 73.7, 77.6]
    }

    # 角度计算
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 创建雷达图
    fig, axs = plt.subplots(1, 3, figsize=(18, 9), subplot_kw=dict(polar=True))

    # 定义颜色
    colors = ['#FF6347', '#4682B4', '#32CD32']

    # 绘制每个模型的雷达图 - 准确率
    def add_to_radar(ax, data, label, color):
        values = data + data[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.25)

    for idx, model in enumerate(supervised_acc.keys()):
        add_to_radar(axs[0], supervised_acc[model], f'{model} ACC', colors[idx])
        add_to_radar(axs[1], supervised_f1[model], f'{model} F1', colors[idx])
        add_to_radar(axs[2], supervised_kappa[model], f'{model} Kappa', colors[idx])
    # 设置显示范围和网格
    for ax, metric in zip(axs, ['Accuracy', 'F1 Score', 'Kappa']):
        ax.set_title(f'Supervised Learning {metric} Comparison', size=20, color='k', y=1.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=15)
        ax.tick_params(pad=10)
        ax.set_ylim(70, 90)  # 设置显示范围
        ax.yaxis.set_ticks(np.arange(70, 90, 5))

    axs[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Models')
    axs[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Models')
    axs[2].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Models')
    # 显示图表
    plt.tight_layout()
    os.makedirs('../result/radar/', exist_ok=True)
    plt.savefig('../result/radar/supervised.svg')
    plt.show()
def bar_un():
    import matplotlib.pyplot as plt
    import numpy as np

    # 数据准备
    groups_sleepedf_78 = [
        ('TS-TCC', 'neuro2vec', 'GMPT1'),
        ('ContraWR', 'GMPT2'),
        ('MulEGG', 'GMPT3')
    ]

    models_sleepedf_20 = ['CPC', 'Ts2VEC', 'TST', 'FEAT', 'C-MAE', 'TS-TCC', 'MTS-LOF', 'GMPT']

    # SleepEDF-78 数据集
    acc_sleepedf_78 = {
        'TS-TCC': 84.05, 'neuro2vec': 86.53, 'GMPT1': 87.35, 'ContraWR': 86.90, 'GMPT2': 87.29, 'MulEGG': 78.06,
        'GMPT3': 84.34
    }
    f1_sleepedf_78 = {
        'TS-TCC': 77.04, 'neuro2vec': 78.94, 'GMPT1': 81.6, 'ContraWR': np.nan, 'GMPT2': 77.91, 'MulEGG': 67.82,
        'GMPT3': 76.61
    }

    # SleepEDF-20 数据集
    acc_sleepedf_20 = {
        'CPC': 82.82, 'Ts2VEC': 83.33, 'TST': 83.43, 'FEAT': 81.76, 'C-MAE': 82.11, 'TS-TCC': 83.00, 'MTS-LOF': 84.35,
        'GMPT': 85.93
    }
    f1_sleepedf_20 = {
        'CPC': 73.94, 'Ts2VEC': 73.23, 'TST': 73.23, 'FEAT': 72.58, 'C-MAE': 71.86, 'TS-TCC': 73.57, 'MTS-LOF': 73.52,
        'GMPT': 80.6
    }

    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))

    # 通用颜色设置
    acc_color = '#1f77b4'
    f1_color = '#ff7f0e'
    gmpt_color = '#2ca02c'

    # SleepEDF-78 的 Group 1
    ax = axs[0, 0]
    labels = ['TS-TCC', 'neuro2vec', 'GMPT']
    acc_values = [acc_sleepedf_78['TS-TCC'], acc_sleepedf_78['neuro2vec'], acc_sleepedf_78['GMPT1']]
    f1_values = [f1_sleepedf_78['TS-TCC'], f1_sleepedf_78['neuro2vec'], f1_sleepedf_78['GMPT1']]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, acc_values, width, color=[acc_color, acc_color, gmpt_color], label='ACC')
    bars2 = ax.bar(x + width / 2, f1_values, width, color=[f1_color, f1_color, gmpt_color], label='MF1')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('SleepEDF-78 Group 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(70, 90)

    # 在条形图上添加数值标签
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # SleepEDF-78 的 Group 2
    ax = axs[0, 1]
    labels = ['ContraWR', 'GMPT']
    acc_values = [acc_sleepedf_78['ContraWR'], acc_sleepedf_78['GMPT2']]
    f1_values = [f1_sleepedf_78['ContraWR'], f1_sleepedf_78['GMPT2']]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, acc_values, width, color=[acc_color, gmpt_color], label='ACC')
    bars2 = ax.bar(x + width / 2, f1_values, width, color=[f1_color, gmpt_color], label='MF1')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('SleepEDF-78 Group 2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(70, 90)

    # 在条形图上添加数值标签
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # SleepEDF-78 的 Group 3
    ax = axs[1, 0]
    labels = ['MulEGG', 'GMPT']
    acc_values = [acc_sleepedf_78['MulEGG'], acc_sleepedf_78['GMPT3']]
    f1_values = [f1_sleepedf_78['MulEGG'], f1_sleepedf_78['GMPT3']]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, acc_values, width, color=[acc_color, gmpt_color], label='ACC')
    bars2 = ax.bar(x + width / 2, f1_values, width, color=[f1_color, gmpt_color], label='MF1')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('SleepEDF-78 Group 3')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(70, 90)

    # 在条形图上添加数值标签
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # SleepEDF-20
    ax = axs[1, 1]
    labels = ['CPC', 'Ts2VEC', 'TST', 'FEAT', 'C-MAE', 'TS-TCC', 'MTS-LOF', 'GMPT']
    acc_values = [acc_sleepedf_20[model] for model in models_sleepedf_20]
    f1_values = [f1_sleepedf_20[model] for model in models_sleepedf_20]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, acc_values, width, color=[acc_color] * (len(labels) - 1) + [gmpt_color], label='ACC')
    bars2 = ax.bar(x + width / 2, f1_values, width, color=[f1_color] * (len(labels) - 1) + [gmpt_color], label='MF1')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('SleepEDF-20')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(70, 90)

    # 在条形图上添加数值标签
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # 自定义图例
    custom_lines = [plt.Line2D([0], [0], color=acc_color, lw=4),
                    plt.Line2D([0], [0], color=f1_color, lw=4),
                    plt.Line2D([0], [0], color=gmpt_color, lw=4)]

    fig.legend(custom_lines, ['ACC', 'MF1', 'GMPT'], loc='upper center', ncol=3)

    # 显示图表
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
if __name__ == '__main__':
    bar_un()