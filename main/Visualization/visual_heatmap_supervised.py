import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 读取 Excel 文件，跳过前 11 行，并将第 12 行设置为列名
file_path = '/Users/hwx_admin/Desktop/演示文稿1/Paper/Table/supervised.xlsx'
df = pd.read_excel(file_path, skiprows=11)

# 删除前 11 列
df = df.iloc[:, 5:].reset_index(drop=True)

# 将第 12 行设置为列名
df.columns = df.iloc[0]
df = df[1:]

# 向前填充 NaN 值
df['Datasets'] = df['Datasets'].ffill()

# 将所有的 '-' 替换为 NaN
df.replace('-', np.nan, inplace=True)

# 转换为长格式并过滤需要的列
df_long = df.melt(id_vars=['Datasets', 'System'], value_vars=['ACC', 'MF1', 'k'], var_name='Metric', value_name='Score')

# 设置模型顺序
model_order = ['SleepGPT', 'XSleepNet2', 'XSleepNet1'] + [m for m in df_long['System'].unique() if
                                                          m not in ['SleepGPT', 'XSleepNet2', 'XSleepNet1']]

# 自定义颜色渐变，调整顺序
colors = ['#619DB8', '#AECDD7', '#E3EEEF', '#FAE7D9', '#F0B79A', '#C85D4D']  # 蓝色表示较小的值，红色表示较大的值
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# 分开绘制热图
metrics = ['ACC', 'MF1', 'k']

for metric in metrics:
    heatmap_data = df_long[df_long['Metric'] == metric].pivot(index='System', columns='Datasets', values='Score')

    # 调整模型顺序
    heatmap_data = heatmap_data.reindex(model_order)

    plt.figure(figsize=(20, 10))
    plt.imshow(np.ones_like(heatmap_data), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    # 添加矩形边框
    # for i in range(heatmap_data.shape[0]):
    #     for j in range(heatmap_data.shape[1]):
    #         plt.gca().add_patch(
    #             plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='black', facecolor='white', fill=True))

    # 添加圆形标记
    min_value = heatmap_data.values[~np.isnan(heatmap_data.values)].min()
    max_value = heatmap_data.values[~np.isnan(heatmap_data.values)].max()
    if min_value == max_value:
        norm = lambda x: 0.5  # 如果最小值和最大值相等，归一化为0.5
    else:
        norm = lambda x: (x - min_value) / (max_value - min_value)

    for x in range(heatmap_data.shape[1]):
        column_data = heatmap_data.iloc[:, x]
        ranked_sizes = column_data.rank(ascending=True)  # 使用rank进行排序，值越大，排名越高
        print(ranked_sizes)
        normalized_sizes = (ranked_sizes - ranked_sizes.min()) / (ranked_sizes.max() - ranked_sizes.min())
        for y in range(heatmap_data.shape[0]):
            value = heatmap_data.iloc[y, x]
            if not np.isnan(value):
                # 设置最小圆圈大小
                size = max(normalized_sizes.iloc[y] * 1000, 50)
                plt.scatter(x, y, s=size, color=cmap(norm(value)), edgecolors='black', marker='o')

    plt.title(f'Model Performance ({metric}) Across Different Datasets')
    plt.xlabel('Datasets')
    plt.ylabel('Models')
    plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)

    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label=metric)
    os.makedirs(f'../result/heatmap', exist_ok=True)
    plt.savefig(f'../result/heatmap/heatmap_supervised_{metric}.svg')
    plt.show()
    plt.close()
