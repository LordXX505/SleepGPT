import numpy as np
import matplotlib.pyplot as plt
import re
def expert2_():
    import matplotlib.pyplot as plt
    import numpy as np

    # 字符串转浮点数列表函数
    def string_to_float_list(data_string):
        return list(map(float, data_string.split()))

    s = "0.795833 0.68088 0.6319 0.66666 0.5117 0.72801 0.787954 0.76959 0.5705 0.72713 0.7038 0.6733 0.71414 0.74216 0.6624"
    s1 = "0.8096 0.8179 0.7355 0.6822 0.6194 0.81223 0.8284 0.859 0.7209 0.7927 0.8049 0.7852 0.7809 0.7466 0.74947"

    # 转换为浮点数列表
    data = string_to_float_list(s)
    data1 = string_to_float_list(s1)
    full_data = np.full(19, np.nan)  # 初始化19个空值
    full_data1 = np.full(19, np.nan)  # 初始化19个空值

    indices = list(range(19))  # 人的索引
    for empty in [3, 7, 14, 15]:  # 索引从0开始，所以减1
        indices.remove(empty)
    # 计算平均值
    average_s = np.nanmean(data)
    average_s1 = np.nanmean(data1)

    # 绘制柱状图
    x = np.arange(19) + 1
    width = 0.2  # 柱体宽度，稍微缩小以增加间隔
    for i, idx in enumerate(indices):
        full_data[idx] = data[i]
        full_data1[idx] = data1[i]
    fig, ax = plt.subplots()
    print(full_data)
    ax.bar(x - width/2, full_data, width, label='s',  color='#df7967')
    ax.bar(x + width/2, full_data1, width, label='s1',  color='#efd1dd')
    # 在最后一列添加平均值柱形图，增加宽度并高亮显示
    ax.bar(len(full_data) + 4 * width, average_s, width * 2, color='#ef6547', label='Avg s', edgecolor='black',
           linewidth=1.5)
    ax.bar(len(full_data) + 6 * width, average_s1, width * 2, color='#e1a3c6', label='Avg s1', edgecolor='black',
           linewidth=1.5)

    # 设置图表的标题和标签
    ax.set_xlabel('Subjects')
    ax.set_ylabel('F1 Score')
    ax.set_xticks(np.append(x, [len(full_data) + 5 * width]))  # 设置x轴标签的位置
    ax.set_xticklabels([f'{i}' for i in range(1, len(full_data) + 1)] + ['Avg'])
    ax.set_ylim([0.5, 0.9])
    plt.savefig('/Users/hwx_admin/Sleep/result/vspindle/exp.svg')
    # 显示图形
    plt.show()


def expert1():

    def string_to_float_list(data_string):
        return list(map(float, data_string.split()))

    # 字符串数据
    s = "0.74642	0.668	0.608695	0.6507	0.6666	0.5449	0.7041	0.73207	0.77259	0.778337	0.638772	0.736	0.676	0.68974	0.58974	0.6558	0.718	0.76907	0.6786"
    s1 = "0.78233	0.6911	0.64705	0.64954	0.655601	0.5614	0.74061	0.72564	0.78545	0.7661	0.6162	0.7273	0.66264	0.7212	0.58536	0.6628	0.71457	0.7539	0.66762"

    # 转换为浮点数列表
    data = string_to_float_list(s)
    data1 = string_to_float_list(s1)

    # 计算平均值
    average_s = np.mean(data)
    average_s1 = np.mean(data1)

    x = np.arange(19) + 1

    width = 0.2 # 柱体宽度

    fig, ax = plt.subplots()
    ax.bar(x - width, data, width, label='s', color='#df7967')
    ax.bar(x, data1, width, label='s1', color='#efd1dd')
    # 在最后一列添加平均值柱形图，增加宽度并高亮显示
    ax.bar(len(data) + 4 * width, average_s, width * 2, color='#ef6547', label='Avg s', edgecolor='black',
           linewidth=1.5)
    ax.bar(len(data) + 6 * width, average_s1, width * 2, color='#e1a3c6', label='Avg s1', edgecolor='black',
           linewidth=1.5)

    ax.set_xlabel('Subjects')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0.5, 0.9])
    ax.set_title('Comparison of F1 Scores and Averages')
    ax.set_xticks(np.append(x, [len(data) + 5 * width]))  # 设置x轴标签的位置
    ax.set_xticklabels([f'{i}' for i in range(1, len(data) + 1)] + ['Avg'])
    plt.savefig('/Users/hwx_admin/Sleep/result/vspindle/exp1.svg')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    expert1()