# 设置绘图风格
import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
channel_colors3 = {
    'C3': '#9db4ce',
    'C4': '#F9C08a',
    'EMG': '#eda1a4',
    'EOG': '#b3d8d5',
    'F3': '#a4cb9e',
    'Fpz': '#f7d08d',
    'O1': '#bf83a5',
    'Pz': '#8684b0'
}
channel_colors2 = {
    'C3': '#a85399',
    'C4': '#e7481b',
    'EMG': '#d0a026',
    'EOG': '#a82324',
    'F3': '#62beb8',
    'Fpz': '#5581c1',
    'O1': '#664496',
    'Pz': '#098584'
}
channel_colors = {
    'C3': '#ADAFB1',
    'C4': '#D8D9DA',
    'EMG': '#ecb884',
    'EOG': '#e4e45f',
    'F3': '#4758A2',
    'Fpz': '#E08D8B',
    'O1': '#AF8CBB',
    'Pz': '#AAD7C8'
}
channel_map = {
    0: 'C3',
    1: 'C4',
    2: 'EMG',
    3: 'EOG',
    4: 'F3',
    5: 'Fpz',
    6: 'O1',
    7: 'Pz'
}
channels_to_display = np.array([[ 'EOG', 'Fpz', 'Pz'],
                       ['C3', 'C4', 'EOG', 'F3', 'O1', 'Pz'],
                       ['C3', 'C4', 'EOG', 'F3', 'Fpz', 'O1', 'Pz'],
                       ['C3', 'C4', 'EOG', 'F3', 'O1', 'Pz'],
                       ['C3', 'C4', 'EOG', 'O1'],
                       ['C3', 'C4', 'EOG', 'F3', 'O1', 'Pz'],
                       ['C3', 'C4','EOG'],
                       ['C3', 'C4',  'EOG', 'F3', 'O1']])
def prepare_data(data):
    rows = []
    for dataset_name in sorted(data):
        losses = data[dataset_name]
        for loss_name, tensors in losses.items():

            channel_data = {i: [] for i in range(8)}
            for tensor in tensors:
                for channel, value in enumerate(tensor.numpy()):
                    channel_data[channel].append(value)

            for channel, values in channel_data.items():
                if not any(np.isnan(values)):
                    Q1 = pd.Series(values).quantile(0.25)
                    Q3 = pd.Series(values).quantile(0.75)
                    IQR = Q3 - Q1
                    filtered_values = [v for v in values if (v >= Q1 - 1.5 * IQR) and (v <= Q3 + 1.5 * IQR)]
                else:
                    filtered_values = [0]
                if filtered_values:
                    for value in filtered_values:
                        rows.append((dataset_name, loss_name, channel, value))
                else:
                    rows.append((dataset_name, loss_name, channel, 0))

    df = pd.DataFrame(rows, columns=['Dataset', 'Loss Type', 'Channel', 'Value'])
    # df['Channel'] = pd.Categorical(df['Channel'], categories=sorted(df['Channel'].unique()), ordered=True)
    return df
def plot_box_emg(df):
    # 设置更好的风格
    colors = {'C3': '#a3a5a6',
        'C4': '#a3a5a6',
        'EMG': '#e31b1e',
        'EOG': '#a3a5a6',
        'F3': '#a3a5a6',
        'Fpz': '#a3a5a6',
        'O1': '#a3a5a6',
        'Pz': '#a3a5a6'
    }
    sns.set(style="whitegrid")
    unique_loss_types = df['Loss Type'].unique()

    for loss_type in sorted(unique_loss_types):
        df_summary = df[df['Loss Type'] == loss_type]
        # 剔除Value为0的行
        filtered_df = df_summary[df_summary['Value'] != 0]

        # 按Channel分组，并计算每个Channel的平均Value
        average_values = filtered_df.groupby('Channel')['Value'].mean()
        print(average_values)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Channel', y='Value', data=df_summary, whis=1.5, linewidth=1.5, fliersize=5, palette=colors)
        plt.savefig(f'/Users/hwx_admin/Sleep/result/reconstruction/emg_{loss_type}_loss.svg')
        plt.show()

def plot_bar_emg(df):
    unique_loss_types = df['Loss Type'].unique()

    for loss_type in sorted(unique_loss_types):
        df_summary = df[df['Loss Type'] == loss_type].groupby('Channel').agg({'Value': ['mean', 'std']}).reset_index()

        df_summary.columns = ['Channel', 'Mean', 'StdDev']
        # 设置更好的风格
        sns.set(style="whitegrid")

        colors = ['skyblue' if channel != 'EMG' else 'orange' for channel in df_summary['Channel']]
        means = df_summary['Mean'].values
        stds = df_summary['StdDev'].values
        sns.barplot(x='Channel', y='Mean', data=df_summary, capsize=0.2, palette=colors)
        plt.errorbar(df_summary['Channel'], means, yerr=stds, fmt='none', c='k', capsize=5, alpha=0.5)

        plt.title('Comparison of Channels with Mean and Standard Deviation')
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Mean', fontsize=12)

        # 移除图例
        plt.legend([], [], frameon=False)

        # 更好的y轴刻度
        plt.ylim(0, max(means + stds) * 1.2)  # 增加一些空间，以免误差线超出图表顶端

        # 移除上边框和右边框
        sns.despine()
        plt.show()
def plot_violin(df):
    unique_datasets = df['Dataset'].unique()
    unique_loss_types = df['Loss Type'].unique()

    for idx, dataset in enumerate(sorted(unique_datasets)):
        for loss_type in sorted(unique_loss_types):
            plt.figure(figsize=(12, 6))
            subset_df = df[
                (df['Loss Type'] == loss_type) & (df['Channel'] != channel_map[2]) & (df['Dataset'] == dataset) & (
                    df['Channel'].isin(channels_to_display[idx]))]
            subset_df = subset_df[subset_df['Loss Type'] == loss_type].sort_values(by='Channel')
            ax = sns.violinplot(x='Channel', y='Value', data=subset_df, inner="box", palette=channel_colors3, order=sorted(df[df['Channel'] != channel_map[2]]['Channel'].unique()))
            ax.set_xlabel('')
            ax.set_ylabel('')
            mean_values = subset_df.groupby('Channel')['Value'].mean().reset_index()
            mean_values = mean_values[mean_values['Channel'] != channel_map[2]]
            valid_means = mean_values[mean_values['Value'] != 0]
            if not valid_means.empty:
                plt.scatter(x=valid_means['Channel'], y=valid_means['Value'], color='r', s=10, label='Mean', zorder=5,
                            marker='o', alpha=0.7)
            # 设置x轴的所有通道
            print(valid_means)
            all_channels = ['C3', 'C4', 'EOG', 'F3', 'Fpz', 'O1', 'Pz']
            ax.set_xticks(range(len(all_channels)))  # 设置x轴刻度
            ax.set_xticklabels(all_channels)  # 设置x轴刻度标签
            # 设置x轴的范围以匹配所有通道的布局
            ax.set_xlim(-0.5, len(all_channels) - 0.5)  # 设置x轴范围
            plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.7, axis='y')
            plt.savefig(f'/Users/hwx_admin/Sleep/result/reconstruction/{loss_type} - {dataset}.svg')
            plt.show()


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df_clean


def main(ckpt_path='../../temp_log/violin.ckpt'):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    df = prepare_data(checkpoint)
    df.loc[(df['Dataset'] == 'EDF') & (df['Channel'] == 2), 'Value'] = 0
    df['Channel'] = df['Channel'].map(channel_map)
    print(df.head(20))
    plot_violin(df, )
    # plot_bar_emg(df)
    # plot_box_emg(df)


if __name__ == '__main__':
    main()
