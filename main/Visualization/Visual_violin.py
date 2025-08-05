import os
import torch
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
channel_name = ["C3", "C4", "EMG", "EOG", "F3", "Fpz", "O1", "Pz"]
def get_all_dataset_reconstruction_result(root_path):
    dataset_list_path = os.listdir(root_path)
    mode = ['visual_mask_same', 'visual_mask_no_fft', 'visual_all']
    res = {}
    res_name = {}
    for dst_path in dataset_list_path:
        if dst_path in mode:
            if dst_path not in res:
                res[dst_path] = {}
                res_name[dst_path] = {}
            for items in glob.glob(os.path.join(root_path, dst_path, '*/*/*')):
                ckpt = torch.load(items, map_location='cpu')
                dst_name = items.split('/')[-3]
                if dst_name not in res[dst_path]:
                    res[dst_path][dst_name] = {'loss': [], 'loss2': []}
                    res_name[dst_path][dst_name] = []
                for n, v in ckpt.items():
                    res[dst_path][dst_name]['loss'].append(v['loss1'])
                    res[dst_path][dst_name]['loss2'].append(v['loss2'])
                    res_name[dst_path][dst_name].append(n)
    return res, res_name
# 处理数据，创建DataFrame
def prepare_data(data, loss_type):
    all_data = []
    for dataset_name, losses in data.items():
        for tensor in losses[loss_type]:
            for channel, value in enumerate(tensor.numpy()):
                all_data.append((dataset_name, channel, value))
    df = pd.DataFrame(all_data, columns=['Dataset', 'Channel', 'Value'])
    df['Channel'] = df['Channel'].astype('category')  # Ensure that Channel is a categorical type
    return df
# 示例函数来清除离群点
def remove_outliers_z(df, column):
    df['z_score'] = stats.zscore(df[column])
    # 保留Z-Score在正负3之内的数据
    df_clean = df[df['z_score'].abs() <= 3]
    df_clean = df_clean.drop(columns=['z_score'])
    return df_clean
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df_clean



def create_separate_violin_plots(df):
    # 获取通道列表
    channels = df['Channel'].unique()

    for loss_name in ['loss', 'loss2']:
        for channel in channels:
            channel_df = df[df['Channel'] == channel]
            plt.figure(figsize=(12, 6))
            sns.violinplot(x='Dataset', y=loss_name, data=channel_df)
            plt.title(f'Violin Plot for Channel {channel}')
            plt.xlabel('Dataset')
            plt.ylabel('Value')

            mean_values = channel_df.groupby('Dataset')[loss_name].mean().values
            print(mean_values)
            plt.scatter(range(len(mean_values)), mean_values, color='red', zorder=3)
            plt.show()
def FacetGris(df):
    df['Dataset_Channel'] = df['Dataset'] + ' - ' + df['Channel']
    for loss_name in ['loss', 'loss2']:
        g = sns.FacetGrid(df, row='Channel', aspect=4, height=2, sharey=True, sharex=True, hue='Dataset')
        g.map_dataframe(sns.violinplot, x='Dataset_Channel', y=loss_name)
        # global_min = df[loss_name].min()
        # global_max = df[loss_name].max()
        # g.set(yticks=np.linspace(global_min, global_max, 5))  # 根据需要调整yticks
        g.set_titles(row_template="{row_name}")  # 设置每个子图的标题为通道名称

        # 调整布局并绘制图形
        plt.subplots_adjust(hspace=0.5)
        plt.show()

def ckpt_plot(ckpt_path, ):
    # Load the checkpoint file
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    # Process the checkpoint data to prepare it for violin plots
    data_frames = {}
    for dataset_name, losses in checkpoint.items():
        for loss_name, values_list in losses.items():
            # Flatten the list of tensors and concatenate them along the first axis
            values = torch.stack(values_list, dim=0).numpy()

            # Create a DataFrame
            df = pd.DataFrame(values, columns=[f'Channel_{channel_name[i]}' for i in range(values.shape[1])])
            df = df.melt(var_name='Channel', value_name=loss_name)
            df['Dataset'] = dataset_name

            # Store the DataFrame in a dictionary
            data_frames[(dataset_name, loss_name)] = df

    # Combine all the DataFrames for violin plots
    all_df = pd.concat(data_frames.values(), ignore_index=True)

    # Remove outliers using IQR
    # Calculate Q1, Q3, and IQR for each loss type
    for loss_name in checkpoint[next(iter(checkpoint))].keys():
        q1 = all_df[loss_name].quantile(0.25)
        q3 = all_df[loss_name].quantile(0.75)
        print(q1, q3)
        iqr = q3 - q1
        # Filter out the outliers
        all_df = all_df[~((all_df[loss_name] < (q1 - 1.5 * iqr)) | (all_df[loss_name] > (q3 + 1.5 * iqr)))]
    # create_separate_violin_plots(all_df)
    FacetGris(all_df)

def main():
    data, name = get_all_dataset_reconstruction_result('../../temp_log')
    data = data['visual_mask_same']
    for loss_type in ['loss', 'loss2']:
        df = prepare_data(data, loss_type)
        df_clean = remove_outliers_iqr(df, 'Value')
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Dataset', y='Value', hue='Channel', data=df_clean)
        plt.title(f'Violin plot for {loss_type}')
        plt.legend(title='Channel')
        plt.show()


if __name__ == '__main__':
    # main()

    ckpt_plot(ckpt_path='../../temp_log/violin.ckpt')
