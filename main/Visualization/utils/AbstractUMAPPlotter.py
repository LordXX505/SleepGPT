import os
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
from abc import ABC, abstractmethod

class AbstractUMAPPlotter(ABC):
    def __init__(self, n_neighbors=30, min_dist=0.1, n_components=2, random_state=None):
        tqdm_kwds = {
            'ncols': 400,
            'colour': '#FFC0CB'
        }
        if random_state is not None:
            njobs = 1
        else:
            njobs = -1
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                 n_components=n_components, random_state=random_state, n_jobs=njobs,
                                 verbose=True, tqdm_kwds=tqdm_kwds)
        self.extra_config = {}

    @abstractmethod
    def load_checkpoint_data(self, checkpoint_dir, down):
        """加载checkpoint数据，并返回cls_feats_feature, predicted, true_label."""
        pass

    def transform_and_plot(self, checkpoint_dir, save_dir, label_to_color, down=1, plot_predicted=False, s=1):
        # 加载并处理所有checkpoint数据
        all_cls_feats, all_predicted, all_true_labels = self.load_checkpoint_data(checkpoint_dir, down=down)

        # 使用UMAP进行降维
        embedding = self.reducer.fit_transform(all_cls_feats, )
        print('embedding finished')
        # 创建自定义颜色映射
        colors = list(label_to_color.values())
        custom_cmap = ListedColormap(colors)

        # 绘制True Label的UMAP图
        plt.figure(figsize=(7, 7))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=all_true_labels, cmap=custom_cmap, s=s, alpha=0.8)
        # self.add_legend(label_to_color)
        if 'portion_aug' in self.extra_config.keys():
            plt.savefig(os.path.join(save_dir, f'{portion_aug}.svg'))
            plt.savefig(os.path.join(save_dir, f'{portion_aug}.png'), dpi=600)
        else:
            plt.savefig(os.path.join(save_dir, f'UMAP.svg'))
            plt.savefig(os.path.join(save_dir, f'UMAP.png'), dpi=600)
        plt.close()

        # 如果需要绘制Predicted的UMAP图
        if plot_predicted:
            plt.figure(figsize=(7, 7))
            colors_predicted = np.array([label_to_color[label] for label in all_predicted])
            plt.scatter(embedding[:, 0], embedding[:, 1], c=colors_predicted, s=1, alpha=0.8)
            if 'portion_aug' in self.extra_config.keys():
                plt.savefig(os.path.join(save_dir, f'predict_{portion_aug}.png'), dpi=600)
                plt.savefig(os.path.join(save_dir, f'predict.png'), dpi=600)
            else:
                plt.savefig(os.path.join(save_dir, f'predict.png'), dpi=600)
            plt.close()

    def add_legend(self, label_to_color):
        """添加图例"""
        legend_handles = []
        for label, color in label_to_color.items():
            handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                   markersize=10, label=f'Label {label}')
            legend_handles.append(handle)
        plt.legend(handles=legend_handles, loc='best')

class UMAPPlotter(AbstractUMAPPlotter):
    def __init__(self, n_neighbors=20, min_dist=0.1, n_components=2, random_state=None):
        super().__init__(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=n_components, random_state=random_state)

    def load_checkpoint_data(self, checkpoint_dir, down=1):
        """具体实现从checkpoint文件加载数据"""
        all_cls_feats = []
        all_predicted = []
        all_true_labels = []
        print('start to process checkpoints')
        # 遍历目录中的所有checkpoint文件
        for ckpt_file in sorted(os.listdir(checkpoint_dir)):
            if ckpt_file.endswith('.ckpt'):
                checkpoint_path = os.path.join(checkpoint_dir, ckpt_file)
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

                cls_feats_feature = checkpoint['cls_feats_feature']
                predicted = checkpoint['predicted']
                true_label = checkpoint['true_lable']

                all_cls_feats.append(cls_feats_feature.view(-1, cls_feats_feature.size(-1)))  # 展平为 [B*T, D]
                all_predicted.append(predicted.view(-1))  # 展平为 [B*T]
                all_true_labels.append(true_label.view(-1))  # 展平为 [B*T]

        # 拼接所有checkpoint的数据
        all_cls_feats = torch.cat(all_cls_feats, dim=0).cpu().numpy()
        all_predicted = torch.cat(all_predicted, dim=0).cpu().numpy()
        all_true_labels = torch.cat(all_true_labels, dim=0).cpu().numpy()
        print('load all checkpoints')
        return all_cls_feats[::down], all_predicted[::down], all_true_labels[::down]

# 示例使用
if __name__ == '__main__':
    label_to_color = {
        0: '#55b7e6',
        1: '#56ba78',
        2: '#ec7069',
        3: '#9eab3f',
        4: '#bc7ab5',
    }

    plotter = UMAPPlotter()
    for mode in ['', '_aug']:
        for subjects in ['_1', '_2', '_5', '_12']:
            portion_aug = 'portion' + mode + subjects
            checkpoint_dir = f'../../result/EDF/UMAP/0/{portion_aug}'
            save_dir = f'../../result/EDF/UMAP/'
            os.makedirs(save_dir, exist_ok=True)
            plotter.extra_config['portion_aug'] = portion_aug
            plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, plot_predicted=True)