import h5py
import numpy as np
import matplotlib.pyplot as plt
import umap
from mpl_toolkits.mplot3d import Axes3D
# label_mapping = {'ins': 0, 'narco': 1, 'nfle': 2, 'plm': 3, 'rbd': 4, 'sdb': 5,}
def load_h5_data_for_umap(h5_file_path, selected_stage, selected_channel):

    data = []
    labels = []
    with h5py.File(h5_file_path, 'r') as h5_file:
        for pathology in h5_file.keys():
            for subject in h5_file[pathology].keys():
                # 检查是否存在指定的 stage
                if selected_stage in h5_file[pathology][subject]:
                    dataset = h5_file[pathology][subject][selected_stage][:]
                    channel_data = dataset[selected_channel, ]
                    if sum(channel_data) != 0:
                        data.append(channel_data)
                        label = f"{pathology}/{subject}"
                        labels.append(label)
                    else:
                        data.append(np.zeros(1536))
                        label = f"{pathology}/{subject}"
                        labels.append(label)
                else:
                    data.append(np.zeros(1536))
                    label = f"{pathology}/{subject}"
                    labels.append(label)
    try:
        np.stack(data, axis=0)
    except:
        print(selected_stage, selected_channel)
    return np.stack(data, axis=0), labels
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_kmeans(data, labels, n_clusters=5, title="KMeans Clustering"):
    """
    使用 KMeans 聚类并绘制可视化。
    """
    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    # 降维到2D进行可视化
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(data)
    pathologies = list(set(label.split("/")[0] for label in labels))
    pathology_to_subjects = {p: [] for p in pathologies}
    for label in labels:
        pathology, subject = label.split("/")
        if subject not in pathology_to_subjects[pathology]:
            pathology_to_subjects[pathology].append(subject)
    base_colors = {
        "nfle": "red",
        "rbd": "blue",
        "n": "green",
        "plm": "purple",
        "narco": "orange",
        "sdb": "cyan",
        "ins": "pink",
    }

    # 绘制
    fig, ax = plt.subplots(figsize=(15, 15))
    for pathology, subjects in pathology_to_subjects.items():
        for idx, subject in enumerate(subjects):
            indices = [i for i, label in enumerate(labels) if label == f"{pathology}/{subject}"]
            if indices:
                ax.scatter(
                    reduced_data[indices, 0],
                    reduced_data[indices, 1],
                    color=base_colors[pathology],
                    cmap="viridis",  # 使用 Viridis 颜色映射
                    s=50,
                    alpha=0.8
                )

    ax.set_title(title)
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    plt.show()

def plot_umap(data, labels, title="UMAP Projection"):
    """
    使用 UMAP 降维并绘制可视化。
    """
    # 提取 pathology 和 subject
    pathologies = list(set(label.split("/")[0] for label in labels))
    pathology_to_subjects = {p: [] for p in pathologies}
    for label in labels:
        pathology, subject = label.split("/")
        if subject not in pathology_to_subjects[pathology]:
            pathology_to_subjects[pathology].append(subject)

    # 定义每个 pathology 的颜色
    base_colors = {
        "nfle": "red",
        "rbd": "blue",
        "n": "green",
        "plm": "purple",
        "narco": "orange",
        "sdb": "cyan",
        "ins": "pink",
    }

    # 初始化 UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        spread=2.0,
        metric='cosine',
        n_epochs=1000,
        random_state=42,
        n_components=3
    )
    embedding = reducer.fit_transform(data)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制
    for pathology, subjects in pathology_to_subjects.items():
        for idx, subject in enumerate(subjects):
            # 筛选属于该 pathology 和 subject 的点
            indices = [i for i, label in enumerate(labels) if label == f"{pathology}/{subject}"]
            if indices:
                ax.scatter(
                    embedding[indices, 0],
                    embedding[indices, 1],
                    embedding[indices, 2],
                    color=base_colors[pathology],
                    label=f"{pathology}" if idx == 0 else None,  # 同一个 pathology 的 subject 共用标签
                    s=20
                )

    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    plt.legend(markerscale=3, loc="best", fontsize=12)
    plt.show()

h5_file_path = '../../result/UMAP/CAP_umap/data.h5'

selected_stage = "0"
selected_channel = 3
data_list = []
label_lits = []
for selected_stage in range(0, 5):
    for selected_channel in range(0, 5):
        data, labels = load_h5_data_for_umap(h5_file_path, str(selected_stage), selected_channel)
        data_list.append(data)
        label_lits.append(labels)

data_list = np.concatenate(data_list, axis=0)
label_lits = np.concatenate(label_lits, axis=0)
res = {}
for data , label in zip(data_list, label_lits):
    if label not in res:
        res[label] = data
    else:
        res[label] = np.concatenate([res[label], data])
res_data = []
res_label = []
for items in res.items():
    res_data.append(items[1])
    res_label.append(items[0])
plot_kmeans(res_data, res_label, title=f"UMAP Visualization (Stage: {selected_stage}, Channel: {selected_channel})")