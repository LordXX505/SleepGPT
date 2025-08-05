import torch
import h5py
import os
import glob
import numpy as np
# 定义 label 与名字的映射
label_to_name = {
    0: "n",        # No pathology
    1: "ins",      # Insomnia
    2: "narco",    # Narcolepsy
    3: "nfle",     # Nocturnal frontal lobe epilepsy
    4: "plm",      # Periodic leg movements
    5: "rbd",      # REM behavior disorder
    6: "sdb",      # Sleep-disordered breathing
}
def find_ckpt_directory(base_path):
    """
    动态找到包含 .ckpt 文件的目录。

    Args:
    -----
    base_path: str
        被试的根目录路径。

    Returns:
    --------
    ckpt_dir: str or None
        如果找到包含 .ckpt 文件的路径，返回路径，否则返回 None。
    """
    # 遍历两层文件夹
    for first_level in glob.glob(os.path.join(base_path, "*")):
        if os.path.isdir(first_level):
            for second_level in glob.glob(os.path.join(first_level, "*")):
                if os.path.isdir(second_level):
                    # 检查是否包含 .ckpt 文件
                    ckpt_files = glob.glob(os.path.join(second_level, "*.ckpt"))
                    if ckpt_files:
                        return second_level
    return None

def extract_ckpts_and_labels(path):
    results = {}
    for pathology in range(7):  # 遍历所有病理
        pathology_path = os.path.join(path, str(pathology))  # 病理路径
        label_name = label_to_name[pathology]  # 映射 label 到名字
        results[label_name] = {}
        # print(f'pathology_path: {pathology_path}')
        # 遍历被试文件夹
        for subject_path in glob.glob(os.path.join(pathology_path, "subject_*")):
            # print(f'subject_path: {subject_path}')
            sub_name = os.path.basename(subject_path)
            results[label_name][sub_name] = []
            ckpt_dir = find_ckpt_directory(subject_path)
            if ckpt_dir is None:
                print(f"Warning: No ckpt directory found for {subject_path}")
                continue
            ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            ckpt_files = sorted(ckpt_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            results[label_name][sub_name].extend(ckpt_files)

    return results


def save_to_h5(ckpt_data, output_file):

    with h5py.File(output_file, 'w') as h5_file:
        for label, subjects in ckpt_data.items():
            for subject, ckpts in subjects.items():
                averaged_data = {}
                for ckpt_file in ckpts:
                    # print(f"Processing {ckpt_file}")
                    # 加载 .ckpt 文件
                    ckpt = torch.load(ckpt_file, map_location='cpu')
                    if 'cls_feats_feature' in ckpt:  # 假设数据在键 'cls_feats_feature'
                        cls_feats = ckpt['cls_feats_feature']  # 形状: (8, 15, dim)
                        stage = ckpt['true_lable'].item()
                        averaged_feats = cls_feats.reshape(8, 15, -1).mean(dim=1)  # 对 dim=1 (15) 取平均, 得到 (8, dim)
                        if stage in averaged_data.keys():
                            averaged_data[stage].append(averaged_feats)
                        else:
                            averaged_data[stage] = [averaged_feats]
                # for key in averaged_data.keys():
                #     print(key)
                if averaged_data:
                    for key in averaged_data.keys():
                        stacked_data = torch.stack(averaged_data[key], dim=0).numpy()  # 形状: (num_ckpts, 8, dim)
                        mean_data = np.mean(stacked_data, axis=0)  # 对维度 0 (num_ckpts) 取平均, 得到 (8, dim)
                        print(mean_data.shape)
                        dataset_path = f"{label}/{subject}/{key}"
                        if dataset_path in h5_file:
                            print(dataset_path)
                        h5_file.create_dataset(f"{label}/{subject}/{key}", data=mean_data)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    path = '/home/cuizaixu_lab/huangweixuan/Sleep/result/UMAP/CAP_umap'
    # 打印结果
    output_h5_path = "data.h5"

    # 提取 ckpt 文件及其路径
    ckpt_data = extract_ckpts_and_labels(path)

    # 保存数据到 h5 文件
    save_to_h5(ckpt_data, '/home/cuizaixu_lab/huangweixuan/Sleep/result/UMAP/CAP_umap/data.h5')