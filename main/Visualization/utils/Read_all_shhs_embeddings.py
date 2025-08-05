import os.path

import h5py
import numpy as np
import glob
import torch
def find_ckpt_directory(base_path):

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
    for subject_path in glob.glob(os.path.join(path, "shhs*")):
        # print(f'subject_path: {subject_path}')
        sub_name = os.path.basename(subject_path)
        results[sub_name] = []
        ckpt_dir = find_ckpt_directory(subject_path)
        if ckpt_dir is None:
            print(f"Warning: No ckpt directory found for {subject_path}")
            continue
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        ckpt_files = sorted(ckpt_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        results[sub_name].extend(ckpt_files)

    return results
def save_to_h5(ckpt_data, output_file):

    with h5py.File(output_file, 'w') as h5_file:
        for subject, ckpts in ckpt_data.items():
            averaged_data = {}
            for ckpt_file in ckpts:
                ckpt = torch.load(ckpt_file, map_location='cpu')
                print(f"ckpt 类型: {type(ckpt)}")
                print(f"ckpt 内容: {ckpt}")
                print(ckpt.keys())  # 这里如果 `ckpt` 是字典，不应该报错
                if 'cls_feats_feature' in ckpt.keys():  # 假设数据在键 'cls_feats_feature'
                    cls_feats = ckpt['cls_feats_feature']  # 形状: (8, 15, dim)
                    stage = ckpt['true_lable'].item()
                    averaged_feats = cls_feats.reshape(4, 15, -1).mean(dim=1)  # 对 dim=1 (15) 取平均, 得到 (8, dim)
                    if stage in averaged_data.keys():
                        averaged_data[stage].append(averaged_feats)
                    else:
                        averaged_data[stage] = [averaged_feats]
            if averaged_data:
                for key in averaged_data.keys():
                    stacked_data = torch.stack(averaged_data[key], dim=0).numpy()  # 形状: (num_ckpts, 8, dim)
                    mean_data = np.mean(stacked_data, axis=0)  # 对维度 0 (num_ckpts) 取平均, 得到 (8, dim)
                    print(mean_data.shape)
                    dataset_path = f"{subject}/{key}"
                    if dataset_path in h5_file:
                        print(dataset_path)
                    h5_file.create_dataset(f"{subject}/{key}", data=mean_data)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    path = '/home/cuizaixu_lab/huangweixuan/Sleep/result/UMAP/shhs1_test_umap'

    output_h5_path = "data.h5"

    ckpt_data = extract_ckpts_and_labels(path)
    save_to_h5(ckpt_data, '/home/cuizaixu_lab/huangweixuan/Sleep/result/UMAP/shhs1_test_umap/data.h5')

