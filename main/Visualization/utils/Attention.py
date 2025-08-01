import h5py
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from main.modules.heads import Attn
from torch.utils.data import DataLoader, Dataset
class SleepDataset(Dataset):
    def __init__(self, data, labels, mask):
        self.data = data
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.mask[idx]
def organize_data_by_subjects(h5_file_path, train_subjects, test_subjects,
                              exclude_c=None):
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    train_attn_mask = []
    test_attn_mask = []
    with h5py.File(h5_file_path, 'r') as h5_file:
        for pathology in h5_file.keys():
            if pathology == 'rbd' or pathology == 'plm' or pathology == 'sdb' or pathology == 'narco':
                continue
            trs = train_subjects[pathology]
            tes = test_subjects[pathology]
            for subject in h5_file[pathology].keys():
                all_data = []
                attn_mask = torch.ones(40)
                for selected_stage in range(0, 5):
                    if str(selected_stage) in h5_file[pathology][subject].keys():
                        dataset = h5_file[pathology][subject][str(selected_stage)][:]
                        all_data.append(torch.tensor(dataset))
                        for c in range(0, 8):
                            if sum(h5_file[pathology][subject][str(selected_stage)][c, :]) == 0:
                                attn_mask[(selected_stage*8) + c] = 0
                    else:
                        all_data.append(torch.zeros(8, 1536))
                        attn_mask[(selected_stage*8):(selected_stage*8+8)] = 0
                all_data = torch.stack(all_data, dim=0).reshape(-1, 1536)
                if subject in trs:
                    train_data.append(all_data)
                    train_attn_mask.append(attn_mask)
                    train_labels.append(pathology)
                else:
                    test_data.append(all_data)
                    test_attn_mask.append(attn_mask)
                    test_labels.append(pathology)
        train_data = torch.stack(train_data, dim=0)
        train_attn_mask = torch.stack(train_attn_mask, dim=0)
        test_data = torch.stack(test_data, dim=0)
        test_attn_mask = torch.stack(test_attn_mask, dim=0)

    return train_data, test_data, train_labels, test_labels, train_attn_mask, test_attn_mask
def split_subjects_by_pathology(h5_file_path, test_size=0.2, random_state=42):
    """
    根据病理按 subject 粒度划分训练和测试数据集。

    返回:
        train_subjects, test_subjects: 包含每个 pathology 对应的划分结果。
    """
    train_subjects, test_subjects = {}, {}

    with h5py.File(h5_file_path, 'r') as h5_file:
        for pathology in h5_file.keys():
            subjects = list(h5_file[pathology].keys())
            train, test = train_test_split(subjects, test_size=test_size, random_state=random_state)
            train_subjects[pathology] = train
            test_subjects[pathology] = test

    return train_subjects, test_subjects

h5_file_path = '../../../result/UMAP/CAP_umap/data.h5'
train_subjects, test_subjects = split_subjects_by_pathology(h5_file_path)
train_data, test_data, train_labels, test_labels, train_attn_mask, test_attn_mask = organize_data_by_subjects(h5_file_path, train_subjects, test_subjects)
unique_labels = list(set(train_labels + test_labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
print(f'label_to_index: {label_to_index}')
train_labels = torch.tensor([label_to_index[label] for label in train_labels])
test_labels = torch.tensor([label_to_index[label] for label in test_labels])
epochs = 20
batch_size = 8
train_dataset = SleepDataset(train_data, train_labels, train_attn_mask)
test_dataset = SleepDataset(test_data, test_labels, test_attn_mask)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dim = 32
model = Attn(hidden_size=1536, out_size=out_dim, channels=1).to(device)

classificaiton = nn.Linear(out_dim, 7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
norm = nn.LayerNorm(out_dim)
for epoch in range(epochs):
    model.train()
    for batch_data, batch_labels, mask_attn in train_loader:  # 假设使用 DataLoader 提供数据
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        b = batch_data.size(0)
        # 前向传播
        logits = classificaiton(norm(model(batch_data,  attn_mask=mask_attn)))
        loss = criterion(logits.reshape(b, -1), batch_labels)
        print(f'loss: {loss}')
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证流程
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels, mask_attn in test_loader:
            b = batch_data.size(0)
            batch_data = batch_data.to(device)
            logits = classificaiton(norm(model(batch_data, attn_mask=mask_attn)))

            preds = torch.argmax(logits.reshape(b, -1), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Epoch {epoch + 1}, Validation Accuracy: {acc:.4f}, cm: {cm}")
