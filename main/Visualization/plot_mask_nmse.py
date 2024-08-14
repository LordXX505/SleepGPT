import os.path
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
dataset_names = {
                'physio_train', 'SHHS1',
                'EDF', 'MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5',
}
def plot_and_save(res, name):
    # Define line styles for each dataset
    markers = {
        'EDF': 'o',  # Circle
        'MASS1': '^',  # Triangle
        'MASS2': 's',  # Square
        'MASS3': 'D',  # Diamond
        'MASS5': 'x',  # X
        'physio_train': 'P'  # Plus
    }

    line_styles = {
        'EDF': 'solid',
        'MASS1': 'dashed',
        'MASS2': 'dashdot',
        'MASS3': 'dotted',
        'MASS5': (0, (3, 1, 1, 1)),
        'physio_train': (0, (5, 2))  # Dashed with long gaps
    }

    # Plotting the data
    plt.figure(figsize=(10, 6))
    for dataset, values in res.items():
        x = list(values.keys())
        y = list(values.values())
        plt.plot(x, y, linestyle=line_styles[dataset], marker=markers[dataset], color='black', label=dataset)

    plt.xlabel('Threshold')
    plt.ylabel('Value')

    plt.title('Data Visualization for Different Datasets')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./result/{name}.svg')
    plt.show()


result_path = '../../result/'
dataset_path_list = {dn:os.path.join(result_path, dn, 'mask_same') for dn in dataset_names}
res = {}
for dn, dpl in dataset_path_list.items():
    res[dn] = {}
    for pm in [0.45, 0.6, 0.75, 0.9]:
        res[dn][pm] = []
        for kfold_item in os.listdir(dpl):
            real_path = os.path.join(dpl, kfold_item, f'overall_[{pm}]', 'res.ckpt')
            item = torch.load(real_path, map_location='cpu')
            res[dn][pm].append(item['loss'])
        res[dn][pm] = torch.mean(torch.stack(res[dn][pm], dim=0), dim=0)

# print(res)
pz_dataset = ['EDF', 'MASS1', 'MASS2', 'MASS3', 'MASS5']
res_pz = {}
for pzd in pz_dataset:
    res_pz[pzd] = {}
    for k, v in res[pzd].items():
        res_pz[pzd][k] = v[-1]
print('pz:', res_pz)
plot_and_save(res_pz,'Pz')


f3_dataset = ['MASS1', 'MASS2', 'MASS3', 'MASS5', 'physio_train']
res_f3 = {}
for f3d in f3_dataset:
    res_f3[f3d] = {}
    for k, v in res[f3d].items():
        res_f3[f3d][k] = v[4]
print('f3:', res_f3)
plot_and_save(res_f3,'F3')

res_c = {}
for c in range(8):
    res_c[c] = {}
    for pm in [0.45, 0.6, 0.75, 0.9]:
        res_c[c][pm] = 0
        cnt = 0
        for dn in dataset_names:
            if res[dn][pm][c] != 0:
                res_c[c][pm] += res[dn][pm][c]
                cnt += 1
        res_c[c][pm] = res_c[c][pm]/cnt
    print(res_c[c][0.9] - res_c[c][0.45])
print('res:', res_c)

plt.figure(figsize=(12, 8))
filtered_data = {k: v for k, v in res_c.items() if max(v.values()) <= 10}
for key, values in filtered_data.items():
    x = list(values.keys())
    y = list(values.values())
    plt.plot(x, y, linestyle='-', marker='o', label=f'Dataset {key}')

plt.xlabel('Threshold')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(f'./result/all_c_new.svg')
plt.ylim(0, 3)
plt.show()
