import numpy as np
import os
import glob
import torch
import pyarrow as pa
import tqdm


def load_pyarrow_data(file_path):
    try:
        tables = pa.ipc.RecordBatchFileReader(
            pa.memory_map(file_path, "r")
        ).read_all()
        return tables
    except Exception as e:
        raise RuntimeError(f"Error reading PyArrow file {file_path}: {e}")


def get_stage(data):
    return torch.from_numpy(np.array(data)).long()


def check_label(path_orig, path_aug):
    orig_names = glob.glob(os.path.join(path_orig, '01-*'))
    print(orig_names)
    for on in orig_names:
        print(f'on ：{on}')
        base_name = on.split('/')[-1]
        for items in glob.glob(os.path.join(on, '*')):
            arr_index = items.split('/')[-1]
            aug_item_path = os.path.join(path_aug, base_name, arr_index)
            tables = load_pyarrow_data(items)
            tables_aug = load_pyarrow_data(aug_item_path)
            stage = get_stage(tables['stage'])
            stage_aug = get_stage(tables_aug['stage'])
            assert stage_aug[0] == stage[0], f'stage_aug:{stage_aug}, stage: {stage}, ' \
                                           f'items: {items}, aug_item_path:{aug_item_path}'

def main():
    path_orig = "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2"
    path_aug = "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/Aug_Random"
    print('check labels')
    check_label(path_orig, path_aug)

if __name__ == '__main__':
    main()

