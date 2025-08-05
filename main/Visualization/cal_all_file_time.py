import glob
import os
import numpy as np

def main():
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS', 'Young',
                'EDF', 'MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5', ]
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/SD",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/Physio/training",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/Physio/test",
                "/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/shhs_new/shhs_new",
                "/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/Young",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS1",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS3",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS4",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS5",
                ]
    total_time = 0
    for dataset_name, dataset_dir in zip(datasets, data_dir):
        files = glob.glob(os.path.join(dataset_dir, '*/*'))
        filtered_files = [file for file in files if not (file.endswith('.npz') or file.endswith('.npy'))]
        print(f'dataset_name: {dataset_name}, nums: {len(filtered_files)}')
        total_time += 30*len(filtered_files)
    print(total_time)
if __name__ == "__main__":
    main()