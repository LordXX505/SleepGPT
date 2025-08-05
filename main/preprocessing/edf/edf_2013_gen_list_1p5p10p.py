import pyarrow.parquet as pq
import pyarrow as pa
import h5py
import numpy as np
import argparse
import os
import glob
from sklearn.model_selection import train_test_split
import re
import pyarrow.feather as feather
import pyarrow.ipc as ipc

def get_stage(data):
    return {'Stage_label':np.array(data).astype(np.long)}

def get_epochs(data):
    if isinstance(data, pa.ChunkedArray):
        x = np.array(data.to_pylist())
    elif isinstance(data, pa.Array) or isinstance(data, pa.ListScalar):
        x = np.array(data.as_py())
    else:
        x = np.array(data)
    x = x.astype(np.float32)
    return x
def read_and_concatenate_arrow_files(file_paths):

    tabels = {}
    tabels['signal'] = []
    tabels['stage'] = []
    for file_path in file_paths:
        if file_path.endswith('.feather'):
            table = feather.read_table(file_path)
        elif file_path.endswith('.arrow'):
            with pa.memory_map(file_path, 'r') as source:
                reader = ipc.RecordBatchFileReader(source)
                table = reader.read_all()
                x = get_epochs(table['x'][0])
                stage = get_stage(table['stage'])['Stage_label']
                tabels['signal'].append(x)
                tabels['stage'].append(stage)
    res = {}
    res['signal'] = np.stack(tabels['signal'], axis=0)
    res['stage'] = np.stack(tabels['stage'], axis=0)
    return res

def pyarrow_to_h5py(arrow_table, h5_file_path):

    with h5py.File(h5_file_path, 'w') as h5_file:
        for column_name in arrow_table.keys():
            np_array = arrow_table[column_name]
            h5_file.create_dataset(column_name, data=np_array)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        # default="/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
                        default="/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="processed",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    outputdir = args.output_dir + '/*'

    names = []
    aug_names = []
    nums = []
    files_dict = {}
    name_index = 0
    print_flag = True
    for _, sub in enumerate(glob.glob(outputdir)):
        if os.path.isdir(sub):
            base_name = os.path.basename(sub)
            subject_nums = base_name[3:5]
            number = re.findall(r'\d+', base_name)[0]
            if int(number) <= 4192:
                if int(subject_nums) not in files_dict:
                    files_dict[int(subject_nums)] = [name_index]
                else:
                    files_dict[int(subject_nums)].append(name_index)
                names.append(sub)
                sub_name_list = sub.split('/')
                sub_name_list[7] = 'Aug_file'
                sub_aug_name = os.path.join(*sub_name_list)
                if print_flag is True:
                    print(f'sub_name_list: {sub_name_list}')
                    print_flag = False
                aug_names.append(sub_aug_name)
                name_index += 1
    print(f'The sum of subjects is : {len(files_dict)}')
    n = len(files_dict)
    res = {}
    res_aug = {}
    for name in names:
        print(f'------{name}-------')
        tmp = 0
        for item in os.listdir(name):
            if os.path.isfile(os.path.join(str(name), str(item))):
                tmp += 1
        print(f'num: {tmp}')
        nums.append(tmp)
    edf20_permutation = np.array([14, 5, 17, 8, 7, 12, 0, 15, 9, 10, 3, 1, 6, 18, 2,
                                  13, 4, 11, 19, 16])
    len_train = int(len(edf20_permutation) * 0.6)
    len_valid = int(len(edf20_permutation) * 0.2)
    ######## TRAINing files ##########
    training_files = edf20_permutation[:len_train]
    # load files
    ######## Validation ##########
    validation_files = edf20_permutation[len_train:(len_train + len_valid)]
    # load files
    ######## TesT ##########
    test_files = edf20_permutation[(len_train + len_valid):]

    for portion in [1, 2, 5, 12]:
        subset_items = training_files[:portion]
        file_name = os.path.join(args.data_dir, 'processed', f'Aug_{portion}_New', )
        file_aug_name = os.path.join(args.data_dir, 'Aug_file', f'Aug_{portion}_New', )

        for i in range(10):
            res[f'train_{i}'] = {}
            res[f'train_{i}']['names'] = []
            res[f'train_{i}']['nums'] = []
            for ds_idx in subset_items:
                for tn_idx in files_dict[ds_idx]:
                    res[f'train_{i}']['names'].append(names[tn_idx])
                    res[f'train_{i}']['nums'].append(nums[tn_idx])
            res[f'val_{i}'] = {}
            res[f'val_{i}']['names'] = []
            res[f'val_{i}']['nums'] = []
            for ds_idx in validation_files:
                for tn_idx in files_dict[ds_idx]:
                    res[f'val_{i}']['names'].append(names[tn_idx])
                    res[f'val_{i}']['nums'].append(nums[tn_idx])
            res[f'test_{i}'] = {}
            res[f'test_{i}']['names'] = []
            res[f'test_{i}']['nums'] = []

            for ds_idx in test_files:
                for tn_idx in files_dict[ds_idx]:
                    res[f'test_{i}']['names'].append(names[tn_idx])
                    res[f'test_{i}']['nums'].append(nums[tn_idx])
            print(f'train name : {res[f"train_{i}"]["names"]}')

            res_aug[f'train_{i}'] = {}
            res_aug[f'train_{i}']['names'] = []
            res_aug[f'train_{i}']['nums'] = []
            for ds_idx in subset_items:
                for tn_idx in files_dict[ds_idx]:
                    res_aug[f'train_{i}']['names'].append(aug_names[tn_idx])
                    res_aug[f'train_{i}']['nums'].append(nums[tn_idx])
            res_aug[f'val_{i}'] = {}
            res_aug[f'val_{i}']['names'] = []
            res_aug[f'val_{i}']['nums'] = []
            res_aug[f'test_{i}'] = {}
            res_aug[f'test_{i}']['names'] = []
            res_aug[f'test_{i}']['nums'] = []

            # for ds_idx in validation_files:
            #     for tn_idx in files_dict[ds_idx]:
            #         res_aug[f'val_{i}']['names'].append(aug_names[tn_idx])
            #         res_aug[f'val_{i}']['nums'].append(nums[tn_idx])
            # print(f'train name : {res_aug[f"train_{i}"]["names"]}')
            # for ds_idx in test_files:
            #     for tn_idx in files_dict[ds_idx]:
            #         res_aug[f'test_{i}']['names'].append(aug_names[tn_idx])
            #         res_aug[f'test_{i}']['nums'].append(nums[tn_idx])
        np.save(file_name, arr=res, allow_pickle=True)
        np.save(file_aug_name, arr=res_aug, allow_pickle=True)

if __name__ == '__main__':
    main()

