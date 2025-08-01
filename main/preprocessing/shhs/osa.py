import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def get_test_subjects(path):
    test_set = np.load(os.path.join(path, 'Test.npz'), allow_pickle=True)
    basename_list = []
    for name in test_set['names']:
        basename = os.path.basename(name).split('-')[-1]
        basename_list.append(basename)
    return basename_list

def max_len_append(list_res, items, max_len):
    if len(list_res)<max_len:
        list_res.append(items)
    return list_res

def get_col(path, file_path, test_name):
    import glob
    data = pd.read_csv(file_path)
    column_name = ['nsrrid', 'ahi_a0h3', 'ahi_a0h4', 'ahi_a0h4a', 'ahi_c0h4', 'ahi_c0h4a', 'ahi_o0h4', 'ahi_o0h4a']
    selected_column = data[column_name].sort_values(by='ahi_a0h4a', ascending=False)
    selected_noosa_column = data[column_name].sort_values(by='ahi_a0h4a', ascending=True)
    basename_list = {}
    for name in glob.glob(os.path.join(path, '/Volumes/T7/data/shhs_new/shhs_new/shhs*')):
        basename = os.path.basename(name).split('-')[-1]
        test_set = len(glob.glob(name + '/*'))
        basename_list[basename] = (name, test_set)
    gather_osa_test_list = []
    gather_osa_train_list = []
    for items in selected_column['nsrrid']:
        if str(items) in test_name:
            if selected_column[selected_column['nsrrid']==items]["ahi_a0h4a"].values > 30 and str(items) in basename_list.keys():
                gather_osa_test_list.append(items)
        else:
            if selected_column[selected_column['nsrrid']==items]["ahi_a0h4a"].values > 30 and str(items) in basename_list.keys():
                gather_osa_train_list = max_len_append(gather_osa_train_list, items, 400)
    # assert len(gather_osa_test_list) == 500, len(gather_osa_test_list)
    # assert len(gather_osa_train_list) == 500, len(gather_osa_train_list)
    gather_train_list, gather_test_list = [], []
    for items in selected_noosa_column['nsrrid']:
        if str(items) in test_name:
            if selected_noosa_column[selected_noosa_column['nsrrid']==items]["ahi_a0h4a"].values < 5 and str(items) in basename_list.keys():
                gather_test_list.append(items)
        else:
            if selected_noosa_column[selected_noosa_column['nsrrid']==items]["ahi_a0h4a"].values < 5 and str(items) in basename_list.keys():
                gather_train_list = max_len_append(gather_train_list, items, 400)
    assert len(gather_train_list) == 400, len(gather_train_list)
    # gather_train_mid_list, gather_test_mid_list = [], []
    # train_mid_candidates = selected_column[
    #     (selected_column['ahi_a0h4a'] >= 5) &
    #     (selected_column['ahi_a0h4a'] <= 30) &
    #     (selected_column['nsrrid'].astype(str).isin(test_name))
    #     ]
    # test_mid_candidates = selected_column[
    #     (selected_column['ahi_a0h4a'] >= 5) &
    #     (selected_column['ahi_a0h4a'] <= 30) &
    #     (~selected_column['nsrrid'].astype(str).isin(test_name))
    #     ]
    # median_value = np.median(train_mid_candidates['ahi_a0h4a'].values)
    # train_mid_candidates['distance_to_median'] = abs(train_mid_candidates['ahi_a0h4a'] - median_value)
    # sorted_candidates = train_mid_candidates.sort_values(by='distance_to_median', ascending=True)
    # selected_train_samples = sorted_candidates.head(100)
    # gather_train_mid_list = selected_train_samples['nsrrid'].values
    #
    # test_mid_candidates['distance_to_median'] = abs(test_mid_candidates['ahi_a0h4a'] - median_value)
    # sorted_candidates = test_mid_candidates.sort_values(by='distance_to_median', ascending=True)
    # selected_test_samples = sorted_candidates.head(100)
    # gather_test_mid_list = selected_test_samples['nsrrid'].values

    # assert len(gather_test_mid_list) == 100, len(gather_osa_train_list) == 100
    return gather_osa_train_list, gather_osa_test_list, gather_train_list, gather_test_list

def save_subjects(path, data_list):
    import glob
    basename_list = {}
    names = []
    nums = []
    for name in glob.glob(os.path.join(path, '/Volumes/T7/data/shhs_new/shhs_new/shhs*')):
        basename = os.path.basename(name).split('-')[-1]
        test_set = len(glob.glob(name+'/*'))
        basename_list[basename] = (name, test_set)
        # names.append(name)
        # nums.append(test_set)
    res = {}
    names = []
    nums = []
    for data in data_list:
        try:
            names.append(basename_list[str(data)][0])
            nums.append(basename_list[str(data)][1])
        except:
            print(data, data in basename_list.keys())
    res['names'] = names
    res['nums'] = nums
    np.save(path, arr=res, allow_pickle=True)

def run_pipeline(*args, **kwargs):
    basename_list = get_test_subjects(kwargs['shhs_test_path'])
    gather_osa_train_list, gather_osa_test_list, gather_train_list, gather_test_list = get_col(kwargs['shhs_test_path'], kwargs['info_path'], basename_list)
    combined_train_dataset = np.concatenate([gather_osa_train_list, gather_train_list, ], )
    combined_test_dataset = np.concatenate([gather_osa_test_list, gather_test_list, ])
    save_subjects(os.path.join(kwargs['shhs_test_path'], 'train_osa_c2_new'), combined_train_dataset)
    save_subjects(os.path.join(kwargs['shhs_test_path'], 'test_osa_c2_new'), combined_test_dataset)
    # save_subjects(os.path.join(kwargs['shhs_test_path'], 'test_osa_new'), None)

if __name__ == '__main__':
    shhs_test_path = '../../../data/shhs_new/'
    info_path = '/Users/hwx_admin/Downloads/shhs_log/shhs/datasets/shhs1-dataset-0.21.0.csv'
    run_pipeline(shhs_test_path=shhs_test_path, info_path=info_path)


