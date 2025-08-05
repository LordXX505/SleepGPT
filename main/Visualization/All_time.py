import os.path
import numpy as np
import glob
import re
import xml.etree.ElementTree as ET

def get_time(path, name):

    file_path = os.path.join(path, name)
    items = np.load(file_path, allow_pickle=True)
    res = 0
    if isinstance(items, np.lib.npyio.NpzFile):
        names = items['names']
        print(f'names: {len(names)}')

        nums = items['nums']
        res = np.sum(nums)
    elif items.dtype == np.dtype('O'):
        try:
            data = items.item()
            names = data['names']
            nums = data['nums']
            res = np.sum(nums)
        except:
            res = sum([len(it) for it in items])
    else:
        names = items
        res = len(names)
    return res

def eegfrmi():
    path = '/Volumes/T7/data/SD'
    name = 'train.npy'
    path2 = '/Volumes/T7/data/Young'
    name2 = 'test.npy'
    path3 = '/Volumes/T7/data/Young'
    name3 = 'val.npy'

    return get_time(path, name), get_time(path2, name2), get_time(path3, name3)

def shhs1():
    path = '../../data/shhs_new'
    name = 'train11.npz'
    test_name = 'Test.npz'
    val_name = 'Val.npz'
    path = "/Volumes/T7/data/shhs_new/shhs_new"
    print(len(glob.glob(path + '/shhs1-*')))

    return get_time(path, name), get_time(path, test_name), get_time(path, val_name)
def shhs2():
    path = "/Volumes/T7/shhs_raw/shhs/polysomnography/annotations-events-profusion/shhs2"
    res = 0
    count = 0
    for items in glob.glob(path + '/*.xml'):
        labels = []
        # Read annotation and its header
        t = ET.parse(items)
        r = t.getroot()
        faulty_File = 0
        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:  # Assign label 4 for REM stage
                labels.append(4)
            else:
                labels.append(lbl)
            if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
                faulty_File = 1
        if faulty_File == 1:
            # print("============================== Faulty file ==================")
            continue
        labels = np.asarray(labels)
        y = labels.astype(np.int32)
        w_edge_mins = 30
        nw_idx = np.where(y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        res += len(select_idx)
        count += 1
    return res, count

def physiotest():
    path = '/Volumes/T7/data/Physio/Physio/test'
    name = 'Physio.npy'
    return get_time(path, name)

def physiotrain():
    path = '/Volumes/T7/data/Physio/Physio/training'
    res = 0
    for items in glob.glob(path + '/tr*'):
        all_items = os.listdir(items)
        file_count = sum(1 for item in all_items if os.path.isfile(os.path.join(items, item)))
        res += file_count
    return res

def edf():
    path = '/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/processed'
    res1 = 0
    for items in glob.glob(path + '/SC*'):
        all_items = os.listdir(items)
        file_count = sum(1 for item in all_items if os.path.isfile(os.path.join(items, item)))
        res1 += file_count
    res2 = 0
    for items in glob.glob(path + '/SC*'):
        base_name = os.path.basename(items)
        number = re.findall(r'\d+', base_name)[0]
        if int(number) <= 4192:
            all_items = os.listdir(items)
            file_count = sum(1 for item in all_items if os.path.isfile(os.path.join(items, item)))
            res2 += file_count
    return res1, res2


def mass():
    path = '/Volumes/T7/MASS_Processed'
    res = []
    for i in range(1, 6):
        nums = []
        for items in glob.glob(path + f'/SS{i}/*'):
            if os.path.isdir(items):
                all_items = os.listdir(items)
                file_count = sum(1 for item in all_items if os.path.isfile(os.path.join(items, item)))
                nums.append(file_count)
        res.append(sum(nums))
    return res

if __name__ == '__main__':
    print(f'eeg:{eegfrmi()}')
    print(f'shhs1(): {shhs1()}')
    print(shhs2())
    print(f'physiotest(): {physiotest()}')
    print(f'edf:{edf()}')
    print(f'physiotrain: {physiotrain()}')
    print(f'mass: {mass()}')
