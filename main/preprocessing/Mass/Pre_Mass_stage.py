import copy
import os
import glob
import ntpath
import logging
import argparse
import sys
import pandas as pd
import mne
import numpy as np
import pyarrow as pa
import gc
import torch

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,  # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}
picks = [
    ['EEG C3-CLE', 'EEG C4-CLE', 'EMG Chin1', 'EMG Chin3', 'EOG Left Horiz', 'EEG F3-CLE', 'EEG O1-CLE', 'EEG Pz-CLE'],
    ['EEG C3-CLE', 'EEG C4-CLE', 'EMG Chin', 'EOG Left Horiz', 'EEG F3-CLE', 'EEG Fpz-CLE', 'EEG O1-CLE', 'EEG Pz-CLE'],
    ['EEG C3-LER', 'EEG C4-LER', 'EMG Chin1', 'EMG Chin3', 'EOG Left Horiz', 'EEG F3-LER',  'EEG O1-LER', 'EEG Pz-LER'],
    ['EEG C3-CLE', 'EEG C4-CLE', 'EMG Chin', 'EOG Left Horiz',  'EEG O1-CLE', ],
    ['EEG C3-LER', 'EEG C4-LER', 'EMG Chin1', 'EMG Chin3', 'EOG Left Horiz', 'EEG F3-LER', 'EEG O1-LER', 'EEG Pz-LER'],
]


def check_chanels(channels, k):
    res = []
    temp_pick = copy.deepcopy(picks[k-1])
    if k == 1:
        anchor = channels[0].split()[-1].split('-')[-1]
        if anchor == 'LER':
            for i in range(len(temp_pick)):
                if 'CLE' in temp_pick[i]:
                    temp_pick[i] = temp_pick[i].replace('CLE', 'LER')
    return temp_pick
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7/",
                            # default="/home/cuizaixu_lab/huangweixuan/data/data",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="MASS_Processed",
                        help="Directory where to save outputs.")
    parser.add_argument("--log_file", type=str, default="info_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    for k in range(2, 6):
        logger.info(f'----------Using Mass SS{k} dataset------------')
        base_data_dir = os.path.join(args.data_dir, f'SS{k}', 'bio')
        # Read raw and annotation from EDF files
        psg_fnames = glob.glob(os.path.join(base_data_dir, "*PSG.edf"))
        ann_fnames = glob.glob(os.path.join(base_data_dir, "*Base.edf"))
        psg_fnames.sort()
        ann_fnames.sort()
        psg_fnames = np.asarray(psg_fnames)
        ann_fnames = np.asarray(ann_fnames)
        resume_idx = 0
        # resume_idx = 46 if k == 1 else 0
        for i in range(resume_idx, len(psg_fnames)):
            # base_name = os.path.basename('/home/cuizaixu_lab/huangweixuan/data/data/SS1/bio/01-01-0048 PSG.edf')
            # if base_name != os.path.basename(psg_fnames[i]):
            #     continue
            logger.info("Loading ...")
            logger.info("Signal file: {}".format(psg_fnames[i]))
            logger.info("Annotation file: {}".format(ann_fnames[i]))

            psg_f = mne.io.read_raw_edf(psg_fnames[i], preload=True)
            ann_f = mne.read_annotations(ann_fnames[i])
            pick_channels = check_chanels(psg_f.ch_names, k)
            psg_f.pick(pick_channels)
            if 'EMG Chin3' in pick_channels:
                def func(array, EMG3):
                    array -= EMG3[0]
                    return array
                data, times = psg_f['EMG Chin3']
                psg_f.apply_function(func, picks=['EMG Chin1'], EMG3=data)
                psg_f.drop_channels(['EMG Chin3'])  # drop channel EMG3
            # print(max(psg_f['EEG Fpz-Cz'][0]), min(psg_f['EEG Fpz-Cz'][0]), max(psg_f[2][0]))
            # print(max(psg_f['EMG submental'][0]), min(psg_f['EMG submental'][0]))
            # print(max(psg_f['EOG horizontal'][0]), min(psg_f['EOG horizontal'][0]))
            # print(max(psg_f['EEG Pz-Oz'][0]), min(psg_f['EEG Pz-Oz'][0]))
            logger.info(f'final channles: {psg_f.ch_names}')
            psg_f.set_annotations(ann_f)
            psg_f.resample(100)
            res = psg_f.crop_by_annotations()
            labels = []
            signals = []
            ##
            test_labels = []
            ##
            for item_indx, item in enumerate(res):
                signals.append(np.copy(item.get_data()[:, :-1]))
                annotations = item.annotations
                ######
                index = 0
                for _, duration in enumerate(annotations.duration):
                    if duration > 29:
                        index = _
                assert index != -1, f'k: {k}, i: {i}, index: {index}, item_indx:{item_indx}'
                test_labels.append([ann2label[item.annotations.description[index]]])
                ########
                for _, duration in enumerate(annotations.duration):
                    if k == 2 or k == 4 or k == 5:
                        if duration > 19:
                            index = _
                            break
                    else:
                        if duration > 29:
                            index = _
                            break
                if index == -1 and item_indx == (len(res)-1):
                    labels.append([6])
                else:
                    assert index != -1, f'k: {k}, i: {i}, index: {index}, item_indx:{item_indx}'
                    labels.append([ann2label[item.annotations.description[index]]])
            new_signals = []
            ##
            test_labels = np.hstack(test_labels)
            ##
            labels = np.hstack(labels)
            ##
            np.set_printoptions(threshold=np.inf)
            print(f'different: test: {test_labels}, label: {labels}')
            ##
            for _, (ann_f_i, labels_j) in enumerate(zip(ann_f.description, labels)):
                if _ != (len(labels)-1):
                    assert ann2label[ann_f_i] == labels_j
            if k == 2 or k == 4 or k == 5:
                for item_index in range(1, len(signals)-1):
                    new_signals.append(np.concatenate([signals[item_index-1][:, 1500:],
                                                       signals[item_index], signals[item_index+1][:, :500]], axis=-1))
                labels = labels[1:-1]
            else:
                new_signals = signals
            assert len(labels) == len(new_signals)

            # # epochs = len(psg_f)//3000
            # # Remove annotations that are longer than the recorded signals
            # # labels = labels[:epochs]
            # # Get epochs and their corresponding labels
            # x = np.asarray(new_signals).astype(np.float32)
            # y = labels.astype(np.int32)
            # logger.info(f"the length of the signal is {x.shape}")
            # # Select only sleep periods
            # w_edge_mins = 30
            # nw_idx = np.where(y != stage_dict["W"])[0]
            # start_idx = nw_idx[0] - (w_edge_mins * 2)
            # end_idx = nw_idx[-1] + (w_edge_mins * 2)
            # if start_idx < 0: start_idx = 0
            # if end_idx >= len(y): end_idx = len(y) - 1
            # select_idx = np.arange(start_idx, end_idx + 1)
            # logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
            # x = x[select_idx]
            # y = y[select_idx]
            # logger.info("Data after selection: {}, {}".format(x.shape, y.shape))
            #
            # # Remove movement and unknown
            # move_idx = np.where(y == stage_dict["MOVE"])[0]
            # unk_idx = np.where(y == stage_dict["UNK"])[0]
            # if len(move_idx) > 0 or len(unk_idx) > 0:
            #     remove_idx = np.union1d(move_idx, unk_idx)
            #     logger.info("Remove irrelavant stages")
            #     logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            #     logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            #     logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            #     logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            #     select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            #     x = x[select_idx]
            #     y = y[select_idx]
            #     logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))
            #
            # # Save
            # cnt = 0
            # name = os.path.basename(psg_fnames[i]).split('.')[0].split()[0]
            # for _x, _y in zip(x, y):
            #     assert _x.shape[1] == 3000, f'shape: {_x.shape}'
            #     assert _x.shape[0] == len(pick_channels) or _x.shape[0] == (len(pick_channels)-1), f'x shape:{_x.shape[0]}, len of channels: {len(pick_channels)}'
            #     dataframe = pd.DataFrame(
            #         {'x': [_x.tolist()], 'stage': _y, }
            #     )
            #     table = pa.Table.from_pandas(dataframe)
            #     os.makedirs(f"{args.output_dir}/SS{k}/{name}", exist_ok=True)
            #     with pa.OSFile(
            #             f"{args.output_dir}/SS{k}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
            #     ) as sink:
            #         with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            #             writer.write_table(table)
            #     cnt += 1
            #     del dataframe
            #     del table
            #     gc.collect()
            # logger.info("\n=======================================\n")


    # outputdir = args.output_dir + '/*'
    #
    # names = []
    # nums = []
    # for sub in glob.glob(outputdir):
    #     if os.path.isdir(sub):
    #         names.append(sub)
    # for name in names:
    #     print(f'------{name}-------')
    #     tmp = 0
    #     for item in os.listdir(name):
    #         if os.path.isfile(os.path.join(str(name), str(item))):
    #             tmp += 1
    #     print(f'num: {tmp}')
    #     nums.append(tmp)
    #
    # nums = np.array(nums)
    # print(f'nums: {nums}')
    # n = len(names)
    # idx = np.arange(n)
    # names = np.array(names)
    # k_split = n//10
    # res = {}
    # for i in range(10):
    #     st = i * k_split
    #     ed = (i + 1) * k_split
    #     idx_split = idx[st:ed]
    #     idx_train = np.setdiff1d(idx, idx_split)
    #     np.random.shuffle(idx_train)
    #     res[f'train_{i}'] = {}
    #     res[f'train_{i}']['names'] = names[idx_train[7:]]
    #     res[f'train_{i}']['nums'] = nums[idx_train[7:]]
    #     res[f'val_{i}'] = {}
    #     res[f'val_{i}']['names'] = names[idx_split]
    #     res[f'val_{i}']['nums'] = nums[idx_split]
    #     res[f'test_{i}'] = {}
    #     res[f'test_{i}']['names'] = names[idx_split]
    #     res[f'test_{i}']['nums'] = nums[idx_split]
    #     print(idx, st, ed, idx_train)
    #     print(f'train name : {res[f"train_{i}"]["names"]}')
    #     print(f'test name : {res[f"test_{i}"]["names"]}')
    #     print(f'val name : {res[f"val_{i}"]["names"]}')
    # np.save(os.path.join(args.output_dir, f'split_k_10'), arr=res, allow_pickle=True)
    # print(f'len: names: {len(names)}')

def get_epochs(data):
    try:
        x = np.array(data.as_py())
    except:
        x = np.array(data.to_pylist())

    x = x * 1e6
    channel = np.array([16, 18, 36, 52])
    x = torch.from_numpy(x).float()
    channel = torch.from_numpy(channel)
    assert x.shape[0] == channel.shape[0], f"x shape: {x.shape[0]}, c shape: {channel.shape[0]}"

    return {'x': [x, channel]}

if __name__ == "__main__":
    main()
    # data_dir = "/Volumes/T7 Shield/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/processed"
    # subject_name = 'SC4241E0'
    # import matplotlib.pyplot as plt
    # subject_path = os.path.join(data_dir, subject_name)
    # arr = os.path.join(subject_path, '00335.arrow')
    # tables = pa.ipc.RecordBatchFileReader(
    #     pa.memory_map(arr, "r")
    # ).read_all()
    # x = get_epochs(tables['x'][0])
    # x = x['x']
    # plt.plot(x[0])
    # plt.show()
