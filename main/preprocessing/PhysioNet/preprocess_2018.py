import os
import glob
import logging
import numpy as np
import mne
from mne.io import concatenate_raws, read_raw_edf
from datetime import datetime
import dhedfreader
import pandas as pd
import pyarrow as pa
import gc
import math
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
EPOCH_SEC_SIZE = 30

output_dir = '/data'
select_ch = ['EOG horizontal', 'EEG Fpz-Cz', 'EEG Pz-Oz', ]


log_file = '/data/info_ch_extract.log'
output_dir = os.path.join(output_dir, 'sleepedf_2018.log')
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(os.path.join(output_dir, log_file))
Data_dir = ['/data/sleep-edf-database-expanded-1.0.0/sleep-telemetry']
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

sucess = []
wrong = []
# Select channel
# select_ch = args.select_ch
# Read raw and annotation from EDF files
for data_dir in Data_dir:
    psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)
    store_path = []
    filename = f'/data/data/{data_dir.split("/")[-1]}'

    logger.info('filename: {}'.format(filename))

    print(os.getcwd())
    for psg_fname, ann_fname in zip(psg_fnames, ann_fnames):
        name = os.path.basename(psg_fname).split('.')[0]
        logger.info('psg name {}'.format(name))
        anno = mne.read_annotations(ann_fname)
        psg = mne.io.read_raw_edf(psg_fname)
        psg.pick(select_ch)
        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fname))  # read the raw data
        logger.info("Annotation file: {}".format(ann_fname))  # read the annotation
        data = psg.get_data()

        f = open(psg_fname, 'r', errors='ignore')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_fname, 'r', errors='ignore')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt
        # if data.shape[1] % (30 * 100) != 0:
        #     raise Exception("Something wrong")

        # Generate label and remove indices
        remove_idx = []  # indicies of the data that will be removed
        labels = []  # indicies of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str[2:-1]]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                label_idx.append(idx)

                print("Include onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str
                ))
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                remove_idx.append(idx)

                print("Remove onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str))
        labels = np.hstack(labels)

        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # Select only the data with labels
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print("after intersect label: {}".format(select_idx.shape))

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                # n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                # n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
                if n_label_trims != 0:
                    # select_idx = select_idx[:-n_trims]
                    labels = labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

        x = signals.astype(np.float32)  # 2650, 3000
        y = labels.astype(np.int32)  # 2650

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]  # 653 epoch # select sleep
        start_idx = nw_idx[0] - (w_edge_mins * 2)  # 916
        end_idx = nw_idx[-1] + (w_edge_mins * 2)  # 1801
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1  # select only sleep before and after 30epochs
        select_idx = np.arange(start_idx, end_idx + 1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))
        cnt = 0
        for _x, _y in zip(x, y):
            dataframe = pd.DataFrame(
                {'x': [_x.tolist()], 'stage': _y}
            )
            table = pa.Table.from_pandas(dataframe)
            os.makedirs(f"{filename}/{name}", exist_ok=True)
            store_path.append(f"{filename}/{name}/{str(cnt).zfill(5)}.arrow")
            with pa.OSFile(
                    f"{filename}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            cnt += 1
            del dataframe
            del table
            gc.collect()
        sucess.append(name)

    # logger.info('np save: ', os.path.join(filename, 'all.npy'))
    # np.save(os.path.join(filename, 'all.npy'), np.stack(store_path), allow_pickle=True)

print("sucess", sucess)
print("wrong", wrong)

