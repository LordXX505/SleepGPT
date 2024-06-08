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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
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

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))
        # base_name = os.path.basename(psg_fnames[i])
        psg_f = mne.io.read_raw_edf(psg_fnames[i])
        ann_f = mne.read_annotations(ann_fnames[i])
        psg_f.pick(['EMG submental',  'EOG horizontal', 'EEG Fpz-Cz', 'EEG Pz-Oz'])
        # print(max(psg_f['EEG Fpz-Cz'][0]), min(psg_f['EEG Fpz-Cz'][0]), max(psg_f[2][0]))
        # print(max(psg_f['EMG submental'][0]), min(psg_f['EMG submental'][0]))
        # print(max(psg_f['EOG horizontal'][0]), min(psg_f['EOG horizontal'][0]))
        # print(max(psg_f['EEG Pz-Oz'][0]), min(psg_f['EEG Pz-Oz'][0]))

        sampling_rate = 100.0
        n_epoch_samples = int(30 * sampling_rate)
        logger.info("Sample rate: {}".format(sampling_rate))
        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.onset, ann_f.duration, ann_f.description
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % 30 != 0:
                logger.info(f"Something wrong: {duration_sec} {30}")
                raise Exception(f"Something wrong: {duration_sec} {30}")
            duration_epoch = int(duration_sec / 30)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)
        epochs = len(psg_f)//3000
        # Remove annotations that are longer than the recorded signals
        labels = labels[:epochs]
        # Get epochs and their corresponding labels
        x = np.asarray(np.split(psg_f.get_data()[:, :epochs*3000], epochs, axis=1)).astype(np.float32)
        y = labels.astype(np.int32)
        logger.info(f"the length of the signal is {x.shape}")
        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
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

        # Save
        cnt = 0
        name = os.path.basename(psg_fnames[i]).split('.')[0].split('-')[0]
        for _x, _y in zip(x, y):
            dataframe = pd.DataFrame(
                {'x': [_x.tolist()], 'stage': _y, }
            )
            table = pa.Table.from_pandas(dataframe)
            os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
            with pa.OSFile(
                    f"{args.output_dir}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            cnt += 1
            del dataframe
            del table
            gc.collect()
        logger.info("\n=======================================\n")

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
