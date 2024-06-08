import argparse
import re
import sys

import numpy as np
import os
import glob
import pyarrow as pa
import torch

def get_epochs( data):
    try:
        x = np.array(data.as_py())
    except:
        x = np.array(data.to_pylist())
    # rank_zero_info(f'settings: {self.settings}')

    x = torch.from_numpy(x).float()

    return {'x': x}

def check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Volumes/T7 Shield/data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
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
    for file in glob.glob(outputdir):
        if os.path.isdir(file):
            for item in os.listdir(file):
                real_path = os.path.join(file, item)
                tables = pa.ipc.RecordBatchFileReader(
                    pa.memory_map(real_path, "r")
                ).read_all()
                x = get_epochs(tables['x'])['x']
                assert x.shape[1] == 4


if __name__ == '__main__':
    check()