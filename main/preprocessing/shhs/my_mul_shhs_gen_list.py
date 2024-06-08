import argparse

import numpy as np
import torch
import os

def gen_shhs(files: np.ndarray, save_path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", type=str,
                        default='/home/cuizaixu_lab/huangweixuan/data/data/shhs_new',
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--select_path", type=str,
                        default='/home/cuizaixu_lab/huangweixuan/Sleep/main/preprocessing',
                        help="File path to the Sleep-EDF dataset.")
    args = parser.parse_args()
    orig_path = args.orig_path
    new_path = args.new_path





if __name__ == '__main__':
    gen_shhs()