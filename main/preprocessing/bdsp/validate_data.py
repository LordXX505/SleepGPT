import os
import glob
import h5py
import zlib
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np
def check_hdf5_file(file_path, error_files):
    try:
        with h5py.File(file_path, 'r') as hf:
            for key in hf.keys():
                try:
                    data = hf[key][:]
                except Exception as e:
                    error_files.put(file_path)
                    print(f"Error reading dataset {key}: {e}, file_path: {file_path}")
        return True
    except Exception as e:
        error_files.put(file_path)
        print(f"Error opening file {file_path}: {e}, file_path: {file_path}")
        return False

def process_file_chunk(data_chunk, error_files):
    for item in data_chunk:
        data_path = os.path.join(item, 'data.h5')
        if os.path.isfile(data_path):
            if not check_hdf5_file(data_path, error_files):
                print(f'{item} is wrong !!!!!!')
def check_data():
    path = '/mnt/myvol/data/MGH/sub-S0001114509910/ses-1/data.h5'
    error_files = []
    check_hdf5_file(path, error_files)


if __name__ == '__main__':

    check_data()


    Root_path = '/mnt/myvol/data/MGH'
    data_path_list = glob.glob(os.path.join(Root_path, '*', '*'))
    num_processes = 16
    data_path_list_len = len(data_path_list)
    piece = data_path_list_len // num_processes

    chunks = [data_path_list[i * piece:(i + 1) * piece] for i in range(num_processes)]
    # If the total number of items is not exactly divisible by num_processes, append the remaining items to the last chunk
    if len(data_path_list) % num_processes != 0:
        chunks[-1].extend(data_path_list[num_processes * piece:])

    error_files = Queue()

    processes = []
    for i in range(num_processes):
        p = Process(target=process_file_chunk, args=(chunks[i], error_files))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Retrieve error files from the queue
    error_files_list = []
    while not error_files.empty():
        error_files_list.append(error_files.get())

    # Convert the error files list to a NumPy array
    error_files_np = np.array(error_files_list)

    # Save the NumPy array to a file
    np.save("error_files.npy", error_files_np)