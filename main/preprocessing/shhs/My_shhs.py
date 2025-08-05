import os
from multiprocessing import Process, current_process
from multiprocessing import Pool
from tqdm import tqdm
import pyarrow as pa
import gc
import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import xml.etree.ElementTree as ET
import numpy as np
import mne
from threading import Thread
import multiprocessing
import time
import glob as glob
from sklearn.linear_model import LinearRegression

bads_idx = {
    'C3': 4,
    'C4': 5,
    'ECG': 15,
    'EMG1': 16,
    'EOG1': 18
}
# self.choose_channels = np.array([4, 5, 15, 16, 18, 22, 23, 36, 38, 39, 52])
# [C3, C4, ECG, EMG, EOG, F3, F4, Fpz, O1, O2, Pz]
def process_shhs(idx, path_list, anno_list):
    print(len(path_list), len(anno_list))
    assert len(path_list) == len(anno_list)
    mne.cuda.set_cuda_device(idx)
    print_ = False
    sucess = []
    wrong = []
    for i in tqdm(range(len(anno_list))):
        data_path = path_list[i]
        anno_path = anno_list[i]
        name = os.path.basename(data_path).split('.')[0]
        try:
            print(f"********************starting {os.path.basename(data_path).split('.')[0]}********************")
            assert os.path.basename(data_path).split('.')[0] in os.path.basename(anno_path).split('.')[0]

            labels = []
            # Read annotation and its header
            t = ET.parse(anno_path)
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
                print("============================== Faulty file ==================")
                continue
            ########################################################################################################

            raw = mne.io.read_raw_edf(data_path)
            channle_names = ['EEG2', 'EEG 2', 'EEG(SEC)', 'EEG sec', 'EEG(sec)']
            EEG2_name = ""
            for i in channle_names:
                if i in raw.ch_names:
                    EEG2_name = i
                    break

            raw.rename_channels({'EEG': 'C4', EEG2_name: 'C3', 'EMG': 'EMG1', 'EOG(L)': 'EOG1', 'EOG(R)': 'EOG2'})

            raw.pick(['C3', 'C4', 'EMG1', 'EOG1', 'ECG'])
            new_ch_names = sorted(raw.ch_names)
            raw = raw.reorder_channels(new_ch_names)
            raw.resample(100)  # down sample to 100hz
            raw.filter(l_freq=0.3, h_freq=35, n_jobs='cuda', method='fir')
            bads = raw.info['bads']
            badsidx = [bads_idx[_] for _ in bads]
            badsidx = sorted(badsidx)
            print(f'{raw.info["bads"]}, idx: {badsidx}')
            labels = np.asarray(labels)

            # Remove movement and unknown stages if any
            data = raw.get_data()
            if print_ is False:
                print(raw.ch_names)

            n_epochs = data.shape[1] / (30 * 100)
            if data.shape[1] % (30 * 100) != 0:
                raise Exception("Something wrong")
            x = np.asarray(np.split(data, n_epochs, axis=1)).astype(np.float32)
            y = labels.astype(np.int32)

            assert len(x) == len(y)
            w_edge_mins = 30
            nw_idx = np.where(y != 0)[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx + 1)
            x = x[select_idx]
            y = y[select_idx]
            filename = '/data/data/shhs_new'
            cnt = 0
            for _x, _y in zip(x, y):
                LR = LinearRegression()
                for c in [0,1]:
                    LR.fit(_x[2].reshape(-1, 1), _x[c])
                    _x[c] -= _x[2]*LR.coef_[0] + LR.intercept_
                dataframe = pd.DataFrame(
                        {'x': [_x.tolist()], 'stage': _y, 'bads': [bads]}
                    )
                table = pa.Table.from_pandas(dataframe)
                os.makedirs(f"{filename}/{name}", exist_ok=True)
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
            print_ = True
        except Exception as e:
            print(f'!!!Wrong!!!------------{e}----{name}')
            wrong.append(name)
    return (sucess, wrong)

if __name__ == '__main__':
    procs = []
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    p = Pool(16)
    Root_path = '/data/shhs/polysomnography/edfs'
    shhs1_root_path = os.path.join(Root_path, 'shhs1')
    shhs2_root_path = os.path.join(Root_path, 'shhs2')
    shhs1_data_path_list = sorted(glob.glob(shhs1_root_path+'/*'))
    shhs2_data_path_list = sorted(glob.glob(shhs2_root_path+'/*'))
    all_data_path_list = shhs1_data_path_list
    anna_root_path = '/data/shhs/anno/polysomnography'
    anna_shhs1_path = os.path.join(anna_root_path, 'shhs1')
    anna_shhs2_path = os.path.join(anna_root_path, 'shhs2')
    anna_shhs1_path_list = sorted(glob.glob(anna_shhs1_path+'/*'))
    anna_shhs2_path_list = sorted(glob.glob(anna_shhs2_path+'/*'))
    all_anno_path_list = anna_shhs1_path_list
    # all_anno_path_list = all_anno_path_list[5800:]
    # all_data_path_list = shhs2_data_path_list[2500:]
    # all_anno_path_list = anna_shhs2_path_list[2500:]
    # result = process_shhs(1, all_data_path_list, all_anno_path_list)
    # print(result)
    result = []
    #
    piece = len(all_data_path_list)//32
    for i in range(32):
        start = i*piece
        if i == 32:
            end = len(all_data_path_list)
        else:
            end = min((i+1)*piece, len(all_data_path_list))
        print(f'start {start}, end {end}')
        # process_shhs(i, all_data_path_list[start:end], all_anno_path_list[start:end])
        result.append(p.apply_async(process_shhs, args=(i%8, all_data_path_list[start:end], all_anno_path_list[start:end])))
        if end == len(all_data_path_list):
            break
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    k = ""
    for res in result:
        suc = res.get()
        # suc=res
        for _ in suc[0]:
            k += _
            k += "\n"
        print(suc[1])
    with open('./shhs_log', 'w') as f:
        f.write(k)
    print('write to shhs_log')
#     end_idx = nw_idx[-1] + (w_edge_mins * 2)
#         if start_idx < 0: start_idx = 0
#         if end_idx >= len(y): end_idx = len(y) - 1
#         select_idx = np.arange(start_idx, end_idx + 1)
#         x = x[select_idx]
#         y = y[select_idx]
#
#         cnt = 0
#
#         for _x, _y in zip(x, y):
#             dataframe = pd.DataFrame(
#                     {'x': [_x.tolist()], 'stage': _y, 'bads': [bads],
#                      'good': [good_channels.tolist()]}
#                 )
#             table = pa.Table.from_pandas(dataframe)
#             with pa.OSFile(
#                         f"{filename}/{name}/{str(cnt).zfill(5)}.arrow", "wb"
#                 ) as sink:
#                     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
#                         writer.write_table(table)
#             cnt += 1
#             del dataframe
#             del table
#             gc.collect()
#         del raw
#         gc.collect()
#         torch.cuda.empty_cache()
#         with open(f"{success_file}", 'w') as f:
#             f.write('success')
#         sucess.append(name)
#         print_ = True
#
#     logger.info("Logging channel sums and counts:")
#     for ch_name in channel_sum_dict.keys():
#         logger.info(
#             f"Channel {ch_name} - Sum Mean: {channel_sum_dict[ch_name]}, Numbers: {channel_count_dict[ch_name]}")
#     end_time = time.time()
#     logger.info(f"Process {process_idx} finished. Total time: {end_time - start_time} seconds.")
#     return (sucess, wrong, channel_sum_dict, channel_count_dict)
#
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python process_shhs.py <process_id>")
#         sys.exit(1)
#     procs = []
#     procs_number = 64
#     Root_path = '/data/shhs_raw/polysomnography/edfs'
#     shhs1_root_path = os.path.join(Root_path, 'shhs1')
#     shhs2_root_path = os.path.join(Root_path, 'shhs2')
#     shhs1_data_path_list = sorted(glob.glob(shhs1_root_path+'/*'))
#     shhs2_data_path_list = sorted(glob.glob(shhs2_root_path+'/*'))
#     all_data_path_list = shhs1_data_path_list + shhs2_data_path_list
#     print(len(all_data_path_list))
#     anna_root_path = '/data/shhs_raw/polysomnography/annotations-events-profusion'
#     anna_shhs1_path = os.path.join(anna_root_path, 'shhs1')
#     anna_shhs2_path = os.path.join(anna_root_path, 'shhs2')
#     anna_shhs1_path_list = sorted(glob.glob(anna_shhs1_path+'/*'))
#     anna_shhs2_path_list = sorted(glob.glob(anna_shhs2_path+'/*'))
#     all_anno_path_list = anna_shhs1_path_list + anna_shhs2_path_list
#     # all_anno_path_list = all_anno_path_list[5800:]
#     # all_data_path_list = shhs2_data_path_list[2500:]
#     # all_anno_path_list = anna_shhs2_path_list[2500:]
#     # result = process_shhs(1, all_data_path_list, all_anno_path_list)
#     # print(result)
#     result = []
#     piece = len(all_data_path_list)//procs_number + 1
#     print(piece)
#     # p = Pool(procs_number)
#     process_id = int(sys.argv[1])
#     start = process_id * piece
#     end = min((process_id+1)*piece, len(all_data_path_list))
#     process_shhs(process_id%8, all_data_path_list[start:end], all_anno_path_list[start:end], process_id)
#     # for i in range(procs_number):
#     #     start = i*piece
#     #     if i == procs_number-1:
#     #         end = len(all_data_path_list)
#     #     else:
#     #         end = min((i+1)*piece, len(all_data_path_list))
#     #     print(f'start {start}, end {end}')
#     #     # process_shhs(i, all_data_path_list[start:end], all_anno_path_list[start:end])
#     #     result.append(p.apply_async(process_shhs, args=(i%8, all_data_path_list[start:end], all_anno_path_list[start:end], i)))
#     #     if end == len(all_data_path_list):
#     #         break
#     # print('Waiting for all subprocesses done...')
#     # p.close()
#     # p.join()
#     # print('All subprocesses done.')
#     k = ""
#     for index, res in enumerate(result):
#         suc = res.get()
#         # suc=res
#         for _ in suc[0]:
#             k += _
#             k += "\n"
#         print(suc[1])
#     combined_channel_sum_dict = {}
#     combined_channel_count_dict = {}
#     for res in result:
#         suc, _, channel_sum_dict, channel_count_dict = res.get()
#         for ch_name in channel_sum_dict:
#             if ch_name not in combined_channel_sum_dict:
#                 combined_channel_sum_dict[ch_name] = 0
#                 combined_channel_count_dict[ch_name] = 0
#             combined_channel_sum_dict[ch_name] += channel_sum_dict[ch_name]
#             combined_channel_count_dict[ch_name] += channel_count_dict[ch_name]
#     os.makedirs('./log/{process_id}', exist_ok=True)
#     with open(f'./log/{process_id}/shhs_log', 'f') as f:
#         for ch_name in combined_channel_sum_dict.keys():
#             log_msg = f"Channel {ch_name} - Sum Mean: {combined_channel_sum_dict[ch_name]}, Numbers: {combined_channel_count_dict[ch_name]}"
#             print(log_msg)
#             f.write(log_msg + "\n")
#
#     print('write to shhs_log')
#     with open(f'./log/{process_id}/shhs_true', 'f') as f:
#         f.write(k)
#     print('write to shhs_log')