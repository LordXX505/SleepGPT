import os
import pandas as pd
import numpy as np
import torch
import mne
from logger import logger as logger
from tqdm import tqdm
import os, glob, functools
import pyarrow as pa
import gc
def modify_to_path(values):
    """Get raw data file path.
    :param values: A dict contains: Final_ID	No. of sleep sessions		Name		project_ID
    :return: raw_path, stages_path, spindles_path, project_name
    """
    def cmp_case(s1, s2):
        if len(s1) < len(s2):
            return -1
        elif len(s1) > len(s2):
            return 1
        else:
            if s1 < s2:
                return -1
            elif s1 == s2:
                return 0
            else:
                return 1
    root_path = '/nd_disk1/weixuan/EEG_preprocessed'
    res = []
    stages = []
    spindles = []
    project_name = []
    if values['project_ID'] == 'Young':
        pass
        suffix = '{prefix}_sleep{num}.txt'
        suffix2 = '{prefix}_Sleep_{num}_SP.txt'
        path = os.path.join(root_path, "Youngdataexport", "sleep_eeg_" + values['Name']+'_forlabel.vhdr')
        if not os.path.exists(path):
            path = os.path.join(root_path, "Youngdataexport", "Sleep_eeg_" + values['Name']+'_forlabel.vhdr')
        if not os.path.exists(path):
             path = os.path.join(root_path, "Youngdataexport", "Sleep_eeg_" + values['Name']+'_sleep_forlabel.vhdr')
        res.append([path])
        suffix = suffix.format(prefix=str(values['Final_ID']).strip(), num='1')
        stage = os.path.join(root_path, 'stages', suffix)
        stages.append(stage)
        suffix2 = suffix2.format(prefix=str(values['Final_ID']).strip(), num='1')
        spindle = os.path.join(root_path, 'events', suffix2)
        spindles.append(spindle)
        project_name.append(values['Final_ID'])
    else:
        index = int(values['No. of sleep sessions'])
        path = glob.glob(os.path.join(root_path,"segment_for_data_share") + '/' + values['project_ID']+'*.vhdr')
        path = sorted(path,  key=functools.cmp_to_key(cmp_case))
        res.append(path)
        for i in range(1, index+1):
            suffix = '{prefix}_sleep{num}.txt'
            suffix2 = '{prefix}_Sleep_{num}_SP.txt'
            suffix = suffix.format(prefix=str(values['Final_ID']).strip(), num=str(i))
            stage = os.path.join(root_path, 'stages', suffix)
            stages.append(stage)
            if i== 1 and (str(values['Final_ID']).strip() == "sub3019" or str(values['Final_ID']).strip() == "sub3039"):
                suffix_special = '{prefix}_Sleep_{num}_SPc.txt'
                spindle = os.path.join(root_path, 'events',
                                       suffix_special.format(prefix=str(values['Final_ID']).strip(), num=str(i)))
            else:
                suffix2 = suffix2.format(prefix=str(values['Final_ID']).strip(), num=str(i))
                spindle = os.path.join(root_path, 'events', suffix2)

            spindles.append(spindle)
        project_name.append(values['Final_ID'])
    return res, stages, spindles, project_name

def get_orig_time(i, project_name):
    """Get orig time
    EEG data of the first 97*30 seconds for sub3019_Sleep1 should be delted to match stages, events, and fMRI
    EEG data of the first 70*30 seconds for sub3039 should be delted to match stages, events, and fMRI
    :param i: int. First night or not, That is to say, 0 or not
    :param project_name: str. project_name
    :return: orig_time: int.
    """
    if i > 0:
        return 0
    if project_name == 'sub3019':
        orig_time = 97 * 30
    elif project_name == 'sub3039':
        orig_time = 70 * 30
    else:
        orig_time = 0
    return orig_time

def get_annotations_spindle(spindles, project_name, tmax, *args, **kwargs):
    """Get annotations of spindles
    Read the data from a pandas dataframe.
    The file of spindles is a txt like:
    onset   endtime   duration
    760.5    761.5      1.0
    .....    .....      ....

    :param spindles: Path.
    :param project_name: str.
    :param args: args.
    :param kwargs: kwargs.
    :return:
    """
    anno_res = []
    logger.info('spindle_path:{}'.format(spindles))
    cnt = 0
    for spindle_path in tqdm(spindles):
        spindle = pd.read_table(spindle_path, names=['onset', 'endtime', 'duration'], sep='\s+')
        onset_list = []
        duration_list = []
        description_list = []
        for i in spindle.index:
            row = spindle.iloc[i]
            # if row['onset'] > tmax[cnt]:
            #     break
            onset_list.append(row['onset'])
            duration_list.append(row['duration'])
            description_list.append('spindle')
        cnt += 1
        logger.info('{project_name}: number of annotations spindle {len}'.format(project_name=project_name, len = len(onset_list)))
        anno = mne.Annotations(onset=onset_list, duration=duration_list, description=description_list)
        anno_res.append(anno)
    return anno_res

def get_annotations_stage(stages, project_name, tmax, *args, **kwargs):
    """
    Get annotations of stage

    Read the data from a pandas dataframe.
    The file of spindles is a txt like:
    label
      0
      1
      0
      2
      3
    0 is Wake.
    1 is stage N1.
    2 is stage N2.
    3 is stage N3.
    4 is Rem.
    EEG data of the first 97*30 seconds for sub3019_Sleep1 should be delted to match stages, events, and fMRI
    EEG data of the first 70*30 seconds for sub3039 should be delted to match stages, events, and fMRI

    :param stages:
    :param project_name:
    :param args:
    :param kwargs:
    :return:
    """
    logger.info('stage_path:{}'.format(stages))
    stage_res = []
    cnt = 0
    for stage_path in tqdm(stages):
        if project_name == 'sub3019' and len(stage_res) == 0:  # only the first night need subtract the offset.
            seconds = 97*30
        elif project_name == 'sub3039' and len(stage_res) ==0:
            seconds = 70*30
        else:
            seconds = 0
        stage = pd.read_table(stage_path, names=['label'], sep='\s+')
        onset_list = []
        duration_list = []
        description_list = []
        for i in stage.index:
            # if seconds > tmax[cnt]:
            #     break
            onset_list.append(seconds)
            duration_list.append(30-1/100)  # The final eeg will down sample to 100hz.
            desp = stage.iloc[i]['label']  # Use int 0,1,2,3,4 to identify the stage.
            description_list.append(desp)
            seconds += 30
        anno = mne.Annotations(onset=onset_list, duration=duration_list, description=description_list)
        stage_res.append(anno)
        cnt += 1
    return stage_res

def get_raws(raws, project_name):
    """
    Get the raw data.
    Raw data need subtract the first seconds to match annotations and fMRI.
    EEG data of the first 97*30 seconds for sub3019_Sleep1 should be delted to match stages, events, and fMRI
    EEG data of the first 70*30 seconds for sub3039 should be delted to match stages, events, and fMRI
    Different dataset has different channels.
    SD EEG: N = 105, 56 EEG, 3 EMG (EMG1-EMG3, EMG2-EMG3), 2 refs, 2 EOG, 1 ECG
    Young EEG: N =33,  57 EEG, 2 EMG, 2 refs, 2 EOG, 1 ECG
    :param raws: Path. raw paths
    :param project_name: Str. project name
    :return: raw eeg.
    """
    raw_res = []
    logger.info('raws_path:{}'.format(raws))
    tmax = []
    for raw_path in tqdm(raws):
        raw = mne.io.read_raw_brainvision(raw_path, scale=1e6)
        tmax.append(raw.times[-1])
        raw.verbose = True
        if int(project_name[-4:])>1036:  # sub1036 is the last one in Young dataset.
            logger.info('M1, M2 **** drop_channels: {}'.format(project_name))
            raw.drop_channels(['FT10', 'FT7', 'FT8', 'FT9', 'M1', 'M2'], on_missing='warn')  # drop the channel to match
            raw.rename_channels({'E1': 'EOG1', 'E2': 'EOG2'})
        else:
            logger.info('A1, A2 **** drop_channels: {}'.format(project_name))
            raw.drop_channels(['A1', 'A2', 'AF7', 'AF8', 'CPz', 'PO7', 'PO8', 'FCz'], on_missing='warn')

        if project_name == 'sub3019' and len(raw_res)==0:
            logger.info('project_name: {project_name}****Original time index:{index}'.format(project_name=project_name, index=raw.n_times))
            raw.crop(tmin=97*30)
            logger.info('project_name: {project_name}****Now time index:{index}'.format(project_name=project_name, index=raw.n_times))
        elif project_name == 'sub3039' and len(raw_res)==0:
            logger.info('project_name: {project_name}****Original time index:{index}'.format(project_name=project_name, index=raw.n_times))
            raw.crop(tmin=70*30)
            logger.info('project_name: {project_name}****Now time index:{index}'.format(project_name=project_name, index=raw.n_times))
        new_ch_names = sorted(raw.ch_names)
        raw = raw.reorder_channels(new_ch_names)
        raw.resample(100)  # down sample to 100hz
        raw_res.append(raw)
    return raw_res, tmax

def get_data_label(raw_list, stage_list, project_name, max_len=30*100):
    """
    Get the final data and labels.
    Change it to pyarrow.
    :param raw_list: List. The cropped raw eeg data.
    :param stage_list: List. The original stage data.
    :param project_name: str.
    :param max_len: int.The length of one epoch.
    :return: None
    """
    epochs = []
    spindle_label = []
    channel_label = []
    stage_label = []
    for _ in tqdm(range(len(raw_list))):
        item = raw_list[_]
        stage = stage_list.description[_]

        stage_label.append(int(float(stage)))
        if int(project_name[-4:])>1036: # SD EEG: N = 105, 56 EEG, 3 EMG (EMG1-EMG3, EMG2-EMG3), 2 refs, 2 EOG, 1 ECG
            def func(array, EMG3):
                array -= EMG3[0]
                return array
            data, times = item['EMG3']
            item.apply_function(func, picks=['EMG1', 'EMG2'], EMG3=data)
            item.drop_channels(['EMG3'])  # drop channel EMG3
        channels = item.ch_names  # save channels
        new_channels = np.argsort(channels)
        channels = np.sort(channels)
        channel_label.append(channels)
        data = np.array(item.get_data())  # save data
        data = data[new_channels]
        epochs.append(data[:max_len])

        annotations = item.annotations
        spindle_label.append(np.zeros(max_len))
        for annot in annotations:
            onset = annot["onset"] - item.first_time
            # be careful about near-zero errors (crop is very picky about this,
            # e.g., -1e-8 is an error)
            if -item.info['sfreq'] / 2 < onset < 0:
                onset = 0
            end = onset + annot["duration"]
            start_idx = item.time_as_index(onset)[0]  # return an array
            end_idx = min(max_len - 1, item.time_as_index(end)[0])  # this will lead length limit exceeded
            for i in range(start_idx, end_idx + 1):
                spindle_label[-1][i] = 1

    return epochs, spindle_label, channel_label, stage_label

def main():
    """
    Main function to preprocess EEG data.
    EEG file:vhdr, eeg, vmrk
    Spindles: txt. Start time, End time, duration. unit: 1s
    Stage: 0,1,2,3,4. unit: 30s
    channels = ['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4','CP5'
     ,'CP6' ,'Cz' ,'ECG' ,'EMG1' ,'EMG2' ,'EOG1' ,'EOG2' ,'F1' ,'F2' ,'F3' ,'F4' ,'F5'
     ,'F6' ,'F7' ,'F8' ,'FC1' ,'FC2' ,'FC3' ,'FC4' ,'FC5' ,'FC6' ,'Fp1' ,'Fp2' ,'Fpz' ,'Fz'
     ,'O1' ,'O2' ,'Oz' ,'P1' ,'P2' ,'P3' ,'P4' ,'P5' ,'P6' ,'P7' ,'P8' ,'PO3' ,'PO4' ,'POz'
     ,'Pz' ,'T7' ,'T8' ,'TP7' ,'TP8']
    :return:
    pyarrow = {
            "x": x,   # [epochs * channels * samples]
            "Spindle_label": Spindle_lable,   # Spindle label [epochs * samples]
            "Stage_label": Stage_label,  # Stage_label [epochs]
            "fs": sampling_rate,    # 100
            "ch_label": select_ch,  # 'Fpz, Oz....'
            "start_datetime": start_datetime,  # datetime.datetime(1989, 4, 24, 16, 13)
            "file_duration": file_duration,  # max time
            "epoch_duration": epoch_duration,  # 30.0
            "n_epochs": len(x),  # epochs
        }
    """
    print(os.getcwd())
    all_channel = ['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4','CP5'
     ,'CP6' ,'Cz' ,'ECG' ,'EMG1' ,'EMG2' ,'EOG1' ,'EOG2' ,'F1' ,'F2' ,'F3' ,'F4' ,'F5'
     ,'F6' ,'F7' ,'F8' ,'FC1' ,'FC2' ,'FC3' ,'FC4' ,'FC5' ,'FC6' ,'Fp1' ,'Fp2' ,'Fpz' ,'Fz'
     ,'O1' ,'O2' ,'Oz' ,'P1' ,'P2' ,'P3' ,'P4' ,'P5' ,'P6' ,'P7' ,'P8' ,'PO3' ,'PO4' ,'POz'
     ,'Pz' ,'T7' ,'T8' ,'TP7' ,'TP8']
    excel_path = '/nd_disk1/weixuan/EEG_preprocessed/EEG_ID_info.xlsx'
    df = pd.read_excel(excel_path)
    dic = {}
    for i in df.index:
        value_i = df.iloc[i]
        try:
            dic[value_i['Name']] = modify_to_path(value_i)
        except Exception as e:
            print('Exception:', e)
            # print(i, value_i)
    error = []
    for key in tqdm(dic.keys()):
        # sub3043
        value = dic[key]
        raws = value[0][0]
        stages = value[1]
        spindles = value[2]
        project_name = value[3][0]

        if int(project_name[-4:])<3043:
            continue
        dataset_root = os.path.join('../data', 'SD' if int(project_name[-4:])>1036 else 'Young')  # make dataset root path
        raw_res, tmax = get_raws(raws, project_name)

        spindle_res = get_annotations_spindle(spindles, project_name, tmax)
        stage_res = get_annotations_stage(stages, project_name, tmax)
        logger.info('****project_name: {project_name} '
                    '|||| All-----spindle_res: {spindle_res}, stage_res: {stage_res}|||| raw_res: {raw_res}'.
                    format(project_name=project_name, spindle_res=len(spindle_res), stage_res=len(stage_res), raw_res=len(raw_res)))
        _epochs, _spindle_label, _channel_label, _stage_label = [], [], [], []
        try:
            for i in tqdm(range(len(raw_res))):
                raw = raw_res[i]
                spindle_anna = spindle_res[i]
                stage_anna = stage_res[i]
                raw.set_annotations(spindle_anna)
                logger.info("number of annotations{}".format(raw.annotations))
                raw_list = raw.crop_by_annotations(stage_anna, verbose=True)
                epochs, spindle_label, channel_label, stage_label = get_data_label(raw_list, stage_anna, project_name)
                _epochs.append(epochs)
                _spindle_label.append(spindle_label)
                _channel_label.append(channel_label)
                _stage_label.append(stage_label)
            _epochs = np.concatenate(_epochs, axis=0).tolist()
            _spindle_label = np.concatenate(_spindle_label, axis=0).tolist()
            _channel_label = np.concatenate(_channel_label, axis=0).tolist()
            _stage_label = np.concatenate(_stage_label, axis=0).tolist()
            logger.info('_epochs:{}, _spindle_label:{}, _channel_label:{}, _stage_label:{}'.format(len(_epochs),len(_spindle_label),len(_channel_label)
                                                                                                  ,len(_stage_label)))
            assert len(_epochs) == len(_stage_label)
            for item in _channel_label:
                assert len(all_channel) == len(item)
                for a, b in zip(all_channel, item):
                    if a!=b:
                        raise RuntimeError

            save_dict = {
                        "x": _epochs,   # epochs * [channels * samples]
                        "Spindle_label": _spindle_label,   # Spindle label  epochs * [channels * samples]
                        "Stage_label": _stage_label,  # Stage_label [epochs]
                        "fs": 100,    # 100
                        "start_datetime": 0,  # datetime.datetime(1989, 4, 24, 16, 13)
                        "file_duration": len(_epochs)*30,  # max time
                        "epoch_duration": 30,  # 30.0
                        "n_epochs": len(_epochs),  # epochs
                    }
            dataframe = pd.DataFrame(save_dict)
            table = pa.Table.from_pandas(dataframe)
            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                    f"{dataset_root}/{project_name}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del save_dict
            gc.collect()
        except Exception as e:
            logger.warning('*************************************************')
            print(e)
            print(project_name)
            error.append(project_name)
    logger.info('Completed. Errors: {}'.format(len(error)))
    print(error)
    np.save('/home/weixuan/Sleep/Sleep/',error)
# save_dict = {
#             "x": x,   # [channels * samples]
#             "Spindle_label": Spindle_lable,   # Spindle label
#             "Stage_label": Stage_label,  # Stage_label
#             "fs": sampling_rate,    # 100
#             "ch_label": select_ch,  # 'Fpz, Oz....'
#             "start_datetime": start_datetime,  # datetime.datetime(1989, 4, 24, 16, 13)
#             "file_duration": file_duration,  # max time
#             "epoch_duration": epoch_duration,  # 30.0
#             "n_epochs": len(x),  # epochs
#         }

if __name__ == '__main__':

    main()