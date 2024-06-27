import datetime
import numpy as np
import pandas as pd
import os
import datetime
import mne

def get_path_from_bdsp(sid, dov, base_folder=None, data_folders=None, raise_error=True):
    """
    Parameters:
    sid: str, HashID
    dov: str, date of visit (deidentified shifted date, in YYYMMDD format)

    Returns:
    signal_path: path to the signal file
    annot_path: path to the annotation file
    """

    if base_folder is None: # default:
        base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
    if data_folders is None: # default:
        data_folders = os.listdir(base_folder)
    
    if type(dov)!=str:
        dov = dov.strftime('%Y%m%d')

    signal_path = None
    annot_path = None
    for df in data_folders:
        if f'{sid}_{dov}' in df:
            data_files = os.listdir(os.path.join(base_folder, df))
            signal_path = [x for x in data_files if 'signal_' in x.lower() and x.lower().endswith('.mat')]
            annotations_path = [x for x in data_files if x.lower().endswith('_annotations.csv')]
            if len(signal_path)==1 and len(annotations_path)==1:
                signal_path = os.path.join(base_folder, df, signal_path[0])
                annot_path  = os.path.join(base_folder, df, annotations_path[0])
                break
    if raise_error:
        assert signal_path is not None and annot_path is not None, f"get_path_from_bdsp failed: sid={sid}, dov={dov}"

    return signal_path, annot_path


def get_grass_start_end_time(starttime_raw, endtime_raw):
    
    time_str_elements = starttime_raw.flatten()
    start_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
    time_str_elements = endtime_raw.flatten()
    end_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))

    start_time = start_time.split(':')
    second_elements = start_time[-1].split('.')
    start_time = datetime.datetime(1990,1,1,hour=int(float(start_time[0])), minute=int(float(start_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))
    end_time = end_time.split(':')
    second_elements = end_time[-1].split('.')
    end_time = datetime.datetime(1990,1,1,hour=int(float(end_time[0])), minute=int(float(end_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))

    return start_time, end_time


def chin_name_standardize(signal_channel_names):
    """Looks like there are different chin EMG configurations, standardize channel name here to 'chin_emg' """
    chin_channels = [x for x in signal_channel_names if 'chin' in x]
    if len(chin_channels) == 0:
        return signal_channel_names

    emg_channel = chin_channels[0]
    signal_channel_names = [x.replace(emg_channel, 'chin_emg') for x in signal_channel_names]

    return signal_channel_names


def eog_name_standardize(signal_channel_names):
    """Different/multiple EOG channels are usually available. rename one to 'eog' here """
    eog_channels = ['e2-m1', 'e1-m2', 'e2-m2', 'e1-m1']
    for eog_channel in eog_channels:
        if eog_channel in signal_channel_names:
            break

    signal_channel_names = [x.replace(eog_channel, 'eog') for x in signal_channel_names]

    return signal_channel_names

def load_bdsp_signal(path_signal):
    
    raw = mne.io.read_raw_edf(path_signal, preload=False, verbose=False)
    params = {'Fs': raw.info['sfreq']}
    ch_available = raw.ch_names
    # select only a subset of ch_available that are present in the code below:
    channel_names_to_load = [
        'ekg', 'ecg', 'ecg-la', 'ecg-v1', 'rleg+', 'rleg-', 'rat', 'lleg+', 'lleg-', 'lat', 
        'sao2', 'spo2', 'chin1', 'chin2', 'chin3', 'chin1-chin3', 'chin', 'chin1-chin2', 'ptaf', 'cflow', 'resp_airflow', 
        'e2-m1', 'e2-m2', 'e1-m2', 'e1-m1', 'c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1', 
        'c3-m1', 'c4-m2', 'f3-m1', 'f4-m2', 'o1-m1', 'o2-m2',
        'abd', 'chest', 'abdomen', 'thorax', 'r leg', 'l leg',
        'f3', 'f4', 'c3', 'c4', 'o1', 'o2', 'e1', 'e2', 'm1', 'm2',
        'f3-avg', 'f4-avg', 'c3-avg', 'c4-avg', 'o1-avg', 'o2-avg', 'e1-avg', 'e2-avg',
    ]

    # select only a subset of ch_available that are present in the code below:
    channel_names_to_load = [x for x in ch_available if x.lower() in channel_names_to_load]
    raw.pick_channels(channel_names_to_load)
    raw.load_data(verbose=False)
    signal = raw.get_data().T
    signal = signal * 1e6 # to correct units (uV in EEG, but "normal/correct" for others, seems to be saved as V unit.)
    signal = pd.DataFrame(signal, columns=[x.lower() for x in channel_names_to_load])
    
    return signal, params



def annotations_preprocess(annotations, fs, t0=None, impute_duration_column=True, return_quality=False, verbose=False):
    """
    input: dataframe annotations.csv
    fs: sampling rate
    t0: start datetime from signal file, if None, from the first line of annotations
    impute_duration_column: if duration column is missing, assume sleep stage epoch=30s, resp event duration=10s
    output: dataframe annotations with new columns: event starts/ends in seconds and ends
    """
    # deal with duration

    quality = 1
    need_to_impute_duration = False
    if 'duration' not in annotations.columns: # sometimes duration column is missing
        need_to_impute_duration = True
    elif all(pd.isna(annotations.duration)): # sometimes duration column is all NaN
        need_to_impute_duration = True
    annotations['event'] = annotations.event.astype(str)
    if need_to_impute_duration & impute_duration_column:
        # assume the following: sleep stage epoch=30 seconds, resp event = 10 seconds.
        annotations['duration'] = np.nan
        annotations.loc[annotations.event.str.contains('sleep',case=False)&annotations.event.str.contains('stage',case=False), 'duration'] = 30
        annotations.loc[annotations.event.str.contains('(?:resp|pnea|rera)',case=False), 'duration'] = 10
    annotations.loc[pd.isna(annotations.duration), 'duration'] = 1/fs
    #annotations['duration'].apply(lambda x: datetime.timedelta(seconds=x))

    # deal with time
    # assumes time ascending order

    if 'Recording Resumed' in list(annotations.event.values):
        if verbose:
            print('Recording Resumed found in annotations. Quality 0.5')
        quality = 0.5 # looking at 10 sample files with those, around half of the files have some time shift between signals and annotations.

    annotations = annotations[pd.notna(annotations.time)].reset_index(drop=True)

    if pd.isna(annotations.epoch).sum() > 0:
        if verbose:
            print('NaN epoch will be removed from annotations:')
            print(annotations.loc[pd.isna(annotations.epoch)])
        annotations = annotations[pd.notna(annotations.epoch)].reset_index(drop=True)

    if any(annotations.epoch < 1):
        annotations = annotations[annotations.epoch >= 1].reset_index(drop=True)

    if t0 is None:
        annotations['time'] = pd.to_datetime(annotations['time']) # but there is no date, use today
        t0 = annotations.time.iloc[0]
    else:
        t0 = pd.to_datetime(t0)  # make sure it is datetime
        annotations['time'] = pd.to_datetime(t0.strftime('%Y-%m-%d ')+annotations['time'].astype(str), format='%Y-%m-%d %H:%M:%S')

    # deal with midnight
    midnight_idx = np.where((annotations.time.dt.hour.values[:-1]==23)&(annotations.time.dt.hour.values[1:]==0))[0]
    if len(midnight_idx)==1:
        annotations.loc[midnight_idx[0]+1:, 'time'] += datetime.timedelta(seconds=86400)
    elif len(midnight_idx)>1:
        raise ValueError('More than 1 midnight point?')

    annotations = annotations.loc[annotations.time >= t0]

    # add columns
    annotations['dt_start'] = (annotations['time'] - t0).dt.total_seconds()
    annotations['dt_end'] = annotations['dt_start'] + annotations['duration']
    annotations['idx_start'] = np.round(annotations['dt_start'] * fs).astype(int)
    annotations['idx_end'] = annotations['idx_start'] + np.round(annotations['duration'] * fs).astype(int)

    if return_quality:
        return annotations, quality

    return annotations


def vectorization(events_annotations_selection, mapping, signal_len):
    """
    Inputs: 
    events_annotations_selection: dataframe (annotations.csv), only rows selected that shall be vectorized.
    mapping: dataframe defining the vectorization mapping.
    Output: 1D numpy array, vectorized annotations
    """
    
    events_vectorized = np.zeros(signal_len, )
    for jloc, row in events_annotations_selection.iterrows():
        for event_type in mapping.index:
            keyword = mapping.loc[event_type, 'keyword']
            if keyword.lower() in row.event.lower():
                value = mapping.loc[event_type, 'value']
                events_vectorized[row['idx_start'] : row['idx_end']] = value
                break # event saved, proceed with next.
                
    return events_vectorized


def vectorize_respiratory_events(annotations, signal_len):
    """
    Input: annotations.csv as dataframe
    Output: vectorized respiratory array (OA: 1, CA: 2, MA: 3, HY: 4, RA: 5)
    """
    
    # definition of the following categories of respiratory event, their keyword and the vectorization-mapping:
    keyword_value_pairs = np.array([['obstructive', 1],
                                   ['central', 2],
                                   ['mixed', 3],
                                   ['hypopnea', 4],
                                   ['rera', 5],
                                   ])

    mapping = pd.DataFrame(index = ['OA', 'CA', 'MA', 'HY', 'RA'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))
    
    resp_events = annotations.loc[annotations.event.str.lower().apply(lambda x: (('resp' in x) & ('event' in x)) | ('apnea' in x) | ('hypopnea' in x) | ('rera' in x)), :].copy()

    resp_events_vectorized = vectorization(resp_events, mapping, signal_len)
    
    return resp_events_vectorized


def vectorize_sleep_stages(annotations, signal_len, noscore_fill=np.nan):
    """
    Input: annotations.csv as dataframe
    Output: vectorized sleepstage array (W: 5, R: 4, N1: 3, N2: 2, N3: 1)
    """
    
    # definition of the following categories of respiratory event, their keyword and the vectorization-mapping:
    keyword_value_pairs = np.array([['3', 1],
                                   ['2', 2],
                                   ['1', 3],
                                   ['r', 4],
                                   ['w', 5],
                                   ])

    mapping = pd.DataFrame(index = ['N3', 'N2', 'N1', 'R', 'W'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))
    
    sleep_stages = annotations.loc[annotations.event.apply(lambda x: 'sleep_stage' in str(x).lower()), :].copy()

    sleep_stages_vectorized = vectorization(sleep_stages, mapping, signal_len)
    sleep_stages_vectorized[sleep_stages_vectorized == 0] = noscore_fill # set no-scored sleep stage to NaN instead of 0.

    return sleep_stages_vectorized


def vectorize_arousals(annotations, signal_len):
    """
    Input: annotations.csv as dataframe
    Output: vectorized sleepstage array (Non-arousal: 0, arousal: 1 )
    """
    
    # definition of the following categories of respiratory event, their keyword and the vectorization-mapping:
    keyword_value_pairs = np.array([['arousal', 1],
                                   ])

    mapping = pd.DataFrame(index = ['arousal'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))
    
    arousal_events = annotations.loc[annotations.event.str.lower().apply(lambda x: ('arousal' in x) & ('post' not in x)), :].copy()
    
    arousal_events_vectorized = vectorization(arousal_events, mapping, signal_len)

    return arousal_events_vectorized



def vectorize_limb_movements(annotations, signal_len):
    """
    Input: annotations.csv as dataframe
    Output: vectorized limb movements array (Isolated: 1, Periodic: 2, Arousal PLM: 3, "Limb Movement" (Natus): 4).
    Note: Grass files have "isolated", "periodic" and "arousal plm" categories, while Natus file have "Limb Movement" annotations only.
    """

    keyword_value_pairs = np.array([['isolated', 1],
                               ['periodic', 2],
                               ['arousal', 3],
                               ['limb', 4],
                               ])

    mapping = pd.DataFrame(index = ['isolated', 'periodic', 'arousal', 'limb'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))

    limb_events = annotations.loc[annotations.event.str.lower().apply(lambda x: ('plm' in x) | ('limb' in x)), :].copy()

    limb_events_vectorized = vectorization(limb_events, mapping, signal_len)
    
    return limb_events_vectorized


def vectorize_body_position(annotations, signal_len, fs, epoch_len_sec=30):
    """
    Input: 
    annotations: annotations.csv as dataframe
    signal_len: length of signal array (in samples)
    fs: sampling frequency
    epoch_len_sec = number of seconds per epoch
    Output: vectorized body position array
    """
    
    position_grass = {'Position - Supine': 1, 
                      'Position - Supine - 1': 1, 
                      'Position - Left': 2,
                      'Position - Left - 1': 2,
                      'Position - Right': 3, 
                      'Position - Right - 1': 3,
                      'Position - Prone': 4,
                      'Position - Prone - 1': 4,
                      'Position - Sitting': 5,
                      'Position - Sitting - 1': 5,
                      'Position - Disconnect': 6,
                      'Position - Disconnect - 1': 6}
    position_natus = {'Body_Position:_Supine': 1, 
                      'Body_Position:_Left': 2,
                      'Body_Position:_Right': 3, 
                      'Body_Position:_Prone': 4, 
                      'Body_Position:_Upright': 5,
                     }
    position_mapping = position_grass.copy()
    position_mapping.update(position_natus)
    
    len_epoch = int(epoch_len_sec * fs)
    body_position = np.zeros(signal_len,)
    annotations_position = annotations[np.isin(annotations.event, list(position_mapping.keys()))].reset_index(drop=True) # annotations filtered for any body position keyword

    for jloc, row in annotations_position.iterrows():
        if jloc == annotations_position.index[-1]:
            body_position[int(row.epoch * len_epoch) : ] = position_mapping[row.event]
        else:
            body_position[int(row.epoch * len_epoch) : int(annotations_position.loc[jloc+1, 'epoch'] * len_epoch)] = position_mapping[row.event]

    return body_position
