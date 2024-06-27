# High-Level Summary
# This script is designed to process sleep study data, including signal data from a .edf file and annotations from a .csv file. It loads the signal data, preprocesses the annotations, and then vectorizes various types of events (respiratory events, sleep stages, arousals, and limb movements) to facilitate further analysis.

import pandas as pd
from bdsp_sleep_functions import load_bdsp_signal, annotations_preprocess, vectorize_respiratory_events, vectorize_sleep_stages, vectorize_arousals, vectorize_limb_movements
import matplotlib.pyplot as plt
# Paths to the signal and annotation files
path_signal = '/Volumes/T7/MGH/eeg/sub-S0001111189075_ses-1_task-psg_eeg.edf'
path_annotations = '/Volumes/T7/MGH/eeg/sub-S0001111189075_ses-1_task-psg_annotations.csv'

# E.g.:
# path_signal = 'sub-S0001111189359_ses-1_task-psg_eeg.edf'
# path_annotations = 'sub-S0001111189359_ses-1_task-psg_annotations.csv'

# Load the signal data from the .edf file

signal, params = load_bdsp_signal(path_signal)

# Get the available channels from the signal data
ch_available = signal.columns

# Load the annotations from the .csv file
annotations = pd.read_csv(path_annotations)
# Get the original sampling frequency from the signal parameters
fs_original = params['Fs']
# Get the length of the signal data
signal_len = signal.shape[0]
plt.plot(signal['ecg-v1'][6000:12000].values)
plt.show()
# Preprocess the annotations to align with the signal data
annotations, annotations_quality = annotations_preprocess(annotations, fs_original, return_quality=True)

# In the vectorization step, specific values are assigned to different types of events in the annotations to create a numerical representation that can be used for further analysis. Here is a summary of the assigned values for each type of event:

# 1. **Respiratory Events:**
#    - Obstructive Apnea (OA): **1**
#    - Central Apnea (CA): **2**
#    - Mixed Apnea (MA): **3**
#    - Hypopnea (HY): **4**
#    - Respiratory Effort-Related Arousal (RERA): **5**

# 2. **Sleep Stages:**
#    - Wake (W): **5**
#    - REM (R): **4**
#    - N1: **3**
#    - N2: **2**
#    - N3: **1**

# 3. **Arousals:**
#    - Non-arousal: **0**
#    - Arousal: **1**

# 4. **Limb Movements:**
#    - Isolated Limb Movement: **1**
#    - Periodic Limb Movement: **2**
#    - Arousal-Associated Limb Movement: **3**
#    - Limb Movement (general, Natus): **4**


# Vectorize the respiratory events from the annotations
resp = vectorize_respiratory_events(annotations, signal_len)
# Vectorize the sleep stages from the annotations
stage = vectorize_sleep_stages(annotations, signal_len)
# Vectorize the arousals from the annotations
arousal = vectorize_arousals(annotations, signal_len)
# Vectorize the limb movements from the annotations
limb = vectorize_limb_movements(annotations, signal_len)


# print shapes:
print(f"Signal shape: {signal.shape}")
print(f"Respiratory events shape: {resp.shape}")
print(f"Sleep stages shape: {stage.shape}")
print(f"Arousals shape: {arousal.shape}")
print(f"Limb movements shape: {limb.shape}")

# E.g. for sub-S0001111189359_ses-1:
# Signal shape: (5529600, 16)
# Respiratory events shape: (5529600,)
# Sleep stages shape: (5529600,)
# Arousals shape: (5529600,)
# Limb movements shape: (5529600,)