import mne
import os

load_path = './1.0.0/ins9.edf'

raw_data = mne.io.read_raw_edf(load_path)
print(raw_data)