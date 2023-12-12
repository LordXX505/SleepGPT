import os
import mne
import matplotlib.pyplot as plt
import torch
import mne
import glob

path_root = '/Volumes/T7 Shield/SS2'
epoch = os.path.join(path_root, 'SS2_bio')
epochs = mne.io.read_raw_edf(glob.glob(epoch + f"/01-02-0019*PSG*")[0])
anno = mne.read_annotations(os.path.join(path_root, 'SS2_ana/01-02-0019 Spindles_E2.edf'))

epochs.set_annotations(anno)
epochs = epochs.crop(tmin=445*20, tmax=447*20)
epochs.plot(n_channels=1)
plt.show()
