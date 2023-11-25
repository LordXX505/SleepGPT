import os

import glob
import numpy as np
root = '/gpfs/share/home/2201210064/Sleep/data/Physio'
for which in ['test', 'training']:
    root_path = os.path.join(root, which)
    files = glob.glob(pathname='*/*.arrow', root_dir=root_path)
    np.save(root_path, np.array(files))
    print(files)
    print(files.shape)