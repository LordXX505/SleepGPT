import os

import glob
import numpy as np

def merge_physio():
    root = '/gpfs/share/home/2201210064/Sleep/main/data/Physio'
    for which in ['test', 'training']:
        root_path = os.path.join(root, which)
        print(root_path + '/*/*.arrow')
        files = glob.glob(pathname=root_path + '/*/*.arrow')
        np.save(root_path, np.array(files))
        print(len(files))

def merge_SD():
    root_path = '/gpfs/share/home/2201210064/Sleep/main/data/SD'
    print(root_path + '/*/*.arrow')
    files = glob.glob(pathname=root_path + '/*/*.arrow')
    np.save(root_path, np.array(files))
    print(len(files))

def merge_young():
    root_path = '/gpfs/share/home/2201210064/Sleep/main/data/Young'
    print(root_path + '/*/*.arrow')
    val = np.load(os.path.join(root_path, 'val.npy'), allow_pickle=True)
    test = np.load(os.path.join(root_path, 'test.npy'), allow_pickle=True)

    files = glob.glob(pathname=root_path + '/*/*.arrow')
    np.save(root_path, np.array(files))
    print(len(files))

if __name__ == '__main__':
    merge_SD()
    merge_young()