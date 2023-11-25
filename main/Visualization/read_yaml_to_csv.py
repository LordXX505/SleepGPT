import os
import yaml
import pandas as pd

def read_yaml_to_excel(subject):
    read_path = '/home/hwx/Sleep/checkpoint/2201210064/experiments/'
    real_path = os.path.join(read_path, subject)
    path = os.path.join(real_path, 'hparams.yaml')
    # path = read_yaml_to_excel('sleep_1_backbone_large_patch200_l1/version_3/hparams.yaml')
    f = open(path)
    y = yaml.load(f, Loader=yaml.FullLoader)
    dct = y['config']
    args1 = []
    args2 = []
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = [dct[key]]
            args1.append(key)
        elif isinstance(value, type(dict)):
            l=[]
            for j in dct[key].keys():
                l.append(dct[key][j])
                args2.append(j)
            dct[key] = l

        else:
            args2.append(key)
    # print(args1, args2)
    dct['loss_names'] = [dct['loss_names']]
    dct['transform_keys'] =[dct['transform_keys']]
    df = pd.DataFrame.from_dict(dct)
    df.to_csv(real_path+'/hparams.csv')
    return path
p = read_yaml_to_excel('sleep_1_backbone_large_patch200_l1/version_3/')
