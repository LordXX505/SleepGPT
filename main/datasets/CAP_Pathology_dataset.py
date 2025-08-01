
from .CAP_dataset import CAPDataset

class CAPDataset_n(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='n', *args, **kwargs)


class CAPDataset_ins(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='ins', *args, **kwargs)


class CAPDataset_narco(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='narco', *args, **kwargs)

class CAPDataset_nfle(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='nfle', *args, **kwargs)

class CAPDataset_plm(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='plm', *args, **kwargs)

class CAPDataset_rbd(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='rbd', *args, **kwargs)

class CAPDataset_sdb(CAPDataset):

    split = 'train'
    transform_keys = ['full']
    data_dir = ['./']
    column_names = ['signal', 'stage', 'good_channels', 'pathology']
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    def __init__(self, split="",  *args, **kwargs):
        super().__init__(split=split, pathology_name='sdb', *args, **kwargs)