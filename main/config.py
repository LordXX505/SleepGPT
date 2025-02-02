from sacred import Experiment

ex = Experiment("Sleep", save_git_info=False)


def _loss_names(d):
    ret = {
        "Spindle": 0,
        "CrossEntropy": 0,
        "mtm": 0,
        "itc": 0,
        "itm": 0,
        'Apnea': 0,
    }

    ret.update(d)

    return ret


def _train_transform_keys(d):
    ret = {
        "keys": [[]],
        "mode": [[]],
    }
    ret.update(d)

    return ret


@ex.config
def config():
    extra_name = None
    exp_name = 'sleep'
    seed = 3407
    precision = "16-mixed"
    mode = 'pretrain'
    kfold=None
    # batch
    batch_size = 1
    max_epoch = -1
    max_steps = -1
    accum_iter = 2

    # data
    datasets = ['MASS']
    data_dir = ""

    # train configs
    dropout = 0.1
    loss_names = _loss_names({"Spindle": 0, "CrossEntropy": 0})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4]], "mode": ["random"]})
    start_epoch = 0
    num_workers = 30
    dropout = 0.
    drop_path_rate = 0.

    spindle = False
    lr = None
    blr = 1e-3
    min_lr = 1e-6
    lr_mult = 1
    end_lr = 1e-5
    warmup_lr = 2.5e-7
    warmup_steps = 0
    patch_size = -1
    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = None
    kfold_load_path = ""
    resume_ckpt_path = ""

    # sche and opt
    lr_policy = 1
    optim = "adam"
    clip_grad = False
    weight_decay = 1e-8

    # device
    device = 'cpu'
    deepspeed = False
    dist_on_itp = False
    num_gpus = -1
    num_nodes = -1

    # evaluation
    dist_eval = True
    eval = False

    # other
    fast_dev_run = 7
    val_check_interval = 1000

    # arch
    model_arch = 'backbone_base_patch200'

    # epochs:
    epoch_duration = 30
    fs = 100

    # mask_ratio
    mask_ratio = [0]

    # max_time_len
    max_time_len = 1

    # random choose channels
    random_choose_channels = 8

    get_recall_metric = False
    limit_val_batches = 1.0
    limit_train_batches = 1.0
    check_val_every_n_epoch = None

    time_only = False
    fft_only = False

    loss_function = 'l1'

    # data settings
    data_setting = {'SHHS': None, 'Young': "AMP", 'SD': 'AMP', 'Physio': None,
                    'MASS': None, 'ISRUC': None, 'EDF': None, 'MGH': None,}

    resume_during_training = None

    # finetune settings
    Swin_window_size = 60
    use_pooling = None
    mixup = 0
    use_relative_pos_emb = False
    all_time = False
    time_size = 11
    decoder_features = 128
    pool = 'attn'
    smoothing = 0.0
    decoder_heads = 12
    use_global_fft = False
    use_all_label = None
    split_len = None
    use_multiway = False
    get_param_method = 'no_layer_decay'
    use_g_mid = False
    local_pooling = False
    multi_y = ['tf']
    poly = False
    num_encoder_layers = 4
    layer_decay = 1.0
    Lambda = 1.0
    use_cb = False
    gradient_clip_val = None
    save_top_k = 2
    # kfold
    actual_channels = None
    kfold_test = None
    grad_name = 'all'

    #spindle
    mass_aug_times = 0
    expert = None
    IOU_th = 0.2
    sp_prob = 0.55
    patch_time = 30
    use_fpfn = None
    Use_FPN = None
    Event_decoder_depth = 4
    Event_enc_dim = 384
    num_queries = 400
    FPN_resnet = False
    CE_Weight = 10
    aug_test = None

    EDF_Mode = None
    subset = None
    visual = False
    visual_setting = {'mask_same': False, 'mode': None}
    persub = None
    return_alpha = False
    kfold_test = None
    show_transform_param = False
    mask_strategies=None
    #triton
    use_triton = False
    aug_dir = None
    aug_prob = 0.

@ex.named_config
def test():
    data_dir = './data'
    lr_policy = "cosine"
    optim = "adam"
@ex.named_config
def test_shhs():
    mode = "finetune"
    extra_name = "finetune_shhs"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None
    mixup = 0.8

    # data
    datasets = ['SHHS1']
    data_dir = ["/DATA/data/shhs_new"]

    # train configs
    dropout = 0

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    use_relative_pos_emb = True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    transform_keys = _train_transform_keys({"keys": [[]], "mode": ["shuffle"]})

@ex.named_config
def test_with_slurm():
    datasets = ['physio']
    data_dir = ["./main/DATA/Physio/test"]
    lr_policy = "cosine"
    optim = "adamw"
    dist_eval = True
    eval = False
    dist_on_itp = True
    device = 'cuda'
    num_gpus = 4
    num_nodes = 2
    max_epoch = 10


@ex.named_config
def SpindleUnet_MPS():
    # batch
    batch_size = 128
    max_epoch = 600
    accum_iter = 2

    # data
    datasets = ['SD', 'Young', 'physio', 'physio']
    data_dir = ["./main/DATA/SD", "./main/DATA/Young", "./main/DATA/Physio/training", "./main/DATA/Physio/test"]

    # train configs
    dropout = 0.
    loss_names = _loss_names({"mtm": 1, "itc": 1, "itm": 1})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr = 1e-4
    blr = 1e-3
    min_lr = 1e-6
    lr_mult = 1
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 1e-8

    # device
    device = 'cpu'
    deepspeed = False
    dist_on_itp = False
    num_gpus = -1
    num_nodes = -1

    # other
    fast_dev_run = 7

    dist_eval = True
    eval = False
@ex.named_config
def finetune_phy_usleep():
    mode = "finetune_downstream"
    extra_name = "Finetune_downstream_usleep"
    kfold = 1
@ex.named_config
def finetune_shhs1():
    mode = "finetune"
    extra_name = "Finetune_shhs1"
    mask_ratio = None

    loss_names = _loss_names({"CrossEntropy": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    use_relative_pos_emb=True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    actual_channels = 'shhs'

@ex.named_config
def finetune_phy():
    mode = "finetune"
    extra_name = "Finetune_phy"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None
    mixup = 0.8

    # data
    datasets = ['physio_train']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/Physio/training"]

    # train configs
    dropout = 0
    loss_names = _loss_names({"CrossEntropy": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    use_relative_pos_emb=True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    actual_channels='physio'

@ex.named_config
def finetune_edf():
    mode = "finetune"
    extra_name = "Finetune_edf"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None
    mixup = 0.8

    # data
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed"]

    # train configs
    dropout = 0
    loss_names = _loss_names({"CrossEntropy": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    use_relative_pos_emb=True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    actual_channels = 'EDF'
@ex.named_config
def finetune_mass_stage():
    mode = "Finetune_mass_all"
    extra_name = "Finetune_mass_all"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None
    mixup = 0.8

    # data
    datasets = ['MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS1",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS3",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS4",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS5",
                ]

    # train configs
    dropout = 0
    loss_names = _loss_names({"CrossEntropy": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    use_relative_pos_emb = True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    data_setting = {'MASS': 'AMP'}
@ex.named_config
def finetune_ISRUC_S1():
    mode = "finetune_ISRUC_S1"
    extra_name = "ISRUC_S1"
    loss_names = _loss_names({"CrossEntropy": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]})
    patch_size = 200
    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True
    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    multi_y = ['tf']
    data_setting = ['ISRUC']
@ex.named_config
def finetune_ISRUC_S3():
    mode = "finetune_ISRUC_S3"
    extra_name = "ISRUC_S3"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None

    # train configs
    dropout = 0
    loss_names = _loss_names({"CrossEntropy": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]})

    patch_size = 200
    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    multi_y = ['tf']
    data_setting = ['ISRUC']

@ex.named_config
def mass_mask_spindle_05():
    datasets = ['MASS']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_aug_new_1_0.5/SS2"]
@ex.named_config
def finetune_MASS_Spindle():
    mode = "Spindledetection"
    extra_name = "finetune_MASS_Spindle"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None

    # data
    datasets = ['MASS']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_aug_new_1/SS2"]
    data_setting = {'MASS': 'AMP'}
    # train configs
    dropout = 0
    loss_names = _loss_names({"Spindle": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3]], "mode": ["shuffle"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    gradient_clip_val = 1.0
    fast_dev_run = 7
@ex.named_config
def finetune_MASS_Apnea():
    mode = "Apneadetection_SS1"
    extra_name = "finetune_MASS_Apnea"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None

    # data
    datasets = ['MASS_Apnea']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_1_Apnea"]
    data_setting = {'MASS': None}
    # train configs
    dropout = 0
    loss_names = _loss_names({"Apnea": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3]], "mode": ["shuffle"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    gradient_clip_val = 1.0
    fast_dev_run = 7
@ex.named_config
def pretrain_datasets():
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1', 'SHHS2', 'Young']
    data_dir = ["/DATA/data/SD", "/DATA/data/Physio/training", "/DATA/data/Physio/test", "/DATA/data/shhs",
                "/DATA/data/shhs", "/DATA/data/Young"]
@ex.named_config
def pretrain_physio_SD_cuda():

    # batch
    batch_size = 48
    max_epoch = 400
    accum_iter = 2
    mask_ratio = [0.75]

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1', 'SHHS2', 'Young']
    data_dir = ["/DATA/data/SD", "/DATA/data/Physio/training", "/DATA/data/Physio/test", "/DATA/data/shhs",
                "/DATA/data/shhs", "/DATA/data/Young"]
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4]], "mode": ["full"]})
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1, "itc": 1, "itm": 1})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def edf_2018():
    mode = "finetune"
    EDF_Mode = '2018'
    kfold = 10
    kfold_test = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]

@ex.named_config
def edf_2013():
    mode = "finetune"
    EDF_Mode = '2013'
    kfold = 20
@ex.named_config
def edf_pr_955():
    mode = "Other_EDF_Pretrain"
    extra_name = "Finetune_edf_955"
    EDF_Mode='9_5_5'
@ex.named_config
def edf_ft_usleep():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_usleep"
    EDF_Mode = 'usleep'
@ex.named_config
def edf_ft_n2v():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_n2v"
    EDF_Mode = 'n2v'
@ex.named_config
def edf_ft_mul():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_mul"
    EDF_Mode = 'mul'
@ex.named_config
def edf_ft_955():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_955"
    EDF_Mode ='9_5_5'

@ex.named_config
def edf_ft_TCC():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_TCC"
    EDF_Mode='TCC'

@ex.named_config
def edf_ft_955_linear():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_955_linear"
    EDF_Mode = '9_5_5'
@ex.named_config
def edf_portion_1_datasets():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_portion_1"
    EDF_Mode = 'Portion_1_New'
@ex.named_config
def edf_portion_2_datasets():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_portion_2"
    EDF_Mode = 'Portion_2_New'
@ex.named_config
def edf_portion_5_datasets():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_portion_5"
    EDF_Mode = 'Portion_5_New'
@ex.named_config
def edf_portion_12_datasets():
    mode = "Other_EDF_Finetune"
    extra_name = "Finetune_edf_portion_12"
    EDF_Mode = 'Portion_12_New'

@ex.named_config
def pretrain_time_fft_mtm():

    # batch
    batch_size = 48
    max_epoch = 200
    accum_iter = 2
    mask_ratio = [0.75]
    mask_strategies= 'all'
    transform_keys = _train_transform_keys({"keys": [[1, 4, 6, 7]], "mode": ["full"]})
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1, "itc": 0, "itm": 0})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 100
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint_log/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""
    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True
    all_time = True
    split_len = 1
    time_size = 1
    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def pretrain_physio_SD_cpu():
    # batch
    batch_size = 48
    max_epoch = 400
    accum_iter = 2
    mask_ratio = [0.6]

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1', 'SHHS2', 'Young']
    data_dir = ["/data/data/SD", "/data/data/Physio/training", "/data/data/Physio/test", "/data/data/shhs",
                "/data/data/shhs", "/data/data/Young"]

    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1, "itc": 1, "itm": 1})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cpu'
    deepspeed = False
    dist_on_itp = True

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def pretrain_time_physio_SD_cuda():

    mode = 'pretrain'
    mask_ratio = [0.75]
    val_check_interval = 1000

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1', 'SHHS2', 'Young']
    data_dir = ["/data/data/SD", "/data/data/Physio/training", "/data/data/Physio/test", "/data/data/shhs", "/data/data/shhs", "/data/data/Young"]
    # datasets = ['physio_test', 'Young']
    # data_dir = ["/data/data/Physio/test", "/data/data/Young"]
    # train configs
    dropout = 0.1
    loss_names = _loss_names({"mtm": 1, "itc": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], "mode": ["shuffle", "full"]})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 1e-5

    warmup_steps = 50

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

    # arch
    model_arch = 'backbone_base_patch200'
    time_only = True

@ex.named_config
def pretrain_fft_physio_SD_cuda():
    mode = 'pretrain'

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1', 'SHHS2', 'Young']
    data_dir = ["/data/data/SD", "/data/data/Physio/training", "/data/data/Physio/test", "/data/data/shhs",
                "/data/data/shhs", "/data/data/Young"]
    # datasets = ['physio_test', 'Young']
    # data_dir = ["/data/data/Physio/test", "/data/data/Young"]
    # train configs
    dropout = 0.1
    loss_names = _loss_names({"itc": 1})
    transform_keys = _train_transform_keys({"keys": [[], []], "mode": ["random", "random"]})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 1e-5
    mask_ratio=None
    warmup_steps = 50

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7
    val_check_interval = 1000


    # arch
    model_arch = 'backbone_base_patch200'
    fft_only = True
@ex.named_config
def pretrain_shhs_stage1():
    # batch
    extra_name = 'Pshhs1'
    batch_size = 48
    max_epoch = 200
    accum_iter = 2
    mask_ratio = [0.75]

    # data
    datasets = ['SHHS1']
    data_dir = ["/data/data/shhs_new"]
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4]], "mode": ["full"]})
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1, "itc": 1, "itm": 1})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def pretrain_shhs_stage2():
    # batch
    extra_name = 'Pshhs2'
    batch_size = 48
    max_epoch = 200
    accum_iter = 2
    mask_ratio = [0.75]

    # data
    datasets = ['SHHS1']
    data_dir = ["/data/data/shhs_new"]
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4]], "mode": ["full"]})
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1, "itc": 0, "itm": 0})
    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = True


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def visualization():
    mask_ratio = [0.33]
    mode = 'visualization'
    kfold = 1
    model_arch = 'backbone_large_patch200'
    batch_size = 1
    # data
    datasets = ['MASS1']
    data_dir = ["/Volumes/T7/MASS_Processed/SS1"]
    # datasets = ['SHHS1']
    # data_dir = ["/Users/hwx_admin/Downloads/deep-learning-for-image-processing-pytorch_classification/Sleep/data/shhs"]
    # datasets = ['SD']
    # data_dir = ["../../main/data/SD"]
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1})
    device = 'cpu'
    transform_keys = _train_transform_keys({"keys": [[]], "mode": ["random"]})
    time_only = False
    # random choose channels
    random_choose_channels = 8
    patch_size = 200
    mask_strategies=None
    # load_path='/data/checkpoint/Pshhs2_cosine_backbone_large_patch200_l1_pretrain/version_1/ModelCheckpoint-epoch=787-val_acc=0.0000-val_score=4.1723.ckpt'
    # load_path = '/mnt/data/checkpoint/sleep_cosine_backbone_huge_patch200_l1_pretrain_all_time/version_60/last.ckpt'

    visual = False

@ex.named_config
def visualization_fft():
    mode = 'visualization'
    model_arch = 'backbone_base_patch200'
    batch_size = 1
    # data
    datasets = ['SHHS1_922_057', 'SHHS2', 'Young']
    data_dir = [ "/data/data/shhs", "/data/data/shhs", "/data/data/Young"]
    # datasets = ['SD']
    # data_dir = ["../../main/data/SD"]
    fft_only = True
    # train configs
    dropout = 0
    loss_names = _loss_names({"itc": 1})
    device = 'cpu'
    transform_keys = _train_transform_keys({"keys": [[], []], "mode": ["random", "random"]})

    data_setting = ['SHHS', 'visualization', 'ECG']
    # random choose channels
    random_choose_channels = 11
    patch_size = 200
    # load_path = '/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_1_backbone_base_patch200_l1/version_3/checkpoints/epoch=49-step=32936.ckpt'
    load_path = '/root/Sleep/cuizaixu/sleep_cosine_backbone_huge_patch200_l1_pretrain_all_time/version_57/ModelCheckpoint-epoch=00-val_acc=0.0000-val_score=4.5766.ckpt'
@ex.named_config
def visualization_block():
    mode = 'visualization'
    model_arch = 'backbone_base_patch200'
    batch_size = 2
    mask_ratio = [0.6]
    # data
    datasets = ['SHHS1', 'SHHS2', 'Young']
    data_dir = [ "/data/data/shhs", "/data/data/shhs", "/data/data/Young"]
    # datasets = ['SD']
    # data_dir = ["../../main/data/SD"]
    # train configs
    dropout = 0
    loss_names = _loss_names({"CrossEntropy": 1})
    device = 'cpu'
    transform_keys = _train_transform_keys({"keys": [[]], "mode": ["random"]})

    data_setting = ['SHHS', 'ECG']
    # random choose channels
    random_choose_channels = 11
    patch_size = 200
    load_path = '/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_1_backbone_base_patch200_l1/version_3/checkpoints/epoch=49-step=32936.ckpt'
@ex.named_config
def visualization_no_fft():
    mask_ratio = [0.75, 1.0]
    visual_setting = {'mask_same': False, 'mode': 'no_fft'}
@ex.named_config
def visualization_mask_same():
    mask_ratio = [0.25]
    visual_setting = {'mask_same': True, 'mode': 'mask_same'}
@ex.named_config
def visualization_using_all_fft():
    mask_ratio = [0.5, 0.0]
    visual_setting = {'mask_same': False, 'mode': 'all_fft'}

@ex.named_config
def visualization_sp():
    mode = "Stagedetection"
    extra_name = "finetune_MASS_Spindle"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = [0.5]
    patch_time = 30
    # data

    # data
    datasets = ['MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS1",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS3",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS4",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS5",]
    #             ]
    # datasets = ['EDF']
    # data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed"]
    # EDF_setting = 'EDF'
    # datasets = ['MASS']
    # data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS/SS2"]
    # data_setting = ['MASS']
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 3]], "mode": ["random"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint_log/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = False

    # evaluation
    dist_eval = False
    eval = False
    show_transform_param = True
    # other
    fast_dev_run = 7
@ex.named_config
def MASS_datasets():
    datasets = ['MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS1",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS3",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS4",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS5", ]
    data_setting = {'MASS': 'AMP'}
    kfold = 20

@ex.named_config
def Local_MASS1_datasets():
    datasets = ['MASS1']
    data_dir = ["/Volumes/T7/MASS_Processed/SS1"]
    data_setting = ['MASS']
@ex.named_config
def MASS_ckpt_version():
    kfold_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
@ex.named_config
def MASS1_datasets():
    datasets = ['MASS1']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS1"]
    data_setting = {'MASS': 'AMP'}
    kfold = 20
@ex.named_config
def MASS2_datasets():
    datasets = ['MASS2']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2"]
    data_setting = {'MASS': 'AMP'}
    kfold = 20

    actual_channels = None
@ex.named_config
def MASS2_aug_random_insert_datasets():
    datasets = ['MASS2']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2",
                ]
    aug_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/Aug_Random"]
    aug_prob = 0.25
    data_setting = {'MASS': 'AMP'}
    actual_channels = None

@ex.named_config
def MASS2_aug_random_datasets():
    datasets = ['MASS2', 'MASS2_AUG']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2","/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/Aug_Random"]
    data_setting = {'MASS': 'AMP'}
    actual_channels = None
@ex.named_config
def MASS3_datasets():
    datasets = ['MASS3']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS3"]
    data_setting = {'MASS': 'AMP'}

    kfold = 20

@ex.named_config
def MASS4_datasets():
    datasets = ['MASS4']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS4"]
    data_setting = {'MASS': 'AMP'}

    kfold = 20

@ex.named_config
def MASS5_datasets():
    datasets = ['MASS5']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS5"]
    data_setting = {'MASS': 'AMP'}

    kfold = 20

@ex.named_config
def SD_datasets():
    datasets = ['SD']
    data_dir = ["/DATA/data/SD"]

@ex.named_config
def edf_aug_f3_datasets():
    datasets = ['EDF']
    # data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_F3"]
    data_dir = ['/lustre/home/2201210064/DATA/data/sleep-cassette/Aug_F3']
    data_setting = 'EDF'
    actual_channels = 'EDF_F3'

@ex.named_config
def edf_aug_consecutive_datasets():
    datasets = ['EDF', 'EDF_AUG']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_consecutive"]
    # data_dir = ['/lustre/home/2201210064/DATA/data/sleep-cassette/Aug_F3_C4']
    data_setting = {'EDF': 'AMP'}
    actual_channels = 'EDF'
@ex.named_config
def edf_aug_all_datasets():
    datasets = ['EDF_AUG']
    data_dir = [
                "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_All"]
    # data_dir = ['/lustre/home/2201210064/DATA/data/sleep-cassette/Aug_F3_C4']
    data_setting = {'EDF': 'AMP'}
    actual_channels = 'EDF_F3_O1'
    #"F3", "Fpz", "O1", "Pz"
@ex.named_config
def edf_aug_half_datasets():
    datasets = ['EDF', 'EDF_AUG']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_Half"]
    data_setting = {'EDF': 'AMP'}
    actual_channels = 'EDF'
@ex.named_config
def edf_half_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed",
               ]
    data_setting = {'EDF': 'AMP'}
    actual_channels = 'EDF'

@ex.named_config
def edf_aug_file_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_file"]
    data_setting = 'EDF'
    actual_channels = 'EDF'
@ex.named_config
def edf_aug_f3_c4_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_F3_C4"]
    # data_dir = ['/lustre/home/2201210064/DATA/data/sleep-cassette/Aug_F3_C4']
    data_setting = 'EDF'
    actual_channels = 'EDF_F3_C4'

@ex.named_config
def edf_aug_c4_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_C4"]
    # data_dir = ['/lustre/home/2201210064/DATA/data/sleep-cassette/Aug_C4']
    data_setting = 'EDF'
    actual_channels = 'EDF_C4'
@ex.named_config
def edf_portion_mix_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed"]
    data_setting = {'EDF': 'AMP'}
    aug_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_All"]
    actual_channels = 'EDF'
@ex.named_config
def edf_portion_datasets():
    datasets = ['EDF', 'EDF_AUG']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/Aug_file"]
    data_setting = {'EDF': 'AMP'}
    actual_channels = 'EDF'

@ex.named_config
def edf_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed"]
    data_setting = {'EDF': 'AMP'}


@ex.named_config
def local_edf_datasets():
    datasets = ['EDF']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed"]
    data_setting = 'EDF'

@ex.named_config
def physio_train_datasets():
    datasets = ['physio_train']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/Physio/training"]
    kfold = 5
@ex.named_config
def physio_test_datasets():
    datasets = ['physio_test']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/Physio/test"]
    kfold = None
    all_time = False
@ex.named_config
def SHHS1_datasets():
    datasets = ['SHHS1']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA_C/data/data/shhs_new/shhs_new"]
    kfold = None
    data_setting ={'SHHS': 'AMP'}
@ex.named_config
def SHHS1_WM_datasets():
    datasets = ['SHHS1']
    data_dir = ["/lustre/home/2201210064/DATA/data/shhs_new/shhs"]
    kfold = None
    data_setting = ['SHHS']
@ex.named_config
def SHHS1_load_path():
    datasets = ['SHHS2']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/data/shhs_new/shhs_new"]
    data_setting = {'SHHS': 'AMP'}
@ex.named_config
def SHHS2_datasets():
    datasets = ['SHHS2']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/data/shhs_new/shhs_new"]
    data_setting = ['SHHS']

@ex.named_config
def Young_datasets():
    datasets = ['Young']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/Young"]
    data_setting = {'Young': None}

@ex.named_config
def Local_Young_datasets():
    datasets = ['Young']
    data_dir = ["/Volumes/T7/DATA/Young"]

@ex.named_config
def ISRUC_S3():
    datasets = ['ISRUC_S3']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/ISRUC_s3/processed"]
    data_setting = ['ISRUC']
@ex.named_config
def Local_ISRUC_S3():
    datasets = ['ISRUC_S3']
    data_dir = ["/Volumes/T7/DATA/ISRUC_s3/processed"]
    data_setting = ['ISRUC']
@ex.named_config
def ISRUC_S1():
    datasets = ['ISRUC_S1']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/DATA/data/ISRUC/processed"]
    data_setting = ['ISRUC']
@ex.named_config
def Local_ISRUC_S1():
    datasets = ['ISRUC_S1']
    data_dir = ["/Volumes/T7/DATA/ISRUC/processed"]
    data_setting = ['ISRUC']
@ex.named_config
def all_datasets():
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1', 'SHHS2', 'Young',
                'EDF', 'MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5','ISRUC_S1',
                'ISRUC_S3']
    data_dir = ["/DATA/data/SD", "/DATA/data/Physio/training", "/DATA/data/Physio/test",
                "/DATA/data/shhs",
                "/DATA/data/shhs", "/DATA/data/Young",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/sleep-cassette/processed",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS1",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS2",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS3",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS4",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/MASS_Processed/SS5",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/ISRUC_s1/processed",
                "/home/cuizaixu_lab/huangweixuan/DATA/data/ISRUC_s3/processed"
                ]

    data_setting = ['MASS', 'SHHS', 'EDF']
@ex.named_config
def all_A800_MGH():
    datasets = ['MGH']
    data_dir = [
                "/root/Sleep/DATA/MGH_NEW"
                ]

@ex.named_config
def all_A800_SHHS():
    datasets = ['SHHS1', 'SHHS2',]
    data_dir = [
                 "/root/Sleep/DATA/shhs",
                "/root/Sleep/DATA/shhs",
                ]

@ex.named_config
def all_A800_datasets():
    datasets = ['SD', 'physio_test', 'SHHS1', 'SHHS2', 'Young','MGH']
    data_dir = ["/root/Sleep/DATA/SD",
                "/root/Sleep/DATA/physio/test",
                "/root/Sleep/DATA/shhs",
                "/root/Sleep/DATA/shhs",
                "/root/Sleep/DATA/Young",
                "/root/Sleep/DATA/MGH_NEW"
                ]

@ex.named_config
def visualization_test():
    mode = "Stagedetection"
    extra_name = "finetune_MASS_Spindle"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = [0.5]
    patch_time = 30
    # data

    dropout = 0
    loss_names = _loss_names({"mtm": 1})
    transform_keys = _train_transform_keys({"keys": [[0, 3]], "mode": ["random"]})

    start_epoch = 0
    num_workers = 0
    drop_path_rate = 0.1
    patch_size = 200
    lr_mult = 1  # multiply lr for downstream heads
    blr = 1.5e-5
    end_lr = 0

    warmup_steps = 5

    # dir
    output_dir = './checkpoint_log/2201210064/experiments'
    log_dir = './checkpoint_log/2201210064/experiments'
    load_path = ""

    # sche and opt
    lr_policy = "cosine"
    optim = "adamw"
    clip_grad = False
    weight_decay = 0.05

    # device
    device = 'cuda'
    deepspeed = False
    dist_on_itp = False

@ex.named_config
def get_mu_std():
    batch_size=1024
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS', 'Young']
    data_dir = ["/DATA/data/SD", "/DATA/data/Physio/training", "/DATA/data/Physio/test", "/DATA/data/shhs",
                "/DATA/data/Young"]
    transform_keys = _train_transform_keys({"keys": [[], []], "mode": ["random", "random"]})

    data_setting = ['ECG', 'SHHS', 'visualization']

