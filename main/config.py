from sacred import Experiment

ex = Experiment("Sleep", save_git_info=False)


def _loss_names(d):
    ret = {
        "FpFn": 0,
        "CrossEntropy": 0,
        "mtm": 0,
        "itc": 0,
        "itm": 0,

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
    seed = 8678
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
    loss_names = _loss_names({"FpFn": 0, "CrossEntropy": 0})
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
    log_dir = './checkpoint/2201210064/experiments'
    load_path = ""

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
    mask_ratio = 0

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

    # physio settings
    physio_settings = ['ECG']
    shhs_settings =['SHHS']

    resume_during_training = None

    # finetune settings
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
    num_encoder_layers = 2
    layer_decay = 1.0
    Lambda = 1.0
    use_cb = False

    # kfold
    actual_channels=None
    kfold_test=None

    #spindle
    expert=None
    IOU_th=0.2
    sp_prob=0.55
    patch_time=30
    mass_settings=None

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
    data_dir = ["/data/data/shhs_new"]

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
    use_relative_pos_emb = True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    transform_keys = _train_transform_keys({"keys": [[]], "mode": ["shuffle"]})

@ex.named_config
def test_with_slurm():
    datasets = ['physio']
    data_dir = ["./main/data/Physio/test"]
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
    data_dir = ["./main/data/SD", "./main/data/Young", "./main/data/Physio/training", "./main/data/Physio/test"]

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
    log_dir = './checkpoint/2201210064/experiments'
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
    data_dir = ["/data/data/Physio/training"]

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
    use_relative_pos_emb=True
    local_pooling = True
    # multi_y = ['tf', 'time', 'fft']
    multi_y = ['tf']
    actual_channels='physio'



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
    data_dir = ["/home/cuizaixu_lab/huangweixuan/data/data/MASS/SS2"]
    mass_settings = ['MASS']
    # train configs
    dropout = 0
    loss_names = _loss_names({"FpFn": 1})
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


    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def pretrain_physio_SD_cuda():

    # batch
    batch_size = 48
    max_epoch = 400
    accum_iter = 2
    mask_ratio = 0.75

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1_922_057', 'SHHS2', 'Young']
    data_dir = ["/data/data/SD", "/data/data/Physio/training", "/data/data/Physio/test", "/data/data/shhs",
                "/data/data/shhs", "/data/data/Young"]
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
def pretrain_time_fft_mtm():

    # batch
    batch_size = 48
    max_epoch = 200
    accum_iter = 2
    mask_ratio = 0.75

    # data
    datasets = ['SD', 'physio_test', 'SHHS1', 'SHHS2', 'Young']
    data_dir = ["/data/data/SD", "/data/data/Physio/test", "/data/data/shhs",
                "/data/data/shhs", "/data/data/Young"]
    transform_keys = _train_transform_keys({"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["random"]})
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
def pretrain_physio_SD_cpu():
    # batch
    batch_size = 48
    max_epoch = 400
    accum_iter = 2
    mask_ratio = 0.6

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1_922_057', 'SHHS2', 'Young']
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
    log_dir = './checkpoint/2201210064/experiments'
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
    mask_ratio = 0.75
    val_check_interval = 1000

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1_922_057', 'SHHS2', 'Young']
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

    # arch
    model_arch = 'backbone_base_patch200'
    time_only = True

@ex.named_config
def pretrain_fft_physio_SD_cuda():
    mode = 'pretrain'

    # data
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS1_922_057', 'SHHS2', 'Young']
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
    mask_ratio = 0.75

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
def pretrain_shhs_stage2():
    # batch
    extra_name = 'Pshhs2'
    batch_size = 48
    max_epoch = 200
    accum_iter = 2
    mask_ratio = 0.75

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
def visualization():
    mask_ratio = 0.75
    mode = 'pretrain'
    model_arch = 'backbone_large_patch200'
    batch_size = 1
    # data
    datasets = ['SD']
    data_dir = ["/Users/hwx_admin/Downloads/deep-learning-for-image-processing-pytorch_classification/Sleep/data/SD"]
    # datasets = ['SHHS1']
    # data_dir = ["/Users/hwx_admin/Downloads/deep-learning-for-image-processing-pytorch_classification/Sleep/data/shhs"]
    # datasets = ['SD']
    # data_dir = ["../../main/data/SD"]
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1})
    device = 'cpu'
    transform_keys = _train_transform_keys({"keys": [[]], "mode": ["random"]})
    physio_settings = ['visualization', 'ECG']
    shhs_settings = ['SHHS']
    time_only = False
    # random choose channels
    random_choose_channels = 8
    patch_size = 200
    # load_path='/data/checkpoint/Pshhs2_cosine_backbone_large_patch200_l1_pretrain/version_1/ModelCheckpoint-epoch=787-val_acc=0.0000-val_score=4.1723.ckpt'
    load_path = '/Users/hwx_admin/Downloads/deep-learning-for-image-processing-pytorch_classification/Sleep/data/checkpoint/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt'


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
    physio_settings = ['visualization', 'ECG']
    shhs_settings = ['SHHS']

    # random choose channels
    random_choose_channels = 11
    patch_size = 200
    load_path = '/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_1_backbone_base_patch200_l1/version_3/checkpoints/epoch=49-step=32936.ckpt'

@ex.named_config
def visualization_block():
    mode = 'visualization'
    model_arch = 'backbone_base_patch200'
    batch_size = 2
    mask_ratio = 0.6
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
    physio_settings = ['ECG']
    shhs_settings = ['SHHS']

    # random choose channels
    random_choose_channels = 11
    patch_size = 200
    load_path = '/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_1_backbone_base_patch200_l1/version_3/checkpoints/epoch=49-step=32936.ckpt'

@ex.named_config
def visualization_sp():
    mode = "Spindledetection"
    extra_name = "finetune_MASS_Spindle"
    # batch
    batch_size = 169
    max_epoch = 50
    accum_iter = 2
    mask_ratio = None

    # data
    datasets = ['MASS']
    data_dir = ["/home/cuizaixu_lab/huangweixuan/data/data/MASS/SS2"]
    mass_settings = ['MASS']
    # train configs
    dropout = 0
    loss_names = _loss_names({"mtm": 1})
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

    # evaluation
    dist_eval = False
    eval = False

    # other
    fast_dev_run = 7

@ex.named_config
def get_mu_std():
    batch_size=1024
    datasets = ['SD', 'physio_train', 'physio_test', 'SHHS', 'Young']
    data_dir = ["/data/data/SD", "/data/data/Physio/training", "/data/data/Physio/test", "/data/data/shhs",
                "/data/data/Young"]
    transform_keys = _train_transform_keys({"keys": [[], []], "mode": ["random", "random"]})
    physio_settings = ['visualization', 'ECG']
    shhs_settings = ['SHHS']
