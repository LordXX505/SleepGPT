#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=4
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=16
#SBATCH --job-name=ft_sleep_shhs
#SBATCH --partition=GPU80G
#SBATCH --output=./temp_log/shhs/pretrain_1_base_finetune%j.out
source activate pytorch
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# ulimit -n 16384

#load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Unify_cosine_backbone_base_patch200_l1_pretrain_all_time/version_13/ModelCheckpoint-epoch=40-val_acc=0.0000-val_score=4.1459.ckpt
#load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Unify_cosine_backbone_base_patch200_l1_pretrain_all_time/version_18/ModelCheckpoint-epoch=49-val_acc=0.0000-val_score=4.2047.ckpt
load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Unify_cosine_backbone_base_patch200_l1_pretrain_all_time/version_18/ModelCheckpoint-epoch=49-val_acc=0.0000-val_score=4.2047.ckpt
srun python3 main.py with finetune_shhs1  SHHS1_WM_datasets \
  num_gpus=4 num_nodes=1 num_workers=64 batch_size=32 model_arch=simple_conv  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.25 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=384 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=4 max_epoch=25 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
  load_path=$load_path  \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smooth ing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' actual_channels='shhs'