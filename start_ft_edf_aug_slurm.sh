#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=2   #This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=2
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=16

#SBATCH --qos=normal
#SBATCH --job-name=ft_sleep_edf_2018
#SBATCH --partition=GPU40G
#SBATCH --output=/lustre/home/2201210064/Sleep/temp_log/%j.out

# activate conda env

source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# ulimit -n 16384
# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

load_path=/lustre/home/2201210064/DATA/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt

srun python3 main_kfold.py with finetune_edf  edf_2018 edf_aug_f3_datasets  \
  num_gpus=2 num_nodes=1 num_workers=9 batch_size=24 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.25 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=30 lr_policy=cosine loss_function='l1' drop_path_rate=0.2 warmup_steps=0.1 split_len=20 \
  load_path=$load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' \
  resume_during_training=0 resume_ckpt_path="" EDF_Mode='2018_F3'