#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=2 # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=2
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=16
#SBATCH --qos=high
#SBATCH --job-name=Test_shhs
#SBATCH --partition=GPU80G
#SBATCH --output=./temp_log/shhs/test_pretrain_2_shhs%j.out


# activate conda env
source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
ulimit -n 4096
#kfold_load_path=/home/cuizaixu_lab/huangweixuan/data/checkpoint
#load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Finetune_shhs1_cosine_backbone_base_patch200_l1_finetune_all_time_swin/version_3/ModelCheckpoint-epoch=21-val_acc=0.8700-val_score=6.6645.ckpt
#load_path=/home/cuizaixu_lab/huangweixuan/data/checkpoint/Finetune_shhs1_cosine_backbone_large_patch200_l1_finetune_all_time_swin/version_0/last.ckpt
#load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Finetune_shhs1_cosine_backbone_base_patch200_l1_finetune_all_time_swin/version_4/ModelCheckpoint-epoch=23-val_acc=0.8750-val_score=6.6777.ckpt
load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Finetune_shhs1_cosine_simple_conv_l1_finetune_all_time_swin/version_2/ModelCheckpoint-epoch=19-val_acc=0.8394-val_score=0.0000.ckpt
srun python3 main_test_kfold.py with   finetune_shhs1  SHHS1_WM_datasets    \
  num_gpus=2 num_nodes=1 num_workers=32 batch_size=64 model_arch=simple_conv  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.5 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=384 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=4 max_epoch=50 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
  load_path=$load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' actual_channels='shhs' \
  eval=True dist_on_itp=False device='cuda'