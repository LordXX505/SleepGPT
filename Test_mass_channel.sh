#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1  #This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9

#SBATCH --qos=high
#SBATCH --job-name=ft_sleep_maass_aug_ss2
#SBATCH --partition=q_ai4
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/channel/%j.out

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
kfold_load_path=/home/cuizaixu_lab/huangweixuan/data/checkpoint
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_1_all_timeswin/fold_0/version_0/ModelCheckpoint-epoch=01-val_acc=0.5160-val_macro=nan-val_score=5.5160.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_2_all_timeswin/fold_0/version_0/ModelCheckpoint-epoch=00-val_acc=0.5160-val_macro=nan-val_score=5.5160.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_5_all_timeswin/fold_0/version_0/ModelCheckpoint-epoch=24-val_acc=0.7960-val_macro=nan-val_score=5.7960.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_12_all_timeswin/fold_0/version_0/ModelCheckpoint-epoch=24-val_acc=0.8280-val_macro=0.6680-val_score=6.4960.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_12_all_timeswin/fold_0/version_1/ModelCheckpoint-epoch=16-val_acc=0.8510-val_macro=0.7883-val_score=6.6393.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_5_all_timeswin/fold_0/version_1/ModelCheckpoint-epoch=22-val_acc=0.7810-val_macro=nan-val_score=5.7810.ckpt
load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_ss3_all_timeswin/0_fold/version_7/ModelCheckpoint-epoch=46-val_acc=0.8440-val_macro=0.7708-val_score=6.6148.ckpt
srun python3 main_test_kfold_persub.py with finetune_mass_stage  MASS3_datasets  mode='Finetune_mass_ss3' \
  num_gpus=1 num_nodes=1 num_workers=16 batch_size=64 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.5 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=50 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
  kfold_load_path=$kfold_load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16   use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True kfold=1 grad_name='all' \
  resume_during_training=0 resume_ckpt_path="" save_top_k=1 \
  actual_channels="F3" eval=True dist_on_itp=False \
  kfold_test=7
