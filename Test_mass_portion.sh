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
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/ss2_p/%j.out

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
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_12_all_timeswin/fold_2/version_2/ModelCheckpoint-epoch=23-val_acc=0.8360-val_macro=0.6957-val_score=6.5317.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_12_all_timeswin/fold_2/version_3/ModelCheckpoint-epoch=24-val_acc=0.8410-val_macro=0.7400-val_score=6.5810.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_5_all_timeswin/fold_0/version_8/ModelCheckpoint-epoch=20-val_acc=0.7170-val_macro=nan-val_score=5.7170.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_5_all_timeswin/fold_0/version_9/ModelCheckpoint-epoch=23-val_acc=0.8250-val_macro=nan-val_score=5.8250.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_2_all_timeswin/fold_0/version_4/ModelCheckpoint-epoch=00-val_acc=0.5780-val_macro=nan-val_score=5.5780.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_2_all_timeswin/fold_0/version_5/ModelCheckpoint-epoch=04-val_acc=0.5090-val_macro=nan-val_score=5.5090.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_1_all_timeswin/fold_0/version_4/ModelCheckpoint-epoch=00-val_acc=0.4760-val_macro=nan-val_score=5.4760.ckpt
load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_mass_all_cosine_backbone_large_patch200_adamw_Finetune_mass_portion_1_all_timeswin/fold_0/version_4/ModelCheckpoint-epoch=01-val_acc=0.4760-val_macro=nan-val_score=5.4760.ckpt
srun python3 main_test_kfold_persub.py with finetune_mass_stage  MASS2_datasets visualization_umap_MASS_1 \
  num_gpus=1 num_nodes=1 num_workers=9 batch_size=6 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.5 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=25 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
  load_path=$load_path  persub=True \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' \
  resume_during_training=0 resume_ckpt_path="" kfold=1  mode='Finetune_mass_portion_1' eval=True dist_on_itp=False \
  kfold_test=2
