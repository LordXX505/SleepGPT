#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1  #This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9

#SBATCH --qos=high
#SBATCH --job-name=ft_sleep_edf_aug_2018
#SBATCH --partition=q_ai4
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/edf_portion_umap/%j.out

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
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_12_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_12_New/fold_0/version_3/ModelCheckpoint-epoch=23-val_acc=0.8580-val_macro=0.7779-val_score=6.6359.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_5_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_5_New/fold_0/version_0/ModelCheckpoint-epoch=20-val_acc=0.8280-val_macro=0.7107-val_score=6.5387.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_2_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_2_New/fold_0/version_0/ModelCheckpoint-epoch=12-val_acc=0.5750-val_macro=nan-val_score=5.5750.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_1_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_1_New/fold_0/version_0/ModelCheckpoint-epoch=08-val_acc=0.4720-val_macro=nan-val_score=5.4720.ckpt

load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_12_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_12_New/fold_0/version_7/ModelCheckpoint-epoch=28-val_acc=0.8440-val_macro=0.7650-val_score=6.6090.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_5_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_5_New/fold_0/version_1/ModelCheckpoint-epoch=21-val_acc=0.8190-val_macro=0.6786-val_score=6.4976.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_2_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_2_New/fold_0/version_1/ModelCheckpoint-epoch=08-val_acc=0.4570-val_macro=nan-val_score=5.4570.ckpt
#load_path=/GPFS/cuizaixu_lab_permanent/huangweixuan/data/checkpoint/2201210064/experiments/Finetune_edf_portion_1_cosine_backbone_large_patch200_adamw_Other_EDF_Finetune_all_timeswin_Portion_1_New/fold_0/version_1/ModelCheckpoint-epoch=10-val_acc=0.4570-val_macro=nan-val_score=5.4570.ckpt

srun python3 main_test_kfold_persub.py with finetune_edf  edf_portion_12_datasets  edf_datasets  visualization_umap \
  num_gpus=1 num_nodes=1 num_workers=9 batch_size=12 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.5 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=30 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
  load_path=$load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' \
  resume_during_training=0 resume_ckpt_path="" kfold=2 use_triton=False actual_channels='EDF' \
  eval=True dist_on_itp=False

