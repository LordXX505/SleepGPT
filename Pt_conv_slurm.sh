#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=2     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=2
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=16
#SBATCH --qos=high
#SBATCH --job-name=Sleep_Pretrain
#SBATCH --partition=GPU80G
#SBATCH --output=./temp_log/shhs/pretrain_1_%j.out

source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#load_path=/lustre/home/2201210064/Sleep/checkpoint/2201210064/experiments/Unify_cosine_backbone_base_patch200_l1_pretrain_all_time/version_12/ModelCheckpoint-epoch=04-val_acc=0.0000-val_score=4.6137.ckpt
load_path=''
srun python3 main.py \
with pretrain_shhs_stage2  SHHS1_WM_datasets \
num_gpus=2 num_nodes=2 num_workers=32 batch_size=640 model_arch=simple_conv \
lr=5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=4 max_epoch=50 lr_policy=cosine loss_function='l1' \
warmup_steps=0.1 val_check_interval=1.0 Lambda=1.0 optim="adamw" mask_ratio=0.75 \
load_path=$load_path extra_name="simple_conv" all_time=True split_len=1 time_size=1