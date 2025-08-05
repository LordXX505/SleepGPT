#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=2     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=2
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=4
#SBATCH --qos=high
#SBATCH --job-name=Sleep_edf_955
#SBATCH --partition=q_ai4
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/data/temp_log/%j.out

source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
load_path=/home/cuizaixu_lab/huangweixuan/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt

srun python3 main_kfold.py \
with pretrain_time_fft_mtm  edf_pr_955 \
num_gpus=2 num_nodes=1 num_workers=8 batch_size=48 model_arch=backbone_large_patch200 \
lr=2.5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=8 max_epoch=25 lr_policy=cosine loss_function='l1' \
warmup_steps=0.1 val_check_interval=1.0 Lambda=1.0 optim="adamw" patch_size=200 mask_ratio=0.75 \
load_path=$load_path extra_name="Unify" kfold=5 grad_name='all'