#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=8     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=8
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=4
#SBATCH --qos=high
#SBATCH --job-name=Sleep_Pretrain
#SBATCH --partition=q_ai8
source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
srun python3 main.py \
with pretrain_time_fft_mtm   \
num_gpus=8 num_nodes=1 num_workers=32 batch_size=48 model_arch=backbone_base_plus_patch200 \
lr=2.5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=8 max_epoch=200 lr_policy=cosine loss_function='l1' \
warmup_steps=0.1 val_check_interval=1.0 Lambda=1.0 optim="adamw" patch_size=100 mask_ratio=0.75 \
load_path=$load_path extra_name="Unify"