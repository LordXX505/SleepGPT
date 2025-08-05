#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1  #This needs to match Trainer(devices=...)
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=Visual_nmse
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/%j.out
#SBATCH --partition=q_ai4

source activate pytorch
load_path=/home/cuizaixu_lab/huangweixuan/DATA_C/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt

srun python3 main_kfold.py with visualization  visualization_mask_same  edf_datasets edf_2018 \
  num_gpus=1 num_nodes=1 num_workers=9 batch_size=256 model_arch=backbone_large_patch200  all_time=True \
  time_size=1 \
  split_len=1 \
  load_path=$load_path  \
  eval=True mode='visualization_mask_ratio_dynamic' visual=True device='cuda'
