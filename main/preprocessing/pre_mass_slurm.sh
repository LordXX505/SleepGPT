#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p q_fat_l
#SBATCH --qos=high
#SBATCH -J write_mass_ss2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source activate pytorch
srun python3 Mass_ss2.py
#srun python3 Mass_ss2.py with visualization_sp device='cpu' \
#load_path='/home/cuizaixu_lab/huangweixuan/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt' \
#model_arch=backbone_large_patch200 mask_ratio=0.75
#
#srun python3 count_true_false_samples.py