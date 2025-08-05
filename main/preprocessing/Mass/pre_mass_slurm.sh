#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p q_fat_c
#SBATCH --qos=high
#SBATCH -J write_mass_ss2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source activate pytorch
srun python3 aug_mass_s2_list.py
#srun python3 Mass_ss2.py
#srun python3 Mass_ss2.py with visualization_sp Young_datasets device='cpu' \
#load_path=/home/cuizaixu_lab/huangweixuan/DATA_C/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt \
#model_arch=backbone_large_patch200 mask_ratio=0.25
#srun python3 generate_mass_list.py
#srun python3 generate_mass_ss_1_3_list.py
#srun python3 count_true_false_samples.py
#srun python3 transfer_split.py
#srun python3 compare_two_file.py
#srun python3 Pre_Mass_stage.py
#srun python3 check.py
#srun python3 check_mass.py