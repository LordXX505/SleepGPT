#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=edf_aug
#SBATCH --partition=q_ai4
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/%j.out

source activate pytorch
#srun python3 edf2018_gen_pre_list.py
#srun python3 edf2018_neuro2vec.py
#srun python3 mul_eeg_generate.py
#srun python3 check.py
srun python3 edf2013_gen_list_TCC.py
#load_path=/home/cuizaixu_lab/huangweixuan/DATA_C/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt
#
#srun python3 augment_channels.py with visualization_sp Young_datasets  visualization_mask_same model_arch=backbone_large_patch200 load_path=$load_path \
#      num_gpus=1 num_nodes=1 num_workers=9
