#!/bin/bash -l

#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)

#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=72
#SBATCH --partition=q_fat_c
source activate pytorch

srun python visual_umap.py
