#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p q_fat_l
#SBATCH --qos=high
#SBATCH -J write_mass_ss2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
source activate pytorch
#srun python3 aug_mass_s2_list.py
#srun python3 check.py

srun python3 gen_apnea_lsit.py