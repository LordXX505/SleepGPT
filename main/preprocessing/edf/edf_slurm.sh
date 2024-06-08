#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p q_fat_l
#SBATCH --qos=high
#SBATCH -J write_mass_ss2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source activate pytorch
#srun python3 edf2018_gen_pre_list.py
#srun python3 edf2018_neuro2vec.py
srun python3 mul_eeg_generate.py
#srun python3 check.py
#srun python3 edf2013_gen_list_TCC.py