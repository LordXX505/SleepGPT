#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p q_cn
#SBATCH --qos=high
#SBATCH -J write_mass_ss2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
source activate pytorch

#srun python3 aug_compare.py
srun python3 edf_2013_gen_list_1p5p10p.py
