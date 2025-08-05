#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p q_fat
#SBATCH --qos=high
#SBATCH -J write_mass_ss2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/%j.out

source activate pytorch
#srun python3 main/preprocessing/Mass/fold_test_spindles.py

srun python3 main/Visualization/cal_all_file_time.py

