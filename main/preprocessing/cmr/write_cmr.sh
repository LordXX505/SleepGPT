#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0512G
#SBATCH --qos=high
#SBATCH -J write_cmr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source activate pytorch

srun python3 merge_physio.py