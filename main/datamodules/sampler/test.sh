#!/bin/bash
#SBATCH --job-name=balanced_sampler_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --qos=high
#SBATCH --partition=q_ai8


source activate pytorch

srun python test_sampler.py