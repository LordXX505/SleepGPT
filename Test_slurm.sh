#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test_job
#SBATCH --partition=q_ai8
#SBATCH --output=test_%j.out

echo "✅ Job started at $(date)"
hostname
nvidia-smi
echo "✅ Job finished at $(date)"