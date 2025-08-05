#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=4     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=4
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=Lightning
#SBATCH --partition=GPU36


# activate conda env
source activate pytorch

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 main.py with pretrain_physio_SD_cuda   \
  num_gpus=4 num_nodes=4 num_workers=36 batch_size=32 model_arch=backbone_large_patch16 \
  blr=5e-4 random_choose_channels=9 max_steps=50000