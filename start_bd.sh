#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=3
#SBATCH --job-name=Lightning
#SBATCH --qos=high
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodelist=gpu02
num_gpus=8
num_nodes=1
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
srun python3 main.py with pretrain_time_physio_SD_cuda  num_gpus=${num_gpus} num_nodes=${num_nodes} num_workers=24 \
  batch_size=108 model_arch=backbone_base_patch200 \
  blr=1e-5 random_choose_channels=9 max_epoch=800 lr_policy='cosine'