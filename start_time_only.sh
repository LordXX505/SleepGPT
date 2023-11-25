#!/bin/bash
#SBATCH --partition=GPU36
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=9
#SBATCH --job-name=Lightning_cos_large
#SBATCH --qos=high
#SBATCH --signal=SIGUSR1@90



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

# run script from abovest
srun python3 main.py with pretrain_time_physio_SD_cuda   \
  num_gpus=4 num_nodes=2 num_workers=36 batch_size=169 model_arch=backbone_large_patch200 \
  blr=8e-5 random_choose_channels=9 max_steps=200000 lr_policy='cosine' loss_function='l1'