#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export OMP_NUM_THREADS=1
export GIT_PYTHON_REFRESH=quiet
nnodes="$1"
NODE_RANK="$2"
JOB_ID=1234567

OMP_NUM_THREADS=1 torchrun \
    --master_addr=172.17.0.3 \
    --nnodes $((nnodes)) \
    --nproc_per_node 8 \
    --rdzv_id 123456 \
    --master_port=6000 \
    --node_rank $((NODE_RANK)) \
   main.py with pretrain_fft_physio_SD_cuda   \
  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=169 model_arch=backbone_base_patch200 \
  lr=1e-3 end_lr=0 random_choose_channels=11 max_epoch=5 lr_policy=1 loss_function='l1' \
  load_path=