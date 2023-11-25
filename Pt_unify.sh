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
load_path="/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_1/ModelCheckpoint-epoch=19-val_acc=0.0000-val_score=4.1498.ckpt"
OMP_NUM_THREADS=1 torchrun \
  --master_addr=172.17.0.3 \
  --nnodes $((nnodes)) \
  --nproc_per_node 8 \
  --rdzv_id 123456 \
  --master_port=6000 \
  --node_rank $((NODE_RANK)) \
 main.py with pretrain_time_fft_mtm   \
num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=160 model_arch=backbone_large_patch200 \
lr=2.5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=8 max_epoch=80 lr_policy=cosine loss_function='l1' \
warmup_steps=0.1 val_check_interval=1.0 Lambda=1.0 optim="adamw" patch_size=200 mask_ratio=0.75 \
load_path=$load_path extra_name="Unify"