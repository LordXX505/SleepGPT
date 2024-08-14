#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0

nnodes="$1"
NODE_RANK="$2"
JOB_ID=1234567

OMP_NUM_THREADS=1 torchrun \
    --master_addr=172.17.172.8 \
    --nnodes $((nnodes)) \
    --nproc_per_node 8 \
    --rdzv_id 123456 \
    --master_port=6000 \
    --node_rank $((NODE_RANK)) \
   main.py with pretrain_time_fft_mtm all_A800_SHHS   \
  num_gpus=8 num_nodes=$((nnodes)) num_workers=124 batch_size=54 model_arch=backbone_huge_patch200 \
  lr=5e-4 end_lr=1e-6 random_choose_channels=11 max_steps=60000 lr_policy='cosine' loss_function='l1' \
  val_check_interval=1000 warmup_steps=10000

#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with pretrain_time_physio_SD_cuda   \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=128 model_arch=backbone_large_patch200 \
#  lr=2.5e-4 end_lr=1e-5 random_choose_channels=8 max_epoch=20 lr_policy=1 loss_function='l1' \

#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with pretrain_fft_physio_SD_cuda   \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=169 model_arch=backbone_large_patch200 \
#  lr=1e-3 end_lr=0 random_choose_channels=8 max_epoch=2 lr_policy=1 loss_function='l1' \
#  load_path='/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_1_backbone_large_patch200_l1_time_only/version_4/checkpoints/epoch=5-step=4880.ckpt'



#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with pretrain_physio_SD_cuda   \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=32 model_arch=backbone_large_patch200 \
#  lr=1e-3 min_lr=2.5e-7 warmup_lr=2.5e-7 random_choose_channels=8 max_epoch=2 lr_policy=cosine loss_function='l1' \
#  warmup_steps=0.2 load_path='/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_1_backbone_large_patch200_l1_fft_only/version_2/checkpoints/last.ckpt'
#OMP_NUM_THREADS=1 torchrun \
#  --master_addr=172.17.0.3 \
#  --nnodes $((nnodes)) \
#  --nproc_per_node 8 \
#  --rdzv_id 123456 \
#  --master_port=6000 \
#  --node_rank $((NODE_RANK)) \
# main.py with pretrain_time_fft_mtm   \
#num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=168 model_arch=backbone_large_patch200 \
#lr=2.5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=8 max_epoch=200 lr_policy=1 loss_function='l1' \
#warmup_steps=0.05 load_path='/home/hwx/Sleep/checkpoint/2201210064/experiments/sleep_cosine_backbone_large_patch200_l1_pretrain/version_6/checkpoints/last.ckpt'
#
#OMP_NUM_THREADS=1 torchrun \
#  --master_addr=172.17.0.3 \
#  --nnodes $((nnodes)) \
#  --nproc_per_node 8 \
#  --rdzv_id 123456 \
#  --master_port=6000 \
#  --node_rank $((NODE_RANK)) \
# main.py with pretrain_shhs_stage1   \
#num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=64 model_arch=backbone_large_patch200 \
#lr=8e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=4 max_epoch=5 lr_policy=1 loss_function='l1' \
#warmup_steps=0.1 val_check_interval=1.0
#
#OMP_NUM_THREADS=1 torchrun \
#  --master_addr=172.17.0.3 \
#  --nnodes $((nnodes)) \
#  --nproc_per_node 8 \
#  --rdzv_id 123456 \
#  --master_port=6000 \
#  --node_rank $((NODE_RANK)) \
# main.py with pretrain_shhs_stage2   \
#num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=384 model_arch=backbone_large_patch200 \
#lr=5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=4 max_epoch=800 lr_policy=cosine loss_function='l1' \
#warmup_steps=0.1 val_check_interval=1.0 \
#load_path='/data/checkpoint/Pshhs1_1_backbone_large_patch200_l1_pretrain/version_0/ModelCheckpoint-epoch=04-val_acc=0.0000-val_score=6.9949.ckpt'
#load_path='/data/checkpoint/Pshhs2_cosine_backbone_base_plus_patch200_l1_pretrain/version_0/last.ckpt'
load_path='/data/checkpoint/continue_cosine_backbone_base_plus_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=94-val_acc=0.0000-val_score=4.0838.ckpt'

#OMP_NUM_THREADS=1 torchrun \
#  --master_addr=172.17.0.3 \
#  --nnodes $((nnodes)) \
#  --nproc_per_node 8 \
#  --rdzv_id 123456 \
#  --master_port=6000 \
#  --node_rank $((NODE_RANK)) \
# main.py with  pretrain_shhs_stage1 \
#num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=64 model_arch=backbone_base_plus_patch200 \
#lr=3e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=4 max_epoch=5 lr_policy=cosine loss_function='l1' \
#warmup_steps=0.0 val_check_interval=1.0 Lambda=10.0 optim="Lion" patch_size=100 mask_ratio=0.6 \
#load_path=$load_path

#OMP_NUM_THREADS=1 torchrun \
#  --master_addr=172.17.0.3 \
#  --nnodes $((nnodes)) \
#  --nproc_per_node 8 \
#  --rdzv_id 123456 \
#  --master_port=6000 \
#  --node_rank $((NODE_RANK)) \
# main.py with pretrain_shhs_stage2   \
#num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=320 model_arch=backbone_base_plus_patch200 \
#lr=2.5e-4 min_lr=5e-8 warmup_lr=5e-8 random_choose_channels=4 max_epoch=150 lr_policy=cosine loss_function='l2' \
#warmup_steps=0.1 val_check_interval=1.0 Lambda=10.0 optim="Lion" patch_size=100 mask_ratio=0.5 \
#load_path=$load_path extra_name="continue"
