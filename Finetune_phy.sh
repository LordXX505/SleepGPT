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
load_path=/data/checkpoint/0_fold/Finetune_phy_cosine_backbone_large_patch200_l1_finetune_all_time_swin/version_0/ModelCheckpoint-epoch=97-val_acc=0.8095-val_score=5.2902.ckpt
#os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with finetune_shhs1  \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=16 model_arch=backbone_large_patch200  lr_mult=20 \
#  warmup_lr=0 val_check_interval=None check_val_every_n_epoch=1 limit_train_batches=0.1 max_steps=1 all_time=True time_size=20 decoder_features=768 pool=None \
#  lr=0 min_lr=0 random_choose_channels=4 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.4 warmup_steps=1 split_len=20 \
#  load_path=$load_path \
#  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
#  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False extra_name="Swin_finetune" weight_decay=0.05 \
#  num_encoder_layers=6 layer_decay=0.85 Lambda=1.0 patch_size=200 use_cb=False

OMP_NUM_THREADS=1 torchrun \
    --master_addr=172.17.0.3 \
    --nnodes $((nnodes)) \
    --nproc_per_node 8 \
    --rdzv_id 123456 \
    --master_port=6000 \
    --node_rank $((NODE_RANK)) \
   main_kfold.py with finetune_phy   \
  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=32 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.5 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
  load_path=$load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  num_encoder_layers=6 layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True kfold=5