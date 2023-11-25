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
load_path=/data/checkpoint/Pshhs2_cosine_backbone_large_patch200_l1_pretrain/version_1/ModelCheckpoint-epoch=787-val_acc=0.0000-val_score=4.1723.ckpt
#load_path='/data/checkpoint/continue_cosine_backbone_base_plus_patch200_l2_pretrain/version_0/ModelCheckpoint-epoch=148-val_acc=0.0000-val_score=4.2884.ckpt'
#os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
#load_path=""
OMP_NUM_THREADS=1 torchrun \
    --master_addr=172.17.0.3 \
    --nnodes $((nnodes)) \
    --nproc_per_node 8 \
    --rdzv_id 123456 \
    --master_port=6000 \
    --node_rank $((NODE_RANK)) \
   main.py with finetune_shhs1   \
  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=16 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=1e-7 val_check_interval=100 check_val_every_n_epoch=None limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=256 pool=None \
  lr=5e-4 min_lr=1e-7 random_choose_channels=4 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.4 warmup_steps=0.2 split_len=20 \
  load_path=$load_path \
  use_pooling='attn' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=8  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="Lion" poly=True extra_name="finetune_final" weight_decay=0.05 \
  num_encoder_layers=2 layer_decay=0.75 Lambda=10.0 patch_size=200
#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with finetune_shhs1   \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=16 model_arch=backbone_large_patch200  lr_mult=20 \
#  warmup_lr=1e-7 val_check_interval=100 check_val_every_n_epoch=None limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=256 pool=None \
#  lr=5e-4 min_lr=1e-7 random_choose_channels=4 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.4 warmup_steps=0.2 split_len=20 \
#  load_path=$load_path \
#  use_pooling='attn' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=8  use_global_fft=True use_all_label='all' \
#  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="Lion" poly=True extra_name="finetune_final" weight_decay=0.05 \
#  num_encoder_layers=2 layer_decay=0.75 Lambda=10.0 patch_size=200
#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with finetune_shhs1   \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=16 model_arch=backbone_large_patch200  lr_mult=20 \
#  warmup_lr=1e-7 val_check_interval=100 check_val_every_n_epoch=None limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=1024 pool=None \
#  lr=5e-4 min_lr=1e-7 random_choose_channels=4 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.4 warmup_steps=0.2 split_len=20 \
#  load_path=$load_path \
#  use_pooling='attn' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=32  use_global_fft=True use_all_label='all' \
#  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="Lion" poly=True extra_name="finetune_con" weight_decay=0.05 \
#  num_encoder_layers=2 layer_decay=0.65 Lambda=10.0 patch_size=100
#OMP_NUM_THREADS=1 torchrun \
#    --master_addr=172.17.0.3 \
#    --nnodes $((nnodes)) \
#    --nproc_per_node 8 \
#    --rdzv_id 123456 \
#    --master_port=6000 \
#    --node_rank $((NODE_RANK)) \
#   main.py with finetune_shhs1   \
#  num_gpus=8 num_nodes=$((nnodes)) num_workers=96 batch_size=16 model_arch=backbone_base_plus_patch200  lr_mult=20 \
#  warmup_lr=1e-7 val_check_interval=None check_val_every_n_epoch=1 limit_train_batches=0.1 max_steps=-1 all_time=True time_size=20 decoder_features=1024 pool=None \
#  lr=5e-4 min_lr=1e-7 random_choose_channels=4 max_epoch=20 lr_policy=cosine loss_function='l1' drop_path_rate=0.4 warmup_steps=35 split_len=20 \
#  load_path=$load_path \
#  use_pooling='attn' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=32  use_global_fft=True use_all_label='all' \
#  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="Lion" poly=True extra_name="Test_2" weight_decay=0.05 \
#  num_encoder_layers=2 layer_decay=0.5 Lambda=10.0 patch_size=100