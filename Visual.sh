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


JOB_ID=1234567

python main/Visualization/visual_sp.py  with pretrain_time_fft_mtm Young_datasets   \
  num_workers=124 batch_size=54 model_arch=backbone_huge_patch200 \
  lr=5e-4 end_lr=1e-6 random_choose_channels=11 max_steps=150000 lr_policy='cosine' loss_function='l2' \
  val_check_interval=1000 warmup_steps=10000   warmup_lr=0 val_check_interval=1.0 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=False time_size=1 pool=None \
  lr=5e-5 min_lr=0 random_choose_channels=8 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=1 \
  load_path=$load_path  \
  use_all_label='all' \
  optim="adamw" weight_decay=0.05 \
  layer_decay=0.75 get_param_method='no_layer_decay' Lambda=1.0 patch_size=200 use_cb=True kfold=5 \
  expert=None IOU_th=0.2 sp_prob=0.55 patch_time=30 dist_on_itp=False


