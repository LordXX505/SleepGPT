#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=2     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=2
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=Spindledetection
#SBATCH --partition=q_ai8


# activate conda env
source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1



# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
ulimit -n 10240
# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0
load_path=/home/cuizaixu_lab/huangweixuan/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt
srun python3 main_kfold.py with finetune_MASS_Spindle   \
  num_gpus=2 num_nodes=1 num_workers=18 batch_size=192 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=1.0 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=1 pool=None \
  lr=1e-4 min_lr=5e-8 random_choose_channels=8 max_epoch=5 lr_policy=cosine loss_function='l1' drop_path_rate=0.2 warmup_steps=0.1 split_len=1 \
  load_path=$load_path \
  use_all_label='all' \
  optim="Lion" weight_decay=1e-8 \
  layer_decay=0.65 get_param_method='no_layer_decay' Lambda=10.0 patch_size=200 use_cb=True kfold=5 \
  expert='E2' IOU_th=0.2 sp_prob=0.55 patch_time=20 use_fpfn='BCE' Use_FPN='MLP' Spindle_decoder_depth=3 Spindle_enc_dim=1280 \
  grad_name='all' num_queries=250 FPN_resnet=False actual_channels='MASS_SP' CE_Weight=20
