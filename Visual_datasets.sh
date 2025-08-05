#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=Test_Visual_mass
#SBATCH --partition=q_ai4
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/visual/%j.out

# activate conda env
source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
ulimit -n 65536
load_path=/home/cuizaixu_lab/huangweixuan/DATA_C/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt

srun python3 main_test_last.py with visualization visualization_using_all_fft Young_datasets   \
  num_gpus=1 num_nodes=1 num_workers=9 batch_size=512 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=1.0 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=false time_size=1 pool=None \
  lr=5e-5 min_lr=0 random_choose_channels=8 max_epoch=100 lr_policy=cosine loss_function="l1" drop_path_rate=0.5 warmup_steps=0.1 split_len=1 \
  load_path=$load_path device="cuda" \
  use_all_label="all" \
  optim="adamw" weight_decay=0.05 \
  layer_decay=0.75 get_param_method="no_layer_decay" Lambda=1.0 patch_size=200 use_cb=True  persub="rec"



