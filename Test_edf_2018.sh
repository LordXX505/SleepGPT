#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1     # This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=Test_edf_2018
#SBATCH --partition=q_ai4
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/edf_2018_umap/%j.out


# activate conda env
source activate pytorch
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
ulimit -n 4096
kfold_load_path=/home/cuizaixu_lab/huangweixuan/data/checkpoint
load_path=/home/cuizaixu_lab/huangweixuan/DATA_C/data/checkpoint/Finetune_edf_cosine_backbone_large_patch200_adamw_finetune_all_timeswin/0_fold/version_1/ModelCheckpoint-epoch=27-val_acc=0.8650-val_score=5.4750.ckpt
srun python3 main_test_kfold_persub.py with finetune_edf  edf_datasets edf_2018 visualization_umap_edf_2018   \
  num_gpus=1 num_nodes=1 num_workers=9 batch_size=64 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=0.5 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=25 lr_policy=cosine loss_function='l1' drop_path_rate=0.15 warmup_steps=0.1 split_len=20 \
  load_path=$load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True kfold=1 grad_name='all' \
  kfold_test=1 eval=True dist_on_itp=False