#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --time=5-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=9
#SBATCH --qos=high
#SBATCH --job-name=Spindledetection
#SBATCH --partition=q_fat_z
#SBATCH --output=/home/cuizaixu_lab/huangweixuan/DATA/temp_log/%j.out

source activate pytorch
load_path=/home/cuizaixu_lab/huangweixuan/DATA_C/data/checkpoint/Unify_cosine_backbone_large_patch200_l1_pretrain/version_3/ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt

srun python3 main/Visualization/Visual_cal_all.py with visualization  MASS1_datasets  \
  num_gpus=4 num_nodes=1 num_workers=36 batch_size=128 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=1.0 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=1 pool=None \
  lr=5e-5 min_lr=0 random_choose_channels=8 max_epoch=100 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=1 \
  load_path=$load_path mask_ratio=0.5 \
  use_all_label='all' \
  optim="adamw" weight_decay=0.05 \
  layer_decay=0.75 get_param_method='no_layer_decay' Lambda=1.0 patch_size=200 use_cb=True kfold=5 \
  expert=None IOU_th=0.2 sp_prob=0.55 patch_time=30 dist_on_itp=False
