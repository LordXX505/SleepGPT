#!/bin/bash
# 单节点多卡分布式训练脚本
# 使用方法:
# ./run_distributed.sh [gpu_ids] [cpu_per_gpu] [num_nodes]

# 参数默认值设置
GPU_IDS=${1:-"0,1,2,3,4,5"}                 # 默认使用前4张GPU
CPU_PER_GPU=${2:-16}                     # 每个GPU分配的CPU核心数
NUM_NODES=${3:-1}                       # 节点数（单节点固定为1）

# 计算实际资源量
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
TOTAL_CPUS=$(( CPU_PER_GPU * NUM_GPUS ))

# 参数校验函数
validate_arguments() {
    # 验证GPU ID格式
    if ! [[ $GPU_IDS =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "错误: GPU ID格式不正确，请使用逗号分隔的数字 (如 0,1,2)"
        exit 1
    fi

    # 验证单节点设置
    if [ "$NUM_NODES" -ne 1 ]; then
        echo "警告: 这是单节点脚本，节点数强制设置为1"
        NUM_NODES=1
    fi

    # 验证CPU核心数
    local total_cores=$(nproc)
    if [ "$TOTAL_CPUS" -gt "$total_cores" ]; then
        echo "警告: 请求的CPU核心数($TOTAL_CPUS)超过系统总量($total_cores)"
        CPU_PER_GPU=$(( total_cores / NUM_GPUS ))
        echo "已自动调整为每个GPU使用 $CPU_PER_GPU 个CPU核心"
    fi
}

# 打印资源配置信息
print_resource_info() {
    echo "===== 分布式训练资源配置 ====="
    echo "使用GPU     : $GPU_IDS"
    echo "GPU数量     : $NUM_GPUS"
    echo "节点数      : $NUM_NODES"
    echo "CPU/GPU     : $CPU_PER_GPU"
    echo "总CPU核心   : $TOTAL_CPUS"
    echo "============================"
}

# 系统资源监控函数
monitor_resources() {
    local interval=10  # 监控间隔(秒)
    echo ""
    echo "开始资源监控 (按Ctrl+C停止)..."
    echo "时间戳       GPU利用率(%)   GPU显存使用(MB)  CPU利用率(%)"

    while true; do
        local timestamp=$(date +"%H:%M:%S")
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
        local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
        local cpu_util=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

        printf "%-10s | %-15s | %-15s | %s\n" "$timestamp" "$gpu_util" "$gpu_mem" "$cpu_util"
        sleep $interval
    done
}

# 主执行流程
validate_arguments
print_resource_info

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export OMP_NUM_THREADS=$CPU_PER_GPU      # 控制每个进程的OpenMP线程数
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=$(ip route get 8.8.8.8 | grep -oP 'dev \K\S+')  # 自动获取主网卡
export NCCL_IB_DISABLE=1                 # 禁用InfiniBand

# 启动资源监控（后台运行）
monitor_resources > monitor.log 2>&1 &

MONITOR_PID=$!

# 捕获退出信号
cleanup() {
    echo "终止资源监控进程..."
    kill $MONITOR_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# 启动训练任务
echo "启动训练任务..."
load_path=/data/checkpoint/2201210064/experiments/Finetune_shhs1_cosine_backbone_large_patch200_l1_finetune_all_time_swin_dual/version_5/ModelCheckpoint-epoch=02-val_acc=0.8560-val_macro=0.7387-val_score=6.5947.ckpt

ulimit -n 65535

#python main.py with finetune_ums  ums_bjnz_spo2_ods_F3_new_dataset fusion_2_concat_layes  \
#  num_gpus=6 num_nodes=1 num_workers=96 batch_size=64 model_arch=backbone_large_patch200  lr_mult=20 \
#  warmup_lr=0 val_check_interval=0.25 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
#  lr=1e-3 min_lr=0 random_choose_channels=8 max_epoch=75 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
#  load_path=$load_path  \
#  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
#  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
#  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' actual_channels='shhs_eeg'

#python3 main_kfold.py with finetune_ums  ums_l40_spo2_ods_F3_new_dataset fusion_2_concat_layes \
#  num_gpus=6 num_nodes=1 num_workers=96 batch_size=80 model_arch=backbone_large_patch200  lr_mult=20 \
#  warmup_lr=0 val_check_interval=1.0 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=20 decoder_features=768 pool=None \
#  lr=1.5e-3 min_lr=0 random_choose_channels=8 max_epoch=60 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=20 \
#  load_path=$load_path \
#  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
#  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
#  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' \
#  resume_during_training=0 resume_ckpt_path="" kfold=20 use_triton=False \

python3 main_test_kfold_persub.py with visualization_rem_shhs l40_SHHS1_datasets \
  num_gpus=$NUM_GPUS num_nodes=1 num_workers=$TOTAL_CPUS batch_size=1024 model_arch=backbone_large_patch200  lr_mult=20 \
  warmup_lr=0 val_check_interval=1.0 check_val_every_n_epoch=1 limit_train_batches=1.0 max_steps=-1 all_time=True time_size=1 decoder_features=768 pool=None \
  lr=1.5e-3 min_lr=0 random_choose_channels=8 max_epoch=60 lr_policy=cosine loss_function='l1' drop_path_rate=0.5 warmup_steps=0.1 split_len=1 \
  load_path=$load_path \
  use_pooling='swin' use_relative_pos_emb=False  mixup=0 smoothing=0.1 decoder_heads=16  use_global_fft=True use_all_label='all' \
  use_multiway="multiway" use_g_mid=False get_param_method='layer_decay'  local_pooling=False optim="adamw" poly=False weight_decay=0.05 \
  layer_decay=0.75 Lambda=1.0 patch_size=200 use_cb=True grad_name='all' \
  resume_during_training=0 resume_ckpt_path="" kfold=20 use_triton=False  actual_channels='shhs' \

# 训练完成后清理
cleanup