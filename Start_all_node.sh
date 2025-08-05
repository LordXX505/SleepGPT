#!/bin/bash

# 定义节点列表
NODE=("172.17.172.8" "111.229.84.10")
Nodename=("node1" "node2")

list1="1 2"
list1_x=($list1)
length=${#list1_x[@]}

# 定义conda安装路径
CONDA_PATH=/root/miniconda3/bin

# 在所有节点上启动新的 run_with_torch_ddp_run.sh 进程
total_nodes=${#NODE[@]}

for ((i=0; i<${length}; i++));
do
    current_node_index=$i
    if [ "${NODE[$i]}" == "172.17.172.8" ]; then
        (
            source ${CONDA_PATH}/activate torch
            if [ $? -ne 0 ]; then
                echo "Failed to activate conda environment on ${Nodename[$i]}" >> ./checkpoint_log/start_output_${Nodename[$i]}.log 2>&1
                exit 1
            fi
            nohup /root/Sleep/run_with_torch_ddp_run.sh ${total_nodes} ${current_node_index} >> ./checkpoint_log/start_output_${Nodename[$i]}.log 2>&1
            if [ $? -ne 0 ]; then
                echo "Failed to start process on ${Nodename[$i]}" >> ./checkpoint_log/start_output_${Nodename[$i]}.log 2>&1
            fi
        ) &
    else
        ssh root@${NODE[$i]} "
            source ${CONDA_PATH}/activate torch
            if [ $? -ne 0 ]; then
                echo 'Failed to activate conda environment on ${Nodename[$i]}' >> /root/Sleep/checkpoint_log/start_output_${Nodename[$i]}.log 2>&1
                exit 1
            fi
            nohup /root/Sleep/run_with_torch_ddp_run.sh ${total_nodes} ${current_node_index} >> /root/Sleep/checkpoint_log/start_output_${Nodename[$i]}.log 2>&1
            if [ $? -ne 0 ]; then
                echo 'Failed to start process on ${Nodename[$i]}' >> /root/Sleep/checkpoint_log/start_output_${Nodename[$i]}.log 2>&1
            fi
        " &
    fi
done

wait

echo "Processes started on all nodes"