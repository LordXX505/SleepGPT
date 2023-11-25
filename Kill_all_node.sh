#!/bin/bash

# 定义节点列表
NODE=("172.17.0.3" "172.17.0.17" "172.17.0.15" "172.17.0.7")
Nodename=("node1" "node2" "node3" "node4")

list1="1 2 3 4"
list1_x=($list1)
length=${#list1_x[@]}

for ((i=0; i<${length}; i++));
# 在每个节点上杀死 start.sh 相关的进程
do
    ssh root@${NODE[$i]} "pkill -f run_with_torch_ddp_run.sh" > ./checkpoint/kill_output_${Nodename[$i]}.log 2>&1 &
    ssh root@${NODE[$i]} "fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print \"kill -9 \" \$i;}' | sh" >> ./checkpoint/kill_output_${Nodename[$i]}.log 2>&1 &
done

wait
