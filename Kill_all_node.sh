#!/bin/bash

# 定义节点列表
NODE=("172.17.172.8" "172.17.172.14")
Nodename=("node1" "node2")

# 终止所有节点上的旧进程
for i in "${!NODE[@]}"; do
    ssh root@"${NODE[$i]}" "pkill -f run_with_torch_ddp_run.sh" > ./checkpoint/kill_output_${Nodename[$i]}.log 2>&1 &
    ssh root@"${NODE[$i]}" "fuser -v /dev/nvidia* | awk '{for(i=1;i<=NF;i++)print \"kill -9 \" \$i;}' | sh" >> ./checkpoint/kill_output_${Nodename[$i]}.log 2>&1 &
done

wait
