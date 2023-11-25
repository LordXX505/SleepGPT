#!/bin/bash

# 定义节点列表
NODE=("172.17.0.3" "172.17.0.17" "172.17.0.15" "172.17.0.7")
Nodename=("node1" "node2" "node3" "node4")

list1="1 2 3 4"
list1_x=($list1)
length=${#list1_x[@]}

for ((i=0; i<${length}; i++));
# 启动每个节点上的 start.sh 脚本并重定向输出
do
    ssh root@${NODE[$i]} "cd /home/hwx/Sleep && sh ./Finetune_mass_sp.sh 4 $i" > ./checkpoint/output_${Nodename[$i]}.log 2>&1 &
done

