#!/bin/bash

#NODE=("172.17.0.3")
#Nodename=("node1")
#
#list1="1"
#list1_x=($list1)
#length=${#list1_x[@]}

NODE=("172.17.0.3" "172.17.0.17" "172.17.0.15" "172.17.0.7")
Nodename=("node1" "node2" "node3" "node4")

list1="1 2 3 4"
list1_x=($list1)
length=${#list1_x[@]}

for ((i=0; i<${length}; i++));
do
    ssh root@${NODE[$i]} "cd /home/hwx/Sleep && sh ./Test_shhs1.sh 4 $i" > ./checkpoint/output_${Nodename[$i]}.log 2>&1 &
done

