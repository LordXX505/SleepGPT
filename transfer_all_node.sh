#!/bin/bash

# 检查是否传递了两个参数
if [ $# -ne 2 ]; then
  echo "Usage: $0 <file_path> <target_directory>"
  exit 1
fi

# 设置源文件路径和目标目录路径
FILE_PATH=$1
TARGET_DIR=$2

# 设置目标机器的用户名和 IP 地址或主机名
TARGET_USER="root"   # 目标机器的用户名
TARGET_HOST="172.17.172.14"  # 目标机器的 IP 地址或主机名

# 使用 scp 进行文件传输
scp -r "$FILE_PATH" "${TARGET_USER}@${TARGET_HOST}:${TARGET_DIR}"

# 检查 scp 命令是否成功
if [ $? -eq 0 ]; then
  echo "File transfer successful"
else
  echo "File transfer failed"
  exit 1
fi