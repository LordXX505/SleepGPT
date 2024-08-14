
#!/bin/bash

# 启动的进程数
NUM_PROCESSES=64

# Python 脚本路径
PYTHON_SCRIPT="Processing_shhs_hdf5.py"

# 记录当前启动的进程数
CURRENT_PROCESSES=0

# 启动多个 Python 进程
for (( i=0; i<$NUM_PROCESSES; i++ ))
do
  echo "Starting process $i"
  python $PYTHON_SCRIPT $i &
  CURRENT_PROCESSES=$((CURRENT_PROCESSES+1))

  # 如果达到最大并行进程数，等待所有进程结束
  if [ "$CURRENT_PROCESSES" -ge "$NUM_PROCESSES" ]; then
    wait
    CURRENT_PROCESSES=0
  fi
done

# 等待所有后台进程结束
wait

echo "All processes are done."
