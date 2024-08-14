import boto3
import re
import os
from tqdm import tqdm
from botocore.config import Config

# 初始化 S3 客户端
s3 = boto3.client('s3')

# 配置参数
bucket_name = 'arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-psg-access-point'  # 这里假设访问点名称遵循标准格式
prefix = 'PSG/bids/'  # 设置为你需要下载的对象的前缀
local_path = '/data'  # 替换为你本地存储文件的路径

# 编译正则表达式模式，用于匹配指定范围内的对象
pattern = re.compile(r'PSG/bids/sub-S000112\d{7}/*')

# 创建本地存储路径（如果不存在）
if not os.path.exists(local_path):
    os.makedirs(local_path)

# 初始化 S3 客户端，使用访问点 ARN
config = Config(
    s3={
        'addressing_style': 'virtual'
    }
)
s3 = boto3.client('s3', config=config)

# 使用分页器列出所有符合前缀的对象
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

# 下载匹配的对象
for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        if pattern.match(key):
            # 创建本地文件路径
            local_file_path = os.path.join(local_path, key)
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            # 获取文件大小
            response = s3.head_object(Bucket=bucket_name, Key=key)
            total_size = response['ContentLength']

            # 下载文件并显示进度条
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=key) as pbar:
                def progress_callback(bytes_transferred):
                    pbar.update(bytes_transferred)

                s3.download_file(bucket_name, key, local_file_path, Callback=progress_callback)

print("Download complete.")