import os
import shutil
from pathlib import Path
import json
name = '第二批'
root_path = Path(os.path.join('/Users/hwx_admin/Downloads', name, "length_summary.json"))
with open(root_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# 更新 overlapping 字段（秒）
for item in data_list:
    if 'overlapping' in item:
        item['overlapping'] = item['overlapping'] * 30

# （可选）写回原文件或另存为新文件
with open(root_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

# name = '第一批'
# root_path = Path(os.path.join('/Users/hwx_admin/Downloads', name, "data"))
#
# dst_root = Path('/Volumes/T7/data/ums/pic')
#
# for parent in root_path.iterdir():
#     if not parent.is_dir() or parent.name == "collected":
#         continue
#     for subfolder in parent.iterdir():
#         if subfolder.is_dir():
#             target_dir = dst_root / parent.name
#             target_dir.mkdir(parents=True, exist_ok=True)
#
#             for f in subfolder.iterdir():
#                 if f.is_file():
#                     # 防止重名冲突：加入上层目录名
#                     new_name = f"{parent.name}_{f.name}"
#                     dst_file = target_dir / new_name
#                     print(f"Moving: {f} --> {dst_file}")
#                     shutil.move(str(f), dst_file)  # 或 shutil.copy2 复制