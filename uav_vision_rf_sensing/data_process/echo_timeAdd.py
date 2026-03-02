import os
import re

# 原始文件夹路径
source_dir = r"D:\Feng_Feng\dataset\uav_gps_beam_process\uav_bs_echo_coor_bearing315_dis100"
# 目标文件夹路径
target_dir = r"D:\Feng_Feng\dataset\uav_gps_beam_process\uav_bs_echo_timeAdd_coor_bearing315_dis100"

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

# 获取原始文件夹中所有coordinate_x.txt文件
files = [f for f in os.listdir(source_dir) if re.match(r'coordinate_(\d+)\.txt', f)]

# 提取文件序号并排序
file_indices = []
for f in files:
    match = re.match(r'coordinate_(\d+)\.txt', f)
    if match:
        file_indices.append(int(match.group(1)))
file_indices.sort()

# 处理文件：计算相邻文件的平均值
for i in range(len(file_indices) - 1):
    # 当前文件序号
    current_idx = file_indices[i]
    next_idx = file_indices[i + 1]

    # 读取当前文件数据
    current_file = os.path.join(source_dir, f'coordinate_{current_idx}.txt')
    with open(current_file, 'r') as f:
        data_current = [float(x) for x in f.read().split()]

    # 读取下一个文件数据
    next_file = os.path.join(source_dir, f'coordinate_{next_idx}.txt')
    with open(next_file, 'r') as f:
        data_next = [float(x) for x in f.read().split()]

    # 计算平均值（保留原始小数格式）
    avg_data = []
    for j in range(4):
        avg_value = (data_current[j] + data_next[j]) / 2.0
        # 保留原始数据格式（整数或浮点）
        if avg_value.is_integer():
            avg_data.append(str(int(avg_value)))
        else:
            avg_data.append(str(avg_value))

    # 写入新文件（文件名使用当前序号）
    new_file = os.path.join(target_dir, f'coordinate_{current_idx}.txt')
    with open(new_file, 'w') as f:
        f.write(' '.join(avg_data))

print(f"处理完成！共生成 {len(file_indices) - 1} 个新文件到目标文件夹。")