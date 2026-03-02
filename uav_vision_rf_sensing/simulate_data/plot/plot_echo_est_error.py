import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 设置文件夹路径
folder1 = r'C:\FengFeng\Muilty_Modal_fusion\uav_vision_assisted\code\dataset\uav_gps_beam_process\uav_bs_coordinate_bearing315_dis100'
folder2 = r'C:\FengFeng\Muilty_Modal_fusion\uav_vision_assisted\code\dataset\uav_gps_beam_process\uav_bs_echo_coor_bearing315_dis100_snr0_p100'

# 确保文件夹存在
if not os.path.exists(folder1):
    print(f"错误：文件夹 '{folder1}' 不存在")
    exit()

if not os.path.exists(folder2):
    print(f"错误：文件夹 '{folder2}' 不存在")
    exit()

# 获取folder1中的所有txt文件
files1 = [f for f in os.listdir(folder1) if f.endswith('.txt')]

if not files1:
    print(f"错误：文件夹 '{folder1}' 中没有找到txt文件")
    exit()

# 存储所有差值
all_diffs = []
file_diffs = []  # 存储每个文件的4个差值
failed_files = []  # 存储处理失败的文件

print(f"在 '{folder1}' 中找到 {len(files1)} 个txt文件")
print(f"开始处理文件...")

# 遍历folder1中的所有txt文件
for file in files1:
    file_path1 = os.path.join(folder1, file)
    file_path2 = os.path.join(folder2, file)

    # 检查folder2中是否存在同名文件
    if not os.path.exists(file_path2):
        failed_files.append(f"文件 '{file}' 在第二个文件夹中不存在")
        continue

    try:
        # 读取第一个文件的数据 - 新方法处理多值行
        with open(file_path1, 'r') as f1:
            # 读取所有行并合并
            lines = f1.readlines()
            # 合并所有行并分割成单个值
            all_values = []
            for line in lines:
                # 移除换行符并分割
                values = line.strip().split()
                all_values.extend(values)

            # 只取前4个有效数值
            data1 = []
            for val in all_values:
                try:
                    data1.append(float(val))
                    if len(data1) >= 4:
                        break
                except ValueError:
                    continue

            if len(data1) < 4:
                failed_files.append(f"文件 '{file}' 中有效数据不足4个 (找到 {len(data1)} 个)")
                continue

        # 读取第二个文件的数据 - 同样方法
        with open(file_path2, 'r') as f2:
            lines = f2.readlines()
            all_values = []
            for line in lines:
                values = line.strip().split()
                all_values.extend(values)

            data2 = []
            for val in all_values:
                try:
                    data2.append(float(val))
                    if len(data2) >= 4:
                        break
                except ValueError:
                    continue

            if len(data2) < 4:
                failed_files.append(f"文件 '{file}' 在第二个文件夹中有效数据不足4个 (找到 {len(data2)} 个)")
                continue

        # 计算差值
        diffs = [(d1 - d2)*0.01745 for d1, d2 in zip(data1[:4], data2[:4])]
        file_diffs.append(diffs)
        all_diffs.extend(diffs)

    except Exception as e:
        failed_files.append(f"处理文件 '{file}' 时出错: {str(e)}")

# 检查是否有成功处理的数据
if not file_diffs:
    print("错误：没有成功处理任何文件")
    if failed_files:
        print("\n处理失败的文件:")
        for msg in failed_files:
            print(f"  - {msg}")
    exit()

# 将数据转换为NumPy数组以便处理
file_diffs = np.array(file_diffs)
all_diffs = np.array(all_diffs)

# 提取每个位置的差值
diff_pos1 = file_diffs[:, 0]
diff_pos2 = file_diffs[:, 1]
diff_pos3 = file_diffs[:, 2]
diff_pos4 = file_diffs[:, 3]

# 打印统计信息
print("\n数据处理完成:")
print(f"成功处理文件数: {len(file_diffs)}")
print(f"总数据点数: {len(all_diffs)}")
print(f"位置1差值统计: 均值={np.mean(diff_pos1):.4f}, 标准差={np.std(diff_pos1):.4f}")
print(f"位置2差值统计: 均值={np.mean(diff_pos2):.4f}, 标准差={np.std(diff_pos2):.4f}")
print(f"位置3差值统计: 均值={np.mean(diff_pos3):.4f}, 标准差={np.std(diff_pos3):.4f}")
print(f"位置4差值统计: 均值={np.mean(diff_pos4):.4f}, 标准差={np.std(diff_pos4):.4f}")

if failed_files:
    print(f"\n处理失败的文件数: {len(failed_files)}")
    print("前10个失败文件:")
    for msg in failed_files[:10]:
        print(f"  - {msg}")
    if len(failed_files) > 10:
        print(f"  ... 共 {len(failed_files)} 个失败文件")

# 创建图表
plt.figure(figsize=(16, 12))

# ==================== 直方图 (2x2网格) ====================
plt.subplot(2, 2, 1)
sns.histplot(diff_pos1, bins=30, color='royalblue', kde=True)
plt.title('位置1差值分布')
plt.xlabel('差值')
plt.ylabel('频率')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 2)
sns.histplot(diff_pos2, bins=30, color='tomato', kde=True)
plt.title('位置2差值分布')
plt.xlabel('差值')
plt.ylabel('频率')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 3)
sns.histplot(diff_pos3, bins=30, color='mediumseagreen', kde=True)
plt.title('位置3差值分布')
plt.xlabel('差值')
plt.ylabel('频率')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 4)
sns.histplot(diff_pos4, bins=30, color='darkorchid', kde=True)
plt.title('位置4差值分布')
plt.xlabel('差值')
plt.ylabel('频率')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('echo_difference_histograms.png', dpi=300)
print("直方图已保存为 'echo_difference_histograms.png'")

# ==================== 箱线图 ====================
plt.figure(figsize=(12, 8))
data = [diff_pos1, diff_pos2, diff_pos3, diff_pos4]
labels = ['位置1', '位置2', '位置3', '位置4']
colors = ['royalblue', 'tomato', 'mediumseagreen', 'darkorchid']

# 创建箱线图
boxplot = plt.boxplot(data, patch_artist=True, labels=labels)

# 设置箱线图颜色
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置其他元素颜色
for element in ['whiskers', 'caps', 'medians']:
    for idx, line in enumerate(boxplot[element]):
        line.set_color(colors[idx // 2])  # 每两个元素属于一组

plt.title('四个位置差值的箱线图比较')
plt.ylabel('差值')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加统计信息
stats_text = (f"位置1: 均值={np.mean(diff_pos1):.4f}, 标准差={np.std(diff_pos1):.4f}\n"
              f"位置2: 均值={np.mean(diff_pos2):.4f}, 标准差={np.std(diff_pos2):.4f}\n"
              f"位置3: 均值={np.mean(diff_pos3):.4f}, 标准差={np.std(diff_pos3):.4f}\n"
              f"位置4: 均值={np.mean(diff_pos4):.4f}, 标准差={np.std(diff_pos4):.4f}")

plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
         ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('echo_difference_boxplots.png', dpi=300)
print("箱线图已保存为 'echo_difference_boxplots.png'")

# 显示所有图表
plt.show()
diff_pos1 = file_diffs[:, 0]
diff_pos2 = file_diffs[:, 1]
az_diff = np.abs(diff_pos1)
az_diff = np.round(az_diff).astype(int)
el_diff = np.abs(diff_pos2)
el_diff = np.round(el_diff).astype(int)
maxdiff = np.maximum(az_diff, el_diff)

# 3. 统计最大差值频率
max_val = int(np.max(maxdiff))  # 最大差值
counts = np.zeros(max_val + 2, dtype=int)  # 创建足够长的数组（包含0到max_val+1）

for val in maxdiff:
    if val < len(counts):
        counts[val] += 1

# 计算频率（百分比）
total = len(maxdiff)
frequencies = (counts / total) * 100

# 4. 绘制柱状图
plt.figure(figsize=(15, 5))
bars = plt.bar(range(1, len(frequencies) + 1), frequencies,
               color='skyblue', edgecolor='black')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom')

# 设置图表属性
plt.xlabel('degree', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Minimum beam width that can cover the target', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(frequencies) + 1),
           [f'{(i*2+1)}' for i in range(len(frequencies))])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 设置坐标轴范围
plt.xlim(0.5, len(frequencies) + 0.5)
plt.ylim(0, max(frequencies) * 1.2)

plt.tight_layout()
plt.show()

# 输出统计结果
print("="*45)
print(f"{'Error Value':<12} | {'Count':<8} | {'Percentage':<10}")
print("-"*45)
for i, (count, freq) in enumerate(zip(counts, frequencies)):
    print(f"{i:<12} | {count:<8} | {freq:.1f}%")
print("="*45)

# YOLOv5性能指标数据
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [0.985, 0.98, 0.98, 0.98]  # 您的实际指标值

# 创建柱状图
plt.figure(figsize=(8, 4))
bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
               edgecolor='black', width=0.7)

# 设置y轴范围以突出显示高值区域
plt.ylim(0.9, 1)

# 添加图表标题和标签
plt.title('YOLOv5 Performance Metrics', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Metrics', fontsize=12, labelpad=10)
plt.ylabel('Score', fontsize=12, labelpad=10)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加背景色
plt.gca().set_facecolor('#f0f0f0')
plt.gcf().set_facecolor('#f5f5f5')


# 添加自定义样式
plt.xticks(fontsize=11, fontweight='bold')

# 添加边框
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


plt.tight_layout()
plt.show()