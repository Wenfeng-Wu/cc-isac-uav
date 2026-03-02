from data_process.set_dataset_time import uav_dataset_set
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# 查看时隙前后 各参数的差值
train_loader, test_loader = uav_dataset_set()

# 初始化列表来累积所有批次的数据
all_az_diffs = []
all_el_diffs = []
all_d_diffs = []
all_v_diffs = []

for i, batch in enumerate(train_loader):
    index = batch['mask_index']
    images = batch['mask_images']
    boxs = batch['target_boxs']
    bs_coors_true = batch['uav_bs_coors_true']
    echo_coors_esti = batch['uav_echo_coors_esti']

    az_true = bs_coors_true[:, :, 0]
    el_true = bs_coors_true[:, :, 1]
    d_true = bs_coors_true[:, :, 2]
    v_true = bs_coors_true[:, :, 3]

    # 计算相邻差值
    az_diff = torch.round(torch.diff(az_true, dim=1)).reshape(-1)
    el_diff = torch.round(torch.diff(el_true, dim=1)).reshape(-1)
    d_diff = (torch.diff(d_true, dim=1)).reshape(-1)
    v_diff = (torch.diff(v_true, dim=1)).reshape(-1)

    # 累积当前批次的数据
    all_az_diffs.append(az_diff)
    all_el_diffs.append(el_diff)
    all_d_diffs.append(d_diff)
    all_v_diffs.append(v_diff)

# 合并所有批次的数据
all_az_diffs = torch.cat(all_az_diffs, dim=0)
all_el_diffs = torch.cat(all_el_diffs, dim=0)
all_d_diffs = torch.cat(all_d_diffs, dim=0)
all_v_diffs = torch.cat(all_v_diffs, dim=0)

# 转换为NumPy数组以便绘图
az_diffs_np = all_az_diffs.cpu().numpy()
el_diffs_np = all_el_diffs.cpu().numpy()
d_diffs_np = all_d_diffs.cpu().numpy()
v_diffs_np = all_v_diffs.cpu().numpy()

# 绘制直方图
plt.figure(figsize=(12, 6))

# 方位角差值直方图
plt.subplot(1, 4, 1)
plt.hist(az_diffs_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Azimuth Differences Distribution')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加统计信息
mean_az = np.mean(az_diffs_np)
std_az = np.std(az_diffs_np)
plt.axvline(mean_az, color='red', linestyle='dashed', linewidth=1)
plt.text(mean_az + 0.1 * std_az, plt.ylim()[1] * 0.9, f'Mean: {mean_az:.4f}\nStd: {std_az:.4f}',
         color='red', fontsize=10)

# 俯仰角差值直方图
plt.subplot(1, 4, 2)
plt.hist(el_diffs_np, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Elevation Differences Distribution')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加统计信息
mean_el = np.mean(el_diffs_np)
std_el = np.std(el_diffs_np)
plt.axvline(mean_el, color='red', linestyle='dashed', linewidth=1)
plt.text(mean_el + 0.1 * std_el, plt.ylim()[1] * 0.9, f'Mean: {mean_el:.4f}\nStd: {std_el:.4f}',
         color='red', fontsize=10)

# 俯仰角差值直方图
plt.subplot(1, 4, 3)
plt.hist(d_diffs_np, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('distance Differences Distribution')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加统计信息
mean_d = np.mean(d_diffs_np)
std_d = np.std(d_diffs_np)
plt.axvline(mean_d, color='red', linestyle='dashed', linewidth=1)
plt.text(mean_d + 0.1 * std_d, plt.ylim()[1] * 0.9, f'Mean: {mean_d:.4f}\nStd: {std_d:.4f}',
         color='red', fontsize=10)

# 俯仰角差值直方图
plt.subplot(1, 4, 4)
plt.hist(v_diffs_np, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('v Differences Distribution')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加统计信息
mean_v = np.mean(v_diffs_np)
std_v = np.std(el_diffs_np)
plt.axvline(mean_v, color='red', linestyle='dashed', linewidth=1)
plt.text(mean_v + 0.1 * std_v, plt.ylim()[1] * 0.9, f'Mean: {mean_v:.4f}\nStd: {std_v:.4f}',
         color='red', fontsize=10)



plt.tight_layout()
plt.savefig('Time_param_differences_histogram.png', dpi=300)
plt.show()

# 创建图形和子图
plt.figure(figsize=(16, 6))

# 颜色方案
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
boxprops = dict(linewidth=1.5, color='black')
whiskerprops = dict(linewidth=1.5, color='gray')
capprops = dict(linewidth=1.5, color='gray')
medianprops = dict(linewidth=2.5, color='red')
flierprops = dict(marker='o', markerfacecolor='lightgray', markersize=5,
                  markeredgecolor='gray', alpha=0.7)

# 方位角差值箱线图
plt.subplot(1, 4, 1)
bp = plt.boxplot(az_diffs_np, patch_artist=True,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops,
                flierprops=flierprops)
# 设置箱体颜色
for box in bp['boxes']:
    box.set(facecolor=colors[0], alpha=0.8)
plt.title('Azimuth Differences', fontsize=14)
plt.ylabel('Difference Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks([1], ['Azimuth'])

# 添加统计信息
mean_az = np.mean(az_diffs_np)
std_az = np.std(az_diffs_np)
plt.text(0.5, 0.95, f'Mean: {mean_az:.4f}\nStd: {std_az:.4f}',
         transform=plt.gca().transAxes, ha='center', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

# 俯仰角差值箱线图
plt.subplot(1, 4, 2)
bp = plt.boxplot(el_diffs_np, patch_artist=True,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops,
                flierprops=flierprops)
for box in bp['boxes']:
    box.set(facecolor=colors[1], alpha=0.8)
plt.title('Elevation Differences', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks([1], ['Elevation'])

# 添加统计信息
mean_el = np.mean(el_diffs_np)
std_el = np.std(el_diffs_np)
plt.text(0.5, 0.95, f'Mean: {mean_el:.4f}\nStd: {std_el:.4f}',
         transform=plt.gca().transAxes, ha='center', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

# 距离差值箱线图
plt.subplot(1, 4, 3)
bp = plt.boxplot(d_diffs_np, patch_artist=True,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops,
                flierprops=flierprops)
for box in bp['boxes']:
    box.set(facecolor=colors[2], alpha=0.8)
plt.title('Distance Differences', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks([1], ['Distance'])

# 添加统计信息
mean_d = np.mean(d_diffs_np)
std_d = np.std(d_diffs_np)
plt.text(0.5, 0.95, f'Mean: {mean_d:.4f}\nStd: {std_d:.4f}',
         transform=plt.gca().transAxes, ha='center', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

# 速度差值箱线图
plt.subplot(1, 4, 4)
bp = plt.boxplot(v_diffs_np, patch_artist=True,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops,
                flierprops=flierprops)
for box in bp['boxes']:
    box.set(facecolor=colors[3], alpha=0.8)
plt.title('Velocity Differences', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks([1], ['Velocity'])

# 添加统计信息
mean_v = np.mean(v_diffs_np)
std_v = np.std(v_diffs_np)  # 修正：使用v_diffs_np自己的标准差
plt.text(0.5, 0.95, f'Mean: {mean_v:.4f}\nStd: {std_v:.4f}',
         transform=plt.gca().transAxes, ha='center', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

# 添加图例说明箱线图元素
legend_elements = [
    mpatches.Patch(facecolor='lightgray', edgecolor='gray', label='Outliers'),
    plt.Line2D([0], [0], color='red', lw=2.5, label='Median'),
    plt.Line2D([0], [0], color='black', lw=1.5, label='IQR'),
    plt.Line2D([0], [0], color='gray', lw=1.5, label='Whiskers')
]

# 添加整体标题
plt.suptitle('Statistical Distribution of Differences', fontsize=16, fontweight='bold')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为整体标题留出空间
plt.subplots_adjust(top=0.85)  # 调整顶部间距
plt.savefig('Time_param_differences_distribution.png', dpi=300)
# 显示图形
plt.show()

# 打印统计信息
print(f"Azimuth Differences:")
print(f"  Total samples: {len(az_diffs_np)}")
print(f"  Mean: {mean_az:.6f}")
print(f"  Standard deviation: {std_az:.6f}")
print(f"  Min: {np.min(az_diffs_np):.6f}")
print(f"  Max: {np.max(az_diffs_np):.6f}")

print(f"\nElevation Differences:")
print(f"  Total samples: {len(el_diffs_np)}")
print(f"  Mean: {mean_el:.6f}")
print(f"  Standard deviation: {std_el:.6f}")
print(f"  Min: {np.min(el_diffs_np):.6f}")
print(f"  Max: {np.max(el_diffs_np):.6f}")