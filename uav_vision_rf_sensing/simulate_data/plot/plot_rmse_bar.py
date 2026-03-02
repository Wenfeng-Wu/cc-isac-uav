import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

'''
labels = ['Vision-Only',
         'Echo-Only(snr= 0)',
         'Echo-Only(snr=-1)',
         'Echo-Only(snr=-2)',
         'Multi-Modal(snr= 0)',
         'Multi-Modal(snr=-1)',
         'Multi-Modal(snr=-2)',
         'Echo-Only(few error)',
         'Echo-Only(all error)',
         'Multi-Modal(few error)',
         'Multi-Modal(lot error)',
         ]
'''

import numpy as np
import matplotlib.pyplot as plt

labels = ['a', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'b4', 'b5', 'c4', 'c5']
az = [0.5051, 0.5236, 0.5823, 0.6467, 0.3806, 0.4210, 0.4580, 0.5315, 0.5388, 0.3832, 0.3832]
el = [1.6321, 0.7648, 0.7711, 0.8377, 0.7396, 0.7360, 0.7772, 0.7896, 0.8216, 0.7569, 0.7918]

# 自定义参数
bar_color_az = '#FFBF4C'
bar_color_el = '#97AF64'
font_size_title = 18
font_size_label = 18
font_size_tick = 18
bar_width = 0.8

# 创建画布
fig = plt.figure(figsize=(6, 7))
x = np.arange(len(labels))

# 定义坐标轴位置参数（确保对齐）
left, width = 0.1, 0.8
height_lower = 0.25  # 下方轴高度
height_upper = 0.1   # 上方轴高度
spacing = 0.05       # 断裂处间距

# 创建上方Azimuth图
ax1 = fig.add_axes([left, 0.62, width, 0.3])  # 位置：[左, 下, 宽, 高]

# 创建Elevation的两个断裂轴（下方和上方）
ax2_lower = fig.add_axes([left, 0.1, width, height_lower])  # 下方轴（0.4-1.0）
ax2_upper = fig.add_axes([left, 0.1 + height_lower + spacing, width, height_upper], sharex=ax2_lower)  # 上方轴（1.6-1.7）
plt.setp(ax2_upper.get_xticklabels(), visible=False)  # 隐藏上方轴的x刻度标签

# 绘制Azimuth图
ax1.bar(x, az, width=bar_width, color=bar_color_az)
ax1.set_title('Azimuth', fontsize=font_size_title)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=font_size_tick)
ax1.set_ylabel('RMSE', fontsize=font_size_label)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim([0.2, 0.7])

# 绘制Elevation下方轴（0.4-1.0）
ax2_lower.bar(x, el, width=bar_width, color=bar_color_el)
ax2_lower.set_xticks(x)
ax2_lower.set_xticklabels(labels, fontsize=font_size_tick)
ax2_lower.set_ylabel('RMSE', fontsize=font_size_label)
ax2_lower.grid(axis='y', linestyle='--', alpha=0.7)
ax2_lower.set_ylim([0.6, 0.9])  # 下方显示范围

# 绘制Elevation上方轴（1.6-1.7）
ax2_upper.bar(x, el, width=bar_width, color=bar_color_el)
ax2_upper.set_title('Elevation', fontsize=font_size_title)
ax2_upper.grid(axis='y', linestyle='--', alpha=0.7)
ax2_upper.set_ylim([1.6, 1.7])  # 上方显示范围

# 添加断裂标记（//）
ax2_lower.text(0.5, 1.01, '//', ha='center', va='bottom',
               transform=ax2_lower.transAxes, fontsize=14, fontweight='bold')
ax2_upper.text(0.5, -0.05, '//', ha='center', va='top',
               transform=ax2_upper.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fig6.png')
plt.show()

'''
这个代码中第二个图需要修改，纵坐标的1.0到1.6能不能折叠起来？
labels = ['a',   'b1',  'b2',   'b3',  'c1',    'c2',   'c3',  'b4',   'b5',   'c4',   'c5']
az = [0.5051, 0.5236, 0.5823, 0.6467, 0.3806, 0.4210, 0.4580, 0.5315, 0.5388, 0.3832, 0.3832]

el = [1.6321, 0.7648, 0.7711, 0.8377, 0.7396, 0.7360, 0.7772, 0.7896, 0.8216, 0.7569, 0.7918]

# 自定义参数 - 您可以根据需要修改这些值
bar_color_az = '#FFBF4C'    # az柱状图颜色
bar_color_el = '#97AF64'    # el柱状图颜色
font_size_title = 18        # 标题字体大小
font_size_label = 18        # 坐标轴标签字体大小
font_size_tick = 18         # 刻度字体大小
bar_width = 0.8             # 柱子宽度

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
x = np.arange(len(labels))  # 标签位置

# 绘制az柱状图
ax1.bar(x, az, width=bar_width, color=bar_color_az)
ax1.set_title('Azimuth', fontsize=font_size_title)
ax1.set_xticks(x)
ax1.set_xticklabels(labels,  fontsize=font_size_tick)
ax1.set_ylabel('RMSE', fontsize=font_size_label)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim([0.2, 1.2])

# 绘制el柱状图
ax2.bar(x, el, width=bar_width, color=bar_color_el)
ax2.set_title('Elevation', fontsize=font_size_title)
ax2.set_xticks(x)
ax2.set_xticklabels(labels,  fontsize=font_size_tick)
ax2.set_ylabel('RMSE', fontsize=font_size_label)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_ylim([0.4, 1.7])

# 调整布局，避免文字重叠
plt.tight_layout()

plt.savefig('fig6.png')
# 显示图形
plt.show()
'''