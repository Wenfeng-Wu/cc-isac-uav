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
         'MMFE(snr= 0)',
         'MMFE(snr=-1)',
         'MMFE(snr=-2)',
         'Echo-Only(few error)',
         'Echo-Only(all error)',
         'MMFE(few error)',
         'MMFE(lot error)',
         'KL-based(snr= 0)',
         'MMFE w/o Fuse(snr= 0)'
         ]
'''

import numpy as np
import matplotlib.pyplot as plt

#labels = ['a', 'b1',   'b2',   'b3',    'c1',   'c2',   'c3', 'd1',    'd2',  'd3',   'e1',  'e2',  'e3',
#          'b4',   'b5',   'c4',   'c5',   'd4', 'd5', 'e4','e5',
#          'b6', 'b7', 'b8', 'c6', 'c7', 'c8']
#az = [0.5051, 0.5236, 0.5823, 0.6467, 0.3806, 0.4210, 0.4580, 0.8405, 0.9943, 1.0806, 0.4459,0.5061,0.5683 ,
#      0.5315, 0.5388, 0.3831, 0.3832,  0.8300, 0.8023,0.4473,0.4411,
#      0.5179, 0.5466, 0.5762, 0.3728, 0.3952, 0.4209]
#el = [1.6321, 0.7648, 0.7711, 0.8377, 0.7396, 0.7360, 0.7772, 0.9369, 1.0568, 1.1561, 0.8276, 0.8261, 0.8598,
#      0.7896, 0.8216, 0.7569, 0.7523, 0.9270, 0.9231, 0.8422, 0.8290,
#      0.7813, 0.7886, 0.7761, 0.7702, 0.7523, 0.7230]

labels = ['a', 'b1',   'b2',   'b3',    'c1',   'c2',   'c3', 'd1',    'd2',  'd3',   'e1',  'e2',  'e3',
          'b4',   'b5',   'c4',   'c5',  'e4','e5',
          'b6', 'b7', 'b8', 'c6', 'c7', 'c8']
az = [0.5051, 0.5236, 0.5823, 0.6467, 0.3806, 0.4210, 0.4580, 0.8405, 0.9943, 1.0806, 0.4459,0.5061,0.5683 ,
      0.5315, 0.5388, 0.3831, 0.3832, 0.4473,0.4411,
      0.5179, 0.5466, 0.5762, 0.3728, 0.3952, 0.4209]
el = [1.6321, 0.7648, 0.7711, 0.8377, 0.7396, 0.7360, 0.7772, 0.9369, 1.0568, 1.1561, 0.8276, 0.8261, 0.8598,
      0.7896, 0.8216, 0.7569, 0.7523, 0.8422, 0.8290,
      0.7813, 0.7886, 0.7761, 0.7702, 0.7523, 0.7230]

# 自定义参数
bar_color_az = '#FFBF4C'
bar_color_el = '#97AF64'
font_size_title = 16
font_size_label = 16
font_size_tick = 16
bar_width = 0.8

# 创建画布
fig = plt.figure(figsize=(10, 7))
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
ax1.set_ylim([0.2, 1.1])

# 绘制Elevation下方轴（0.4-1.0）
ax2_lower.bar(x, el, width=bar_width, color=bar_color_el)
ax2_lower.set_xticks(x)
ax2_lower.set_xticklabels(labels, fontsize=font_size_tick)
ax2_lower.set_ylabel('RMSE', fontsize=font_size_label)
ax2_lower.grid(axis='y', linestyle='--', alpha=0.7)
ax2_lower.set_ylim([0.6, 1.2])  # 下方显示范围

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
plt.savefig('fig6.png', dpi=900)
plt.show()



#=====================

# 创建画布
fig = plt.figure(figsize=(12,4))
x = np.arange(len(labels))

# 左右布局
bottom = 0.15
lower_h = 0.54
upper_h = lower_h / 6
gap = 0.04

height = lower_h + upper_h + gap

# 左侧 Azimuth
ax1 = fig.add_axes([0.07, bottom, 0.40, height])

# 右侧 Elevation（断裂比例 1:6）
lower_h = 0.54
upper_h = lower_h / 6
gap = 0.04

ax2_lower = fig.add_axes([0.55, bottom, 0.40, lower_h])
ax2_upper = fig.add_axes([0.55, bottom + lower_h + gap, 0.40, upper_h], sharex=ax2_lower)

plt.setp(ax2_upper.get_xticklabels(), visible=False)

# ---------------- Azimuth ----------------
ax1.bar(x, az, width=bar_width, color=bar_color_az)
ax1.set_title('Azimuth', fontsize=font_size_title)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=font_size_tick, rotation=90)
ax1.set_ylabel('RMSE', fontsize=font_size_label)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim([0.2, 1.1])

# ---------------- Elevation lower ----------------
ax2_lower.bar(x, el, width=bar_width, color=bar_color_el)
ax2_lower.set_xticks(x)
ax2_lower.set_xticklabels(labels, fontsize=font_size_tick, rotation=90)
ax2_lower.grid(axis='y', linestyle='--', alpha=0.7)
ax2_lower.set_ylim([0.6, 1.2])

# ---------------- Elevation upper ----------------
ax2_upper.bar(x, el, width=bar_width, color=bar_color_el)
ax2_upper.set_title('Elevation', fontsize=font_size_title)
ax2_upper.grid(axis='y', linestyle='--', alpha=0.7)
ax2_upper.set_ylim([1.6, 1.7])

# 断裂标记
ax2_lower.text(0.5, 1.02, '//', ha='center', va='bottom',
               transform=ax2_lower.transAxes, fontsize=14, fontweight='bold')

ax2_upper.text(0.5, -0.08, '//', ha='center', va='top',
               transform=ax2_upper.transAxes, fontsize=14, fontweight='bold')

# Elevation整体RMSE标签（居中）
fig.text(0.515, bottom + height/2, 'RMSE',
         rotation=90, va='center', ha='center', fontsize=font_size_label)

plt.savefig('fig6.png', dpi=900)
plt.show()