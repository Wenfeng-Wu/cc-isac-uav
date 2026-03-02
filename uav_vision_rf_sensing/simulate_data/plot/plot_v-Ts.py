import matplotlib.pyplot as plt
import matplotlib as mpl

# 全局字体设为 Times New Roman,统一字体大小
mpl.rcParams['font.family'] = 'serif'              # 先声明使用 serif
mpl.rcParams['font.serif'] = ['Times New Roman']   # 再把 serif 指向 Times New Roman
mpl.rcParams['mathtext.fontset'] = 'stix'          # 数学公式也用 Times 风格
mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size'       : 20,        # 全局默认
    'axes.titlesize'  : 20,
    'axes.labelsize'  : 20,
    'xtick.labelsize' : 20,
    'ytick.labelsize' : 20,
    'legend.fontsize' : 20,
})

"""
Δα vs Δt 曲线，并标出三条水平阈值线
"""
import numpy as np

# ---------- 参数 ----------
D, V = 100, 25                       # 距离(m)、速度(m/s)
DT = np.arange(0, 300)               # 横坐标：Δt(ms)

THRESHOLDS = 1 / 2 ** np.array([4, 5, 6])
LABELS = [r'21$\times$21(s=3)', r'32$\times$32(s=4)', r'64$\times$64(s=4)']
COLORS = ['#FF6F61', '#6B5B95', '#88B04B']

# ---------- 计算 ----------
delta_alpha = np.arctan(V * DT / (1000 * D))

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(DT, delta_alpha, color='#55A868', lw=2.5, label=f'v={V} m/s')

for th, lab, c in zip(THRESHOLDS, LABELS, COLORS):
    ax.axhline(th, color=c, ls='--', lw=1.8, alpha=.8)

    # 交点
    idx = np.argmax(delta_alpha > th)
    ax.plot(DT[idx], th, 'o', ms=10, color=c, mec='white', mew=1.5)
    ax.text(DT[idx], th-0.008, f'{DT[idx]} ms', color=c, weight='bold',
            bbox=dict(fc='white', ec='none', pad=2))

    # 左侧标签
    ax.text(5, th+0.002, lab, color=c, weight='bold',
            bbox=dict(fc='white', ec='none', pad=2))

# 全局样式
ax.set(xlabel=r'$\Delta t$ (ms)', ylabel=r'$\Delta \alpha$ (rad)',
       xlim=(0, 315), ylim=(0, delta_alpha.max()*1.15))
ax.grid(ls='--', alpha=.3)
ax.legend(loc='lower right')
fig.tight_layout()
plt.show()

"""
v-Δt 伪彩色图，并叠加三条阈值曲线
"""
from matplotlib.ticker import AutoMinorLocator

# ---------- 参数 ----------
D = 200                              # 距离(m)
V_MAX, T_MAX = 30, 300               # 速度、时间范围

THRESHOLDS = 1 / 2 ** np.array([4, 5, 6])  # DELTA_ALPHA的临界值
LABELS = [r'21$\times$21(s=3)', r'32$\times$32(s=4)', r'64$\times$64(s=5)']
#COLORS = ['white', '#6B5B95', 'black']
COLORS = ['white', 'white', 'black']

# ---------- 数据网格 ----------
v = np.linspace(0, V_MAX, 11)
t = np.linspace(0, T_MAX, 30)
T, V_GRID = np.meshgrid(t, v)
DELTA_ALPHA = np.arctan(V_GRID * (T/1000) /  D) * 1000

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(5, 5))

im = ax.pcolormesh(T, V_GRID, DELTA_ALPHA,
                   shading='auto',
                   cmap='YlGnBu')
fig.colorbar(im, label=r'$\Delta \alpha$ (mrad)')

for th, lab, c in zip(THRESHOLDS, LABELS, COLORS):
    v_line = 1000 * D * np.tan(th) / t[1:]     # 忽略 t=0
    if th==0.015625 or th==0.03125:
        ax.plot(t[1:], v_line, '--', color=c, lw=1.5, label=lab)
        ax.text(180, v_line[-1]-2, lab, color=c, fontsize=15)

# 全局样式
ax.set(xlabel=r'$\Delta t$ (ms)',
       ylabel=r'Maximum $v_\perp$ (m/s)',
       xlim=(0, T_MAX), ylim=(0, V_MAX))
ax.grid(ls=':', color='gray', lw=.5)
ax.minorticks_on()
ax.grid(which='minor', ls=':', color='gray', lw=.5)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.tight_layout()
fig.savefig('sim1_Ts_200.png', dpi=600, bbox_inches='tight')
plt.show()