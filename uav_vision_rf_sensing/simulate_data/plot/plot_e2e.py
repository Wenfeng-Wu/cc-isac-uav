import matplotlib.pyplot as plt

# ====================== 全局字体、样式设置（论文格式） ======================
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

SIZE_LABEL = 18
SIZE_TICK = 18
SIZE_LEGEND = 18
SIZE_TEXT = 20
LINEWIDTH = 1.5
MARKERSIZE = 12

colors = ['#FFA500', '#6B8E23', '#008080']

# ====================== 基础参数 ======================
t = 1/15
nb_ofdm_in_halfframe = 5*14
s_list = [2, 3, 4, 5, 6]

def cul_Tcomm(a):
    return (nb_ofdm_in_halfframe - a*4) * t

# ====================== 实验数据 ======================
hier_scan    = [5, 7.5, 10, 12.5, 15]
v2eda        = [1.2908, 1.4634, 2.0048, 3.4936, 7.2173]

echo_only_snr0 = [1.1308, 1.1668, 1.4478, 1.9738, 3.2682]
echo_only_snr_neg1 = [1.1611, 1.2032, 1.4557, 1.9814, 3.4110]
echo_only_snr_neg2 = [1.1890, 1.2339, 1.4821, 2.0501, 3.7365]
echo_only_few_lost = [1.1319, 1.1722, 1.4679, 1.9882, 3.3192]
echo_only_lot_lost = [1.1336, 1.1761, 1.4590, 2.0383, 3.3848]

kf_based_snr0   = [1.2418, 1.3325, 1.6288, 2.3809, 4.7797]
kf_based_snr_neg1 = [1.3266, 1.4267, 1.7496, 2.6232, 5.6807]
kf_based_snr_neg2 = [1.3682, 1.4904, 1.8447, 2.8739, 6.1463]

mmfe_snr0       = [1.1003, 1.1297, 1.3680, 1.8767, 2.9875]
mmfe_snr_neg1   = [1.1218, 1.1556, 1.3709, 1.8909, 3.1129]
mmfe_snr_neg2   = [1.1312, 1.1628, 1.3739, 1.9077, 3.3198]
mmfe_few_lost   = [1.1013, 1.1330, 1.3868, 1.8901, 3.0246]
mmfe_lot_lost   = [1.1144, 1.1460, 1.3914, 1.9520, 3.1761]

# ====================== 统一计算 Tcomm ======================
hier_T    = [cul_Tcomm(a) for a in hier_scan]
v2eda_T   = [cul_Tcomm(a) for a in v2eda]

echo0     = [cul_Tcomm(a) for a in echo_only_snr0]
kf0       = [cul_Tcomm(a) for a in kf_based_snr0]
mmfe0     = [cul_Tcomm(a) for a in mmfe_snr0]

echo_neg1 = [cul_Tcomm(a) for a in echo_only_snr_neg1]
kf_neg1   = [cul_Tcomm(a) for a in kf_based_snr_neg1]
mmfe_neg1 = [cul_Tcomm(a) for a in mmfe_snr_neg1]

echo_neg2 = [cul_Tcomm(a) for a in echo_only_snr_neg2]
kf_neg2   = [cul_Tcomm(a) for a in kf_based_snr_neg2]
mmfe_neg2 = [cul_Tcomm(a) for a in mmfe_snr_neg2]

echo_few  = [cul_Tcomm(a) for a in echo_only_few_lost]
mmfe_few  = [cul_Tcomm(a) for a in mmfe_few_lost]

echo_lot  = [cul_Tcomm(a) for a in echo_only_lot_lost]
mmfe_lot  = [cul_Tcomm(a) for a in mmfe_lot_lost]

# ====================== 统一 Y 轴范围 ======================
all_T = hier_T + v2eda_T + echo0 + kf0 + mmfe0 + echo_neg1 + kf_neg1 + mmfe_neg1 + echo_neg2 + kf_neg2 + mmfe_neg2 + echo_few + mmfe_few + echo_lot + mmfe_lot
ymin = 3
ymax = 4.5

# ====================== 1×6 子图 ======================
fig, axs = plt.subplots(1, 6, figsize=(20, 6))
plt.subplots_adjust(wspace=0.3)

# -------------------- (a) Scan --------------------
ax = axs[0]
ax.plot(s_list, hier_T, label='Hierarchical', color=colors[0], marker='>', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, v2eda_T, label='V2EDA', color=colors[1], marker='*', lw=LINEWIDTH, ms=MARKERSIZE)
ax.set_xlabel('s', fontsize=SIZE_LABEL)
ax.set_ylabel('$T_{Comm.}$ (ms)', fontsize=SIZE_LABEL)
ax.tick_params(labelsize=SIZE_TICK)
ax.legend(fontsize=SIZE_LEGEND)
ax.grid(alpha=0.3)
ax.set_xticks(s_list)
#ax.set_ylim(ymin, ymax)
ax.text(0.5, -0.21, '(a) Beam Steering', transform=ax.transAxes, ha='center', fontsize=SIZE_TEXT)

# -------------------- (b) SNR=0 --------------------
ax = axs[1]
ax.plot(s_list, echo0, label='Echo-Only', color=colors[0], marker='o', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, kf0, label='KF-Based', color=colors[1], marker='s', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, mmfe0, label='MMFE', color=colors[2], marker='^', lw=LINEWIDTH, ms=MARKERSIZE)
ax.set_xlabel('s', fontsize=SIZE_LABEL)
ax.tick_params(labelsize=SIZE_TICK)
ax.legend(fontsize=SIZE_LEGEND)
ax.grid(alpha=0.3)
ax.set_xticks(s_list)
ax.set_ylim(ymin, ymax)
ax.text(0.5, -0.21, '(b) SNR=0', transform=ax.transAxes, ha='center', fontsize=SIZE_TEXT)

# -------------------- (c) SNR=-1 --------------------
ax = axs[2]
ax.plot(s_list, echo_neg1, label='Echo-Only',color=colors[0], marker='o', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, kf_neg1, label='KF-Based', color=colors[1], marker='s', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, mmfe_neg1, label='MMFE', color=colors[2], marker='^', lw=LINEWIDTH, ms=MARKERSIZE)
ax.set_xlabel('s', fontsize=SIZE_LABEL)
ax.tick_params(labelsize=SIZE_TICK)
ax.legend(fontsize=SIZE_LEGEND)
ax.grid(alpha=0.3)
ax.set_xticks(s_list)
ax.set_ylim(ymin, ymax)
ax.text(0.5, -0.21, '(c) SNR=-1', transform=ax.transAxes, ha='center', fontsize=SIZE_TEXT)

# -------------------- (d) SNR=-2 --------------------
ax = axs[3]
ax.plot(s_list, echo_neg2, label='Echo-Only', color=colors[0], marker='o', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, kf_neg2, label='KF-Based', color=colors[1], marker='s', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, mmfe_neg2, label='MMFE', color=colors[2], marker='^', lw=LINEWIDTH, ms=MARKERSIZE)
ax.set_xlabel('s', fontsize=SIZE_LABEL)
ax.tick_params(labelsize=SIZE_TICK)
ax.legend(fontsize=SIZE_LEGEND)
ax.grid(alpha=0.3)
ax.set_xticks(s_list)
ax.set_ylim(ymin, ymax)
ax.text(0.5, -0.21, '(d) SNR=-2', transform=ax.transAxes, ha='center', fontsize=SIZE_TEXT)

# -------------------- (e) Few lost --------------------
ymin = 3.7
ymax = 4.45

ax = axs[4]
ax.plot(s_list, echo_few, label='Echo-Only', color=colors[0], marker='o', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, mmfe_few, label='MMFE',  color=colors[2], marker='^', lw=LINEWIDTH, ms=MARKERSIZE)
ax.set_xlabel('s', fontsize=SIZE_LABEL)
ax.tick_params(labelsize=SIZE_TICK)
ax.legend(fontsize=SIZE_LEGEND)
ax.grid(alpha=0.3)
ax.set_xticks(s_list)
ax.set_ylim(ymin, ymax)
ax.text(0.5, -0.21, '(e) Few lost', transform=ax.transAxes, ha='center', fontsize=SIZE_TEXT)

# -------------------- (f) Lot lost --------------------
ax = axs[5]
ax.plot(s_list, echo_lot, label='Echo-Only', color=colors[0], marker='o', lw=LINEWIDTH, ms=MARKERSIZE)
ax.plot(s_list, mmfe_lot, label='MMFE',  color=colors[2], marker='^', lw=LINEWIDTH, ms=MARKERSIZE)
ax.set_xlabel('s', fontsize=SIZE_LABEL)
ax.tick_params(labelsize=SIZE_TICK)
ax.legend(fontsize=SIZE_LEGEND)
ax.grid(alpha=0.3)
ax.set_xticks(s_list)
ax.set_ylim(ymin, ymax)
ax.text(0.5, -0.21, '(f) Lot lost', transform=ax.transAxes, ha='center', fontsize=SIZE_TEXT)

plt.tight_layout()
plt.show()

plt.savefig("fig11", dpi=900, bbox_inches='tight')