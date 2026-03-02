import math

import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def calculate_codebook_level(N):
    """
    计算 N×N 天线阵列的码本推荐等级 S

    参数:
        N (int): 天线阵列一维尺寸 (例如 21 表示 21x21 阵列)

    返回:
        tuple: (S_recommended, min_beamwidth_deg)
               S_recommended - 推荐的码本等级
               min_beamwidth_deg - 阵列理论最小波束宽度(度)
    """
    # = 1. 计算阵列理论最小波束宽度 =
    # 假设天线间距为半波长 (λ/2)
    min_beamwidth_rad = 1.772 / N  # 法线方向最小波束宽度(弧度)
    min_beamwidth_deg = min_beamwidth_rad * (180 / math.pi)  # 转换为度

    # = 2. 码本搜索范围设置 =
    # 假设覆盖半球 (0-180度物理角度0-90度物理角度)
    # 对应的ψ域范围宽度为 2π 和 π 弧度
    Delta_Psi = math.pi  # ψ_v域总搜索范围(弧度)

    # = 3. 计算最大可行等级 S_max =
    # 物理约束: 码本最小波束宽度 >= 阵列理论最小波束宽度
    S_max = math.floor(math.log2(1 / min_beamwidth_rad))

    # = 4. 推荐等级策略 =
    # 考虑混合波束赋形硬件限制，推荐比理论值保守1级(仿真不考虑降一级)
    if S_max > 3:
        S_recommended = S_max
    else:
        S_recommended = max(S_max, 1)  # 至少1级

    # = 5. 显示结果 =
    # 如果需要显示结果，可以取消以下注释
    """
    print(f'===== {N}×{N} 天线阵列码本等级计算 =====')
    print(f'理论最小波束宽度: {min_beamwidth_deg:.2f}° (法线方向)')
    print(f'理论最大可行等级 S_max: {S_max}')
    print(f'推荐实用等级 S: {S_recommended}')
    print('-----------------------------------')
    print('该等级下波束宽度参考:')
    print(f' - 法线方向: {57.3 / (2**S_recommended):.2f}°')

    # 计算典型方向的余弦值
    cos60 = math.cos(math.radians(60))
    cos30 = math.cos(math.radians(30))

    # 计算水平方向波束宽度
    horizontal_bw = 180 / (math.pi * (2**S_recommended) * cos60 * cos30)
    # 计算垂直方向波束宽度
    vertical_bw = 180 / (math.pi * (2**S_recommended) * cos30)

    print(' - 典型方向(θ=60°,φ=30°):')
    print(f'     水平: {horizontal_bw:.2f}°')
    print(f'     垂直: {vertical_bw:.2f}°')
    """

    return S_recommended, min_beamwidth_deg

def is_target_in_beam(S, theta_p, phi_p, theta_t, phi_t, in_degrees=True):
    """
    判断目标点 (theta_t, phi_t) 是否在指定点 (theta_p, phi_p) 所在的最小波束内

    参数:
        S (int): 码本等级
        theta_p (float): 参考点水平角度
        phi_p (float): 参考点垂直角度
        theta_t (float): 目标点水平角度
        phi_t (float): 目标点垂直角度
        in_degrees (bool): 角度输入是否为度数（默认为True）

    返回:
        bool: True表示目标点在参考点所在波束内，False表示不在
    """
    # 如果输入是度数，转换为弧度
    if in_degrees:
        theta_p = math.radians(theta_p)
        phi_p = math.radians(phi_p)
        theta_t = math.radians(theta_t)
        phi_t = math.radians(phi_t)

    # 1. 计算ψ域坐标
    # 参考点
    psi_h_p = math.pi * math.sin(theta_p) * math.cos(phi_p)
    psi_v_p = math.pi * math.sin(phi_p)

    # 目标点
    psi_h_t = math.pi * math.sin(theta_t) * math.cos(phi_t)
    psi_v_t = math.pi * math.sin(phi_t)

    # 2. 计算最高等级波束宽度（ψ域）
    # 每个维度被划分为 2^S 个区间
    num_bins = 2 ** S
    psi_bin_width = math.pi / num_bins

    # 3. 计算参考点所在的波束区间索引
    # 水平索引
    bin_h_p = min(int((psi_h_p) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_p = min(int((psi_v_p) // psi_bin_width), num_bins - 1)

    # 4. 计算目标点所在的波束区间索引
    # 水平索引
    bin_h_t = min(int((psi_h_t) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_t = min(int((psi_v_t) // psi_bin_width), num_bins - 1)

    # 5. 判断目标点是否在参考点的波束内
    return (bin_h_p == bin_h_t) and (bin_v_p == bin_v_t)

def is_target_in_4neighbor_beam(S, theta_p, phi_p, theta_t, phi_t, in_degrees=True):
    """
    判断目标点 (theta_t, phi_t) 是否在指定点 (theta_p, phi_p) 所在的最小波束内

    参数:
        S (int): 码本等级
        theta_p (float): 参考点水平角度
        phi_p (float): 参考点垂直角度
        theta_t (float): 目标点水平角度
        phi_t (float): 目标点垂直角度
        in_degrees (bool): 角度输入是否为度数（默认为True）

    返回:
        bool: True表示目标点在参考点所在波束内，False表示不在
    """
    # 如果输入是度数，转换为弧度
    if in_degrees:
        theta_p = math.radians(theta_p)
        phi_p = math.radians(phi_p)
        theta_t = math.radians(theta_t)
        phi_t = math.radians(phi_t)

    # 1. 计算ψ域坐标
    # 参考点
    psi_h_p = math.pi * math.sin(theta_p) * math.cos(phi_p)
    psi_v_p = math.pi * math.sin(phi_p)

    # 目标点
    psi_h_t = math.pi * math.sin(theta_t) * math.cos(phi_t)
    psi_v_t = math.pi * math.sin(phi_t)

    # 2. 计算最高等级波束宽度（ψ域）
    # 每个维度被划分为 2^S 个区间
    num_bins = 2 ** S
    psi_bin_width = math.pi / num_bins

    # 3. 计算参考点所在的波束区间索引
    # 水平索引
    bin_h_p = min(int((psi_h_p) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_p = min(int((psi_v_p) // psi_bin_width), num_bins - 1)

    bin_p_4 = [[bin_h_p-1,bin_v_p],
                 [bin_h_p+1,bin_v_p],
                 [bin_h_p,bin_v_p+1],
                 [bin_h_p,bin_v_p-1]]

    # 4. 计算目标点所在的波束区间索引
    # 水平索引
    bin_h_t = min(int((psi_h_t) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_t = min(int((psi_v_t) // psi_bin_width), num_bins - 1)

    bin_t = [bin_h_t,bin_v_t]
    # 5. 判断目标点是否在参考点的波束内
    return bin_t in bin_p_4

def is_target_in_8neighbor_beam(S, theta_p, phi_p, theta_t, phi_t, in_degrees=True):
    """
    判断目标点 (theta_t, phi_t) 是否在指定点 (theta_p, phi_p) 所在的最小波束内

    参数:
        S (int): 码本等级
        theta_p (float): 参考点水平角度
        phi_p (float): 参考点垂直角度
        theta_t (float): 目标点水平角度
        phi_t (float): 目标点垂直角度
        in_degrees (bool): 角度输入是否为度数（默认为True）

    返回:
        bool: True表示目标点在参考点所在波束内，False表示不在
    """
    # 如果输入是度数，转换为弧度
    if in_degrees:
        theta_p = math.radians(theta_p)
        phi_p = math.radians(phi_p)
        theta_t = math.radians(theta_t)
        phi_t = math.radians(phi_t)

    # 1. 计算ψ域坐标
    # 参考点
    psi_h_p = math.pi * math.sin(theta_p) * math.cos(phi_p)
    psi_v_p = math.pi * math.sin(phi_p)

    # 目标点
    psi_h_t = math.pi * math.sin(theta_t) * math.cos(phi_t)
    psi_v_t = math.pi * math.sin(phi_t)

    # 2. 计算最高等级波束宽度（ψ域）
    # 每个维度被划分为 2^S 个区间
    num_bins = 2 ** S
    psi_bin_width = math.pi / num_bins

    # 3. 计算参考点所在的波束区间索引
    # 水平索引
    bin_h_p = min(int((psi_h_p) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_p = min(int((psi_v_p) // psi_bin_width), num_bins - 1)

    bin_p_8 = [[bin_h_p-1,bin_v_p],
               [bin_h_p-1,bin_v_p-1],
               [bin_h_p-1,bin_v_p+1],
               [bin_h_p + 1, bin_v_p],
               [bin_h_p + 1, bin_v_p-1],
               [bin_h_p + 1, bin_v_p+1],
               [bin_h_p,bin_v_p+1],
                [bin_h_p,bin_v_p-1]]

    # 4. 计算目标点所在的波束区间索引
    # 水平索引
    bin_h_t = min(int((psi_h_t) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_t = min(int((psi_v_t) // psi_bin_width), num_bins - 1)

    bin_t = [bin_h_t,bin_v_t]

    # 5. 判断目标点是否在参考点的波束内
    return bin_t in bin_p_8


def is_target_in_25neighbor_beam(S, theta_p, phi_p, theta_t, phi_t, in_degrees=True):
    """
    判断目标点 (theta_t, phi_t) 是否在指定点 (theta_p, phi_p) 所在的最小波束内

    参数:
        S (int): 码本等级
        theta_p (float): 参考点水平角度
        phi_p (float): 参考点垂直角度
        theta_t (float): 目标点水平角度
        phi_t (float): 目标点垂直角度
        in_degrees (bool): 角度输入是否为度数（默认为True）

    返回:
        bool: True表示目标点在参考点所在波束内，False表示不在
    """
    # 如果输入是度数，转换为弧度
    if in_degrees:
        theta_p = math.radians(theta_p)
        phi_p = math.radians(phi_p)
        theta_t = math.radians(theta_t)
        phi_t = math.radians(phi_t)

    # 1. 计算ψ域坐标
    # 参考点
    psi_h_p = math.pi * math.sin(theta_p) * math.cos(phi_p)
    psi_v_p = math.pi * math.sin(phi_p)

    # 目标点
    psi_h_t = math.pi * math.sin(theta_t) * math.cos(phi_t)
    psi_v_t = math.pi * math.sin(phi_t)

    # 2. 计算最高等级波束宽度（ψ域）
    # 每个维度被划分为 2^S 个区间
    num_bins = 2 ** S
    psi_bin_width = math.pi / num_bins

    # 3. 计算参考点所在的波束区间索引
    # 水平索引
    bin_h_p = min(int((psi_h_p) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_p = min(int((psi_v_p) // psi_bin_width), num_bins - 1)

    bin_p_25 = [[bin_h_p-2,bin_v_p-2],
               [bin_h_p-2,bin_v_p-1],
               [bin_h_p-2,bin_v_p],
               [bin_h_p-2,bin_v_p+1],
               [bin_h_p-2,bin_v_p+2],
               [bin_h_p-1,bin_v_p-2],
               [bin_h_p-1,bin_v_p+2],
               [bin_h_p,bin_v_p-2],
               [bin_h_p,bin_v_p+2],
               [bin_h_p + 1, bin_v_p-2],
               [bin_h_p + 1, bin_v_p+2],
               [bin_h_p + 2, bin_v_p-2],
               [bin_h_p + 2, bin_v_p-1],
               [bin_h_p + 2, bin_v_p],
               [bin_h_p + 2, bin_v_p+1],
               [bin_h_p + 2, bin_v_p+2]]

    # 4. 计算目标点所在的波束区间索引
    # 水平索引
    bin_h_t = min(int((psi_h_t) // psi_bin_width), num_bins - 1)
    # 垂直索引
    bin_v_t = min(int((psi_v_t) // psi_bin_width), num_bins - 1)

    bin_t = [bin_h_t,bin_v_t]

    # 5. 判断目标点是否在参考点的波束内
    return bin_t in bin_p_25



def plot_histogram_line(ax, data_list, colors, linstyle, labels,  xlabel, title=None, rmse_text=None, bins=30):
    """
    绘制多组数据的直方图折线表示（非平滑的阶梯状曲线）

    参数:
    ax: matplotlib轴对象
    data_list: 数据列表，每个元素是一个数据数组
    colors: 每组数据对应的颜色列表
    labels: 每组数据对应的标签列表
    xlabel: x轴标签
    title: 图表标题(可选)
    rmse_text: 在图表上显示的文本(可选)
    bins: 分箱数量(默认30)
    """
    for data, color, lins, label in zip(data_list, colors, linstyle, labels):
        # 计算直方图数据（但不绘制）
        counts, bin_edges = np.histogram(data, bins=bins)

        # 计算每个bin的中心位置
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 绘制折线图（连接bin中心点）
        ax.plot(bin_centers, counts,
                color=color,
                label=label,
                linewidth=2,
                #marker='o',  # 添加数据点标记
                markersize=4,
                linestyle=lins)  # 实线连接

    # 设置图表属性
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    if title:
        ax.set_title(title)

    if rmse_text:
        ax.text(0.95, 0.95, rmse_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))


def plot_topk_line(az_pred, el_pred, az_true, el_true, figname):
    import matplotlib.pyplot as plt
    data_S_topk = []
    find_num_proposed = []
    find_num_proposed = []
    find_num_proposed = []
    for S in range(2,7):
        print("=====当前测试的码本最高等级:", S,"=====")

        top0, top1, top2, top3, top4, top5, top6= 0, 0, 0, 0, 0, 0, 0
        ##=========================== 统计top-k ========================##
        for i in range(len(az_true)):
            #print(az_true[i], az_pred[i], el_true[i], el_pred[i])

            if S>=0 and is_target_in_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top0 += 1
            elif is_target_in_4neighbor_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top1 += 1
            elif is_target_in_8neighbor_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top2 += 1
            elif is_target_in_25neighbor_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top3 += 1
            else:
                top6 += 1


        data_S_topk.append([top0/(i+1), top1/(i+1), top2/(i+1), top3/(i+1), top6/(i+1)])
        #data_S_topk.append([top0/(i+1), top0/(i+1)+top1/(i+1), top0/(i+1)+top1/(i+1)+top2/(i+1), top0/(i+1)+top1/(i+1)+top2/(i+1)+top3/(i+1)])
        #print("top-k rate", top0/(i+1), top1/(i+1), top2/(i+1), top6/(i+1))
        #print("分层平均搜索次数：", 2.5 * S)
        #print("视觉辅助时平均搜索次数(总是逐层往上找)：", top0/(i+1) + top1/(i+1)*3.5 + top2/(i+1)*7.5 + top6/(i+1)* (8+2.5*S))

    fig, ax = plt.subplots(figsize=(3, 4), dpi=900)

    colors = ['#FFA500', '#6B8E23', '#008080', '#6495ED', '#DB7093']  # Tableau调色板
    markers = ['o', 's', '^', 'D', '+']  # 不同的标记样式
    line_styles = ['-', '-.', '--', ':', (0,(2, 2))]  # 不同的线型
    labels = [r'$8\times 8$ (s=2)', r'$16\times 16$ (s=3)', r'$32\times 32$ (s=4)',
              r'$64\times 64$ (s=5)', r'$128\times 128$ (s=6)']

    for i, series in enumerate(data_S_topk):
        if i == 5:
            break
        x = np.arange(0, len(series))  # X轴值 (1, 2, 3, ...)

        # 绘制折线
        ax.plot(x, series,
                color=colors[i],
                linestyle=line_styles[i],
                marker=markers[i],
                markersize=9,
                linewidth=2.5,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=labels[i])

        # 添加数据标签 (可选)
        for j, val in enumerate(series):
            if j < 0:
                ax.annotate(f'{val:.2f}%',
                            xy=(x[j], val),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            fontsize=10,
                            color=colors[i])

    # 设置轴标签和标题
    ax.set_xlabel(r'$\mathcal{I}^{(i)}$', fontsize=16)  # , fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=18)  # ,, fontweight='bold')
    # ax.set_title('Performance Comparison of Different s',
    #             fontsize=16, fontweight='bold', pad=20)
    # 让 0,1,2,3,4 显示成 A,B,C,D,E
    custom_labels = ['1', '2', '3', '4']
    ax.set_xticks(range(4))  # 先把刻度位置定好
    ax.set_xticklabels(custom_labels)  # 再换成想要的文字
    # 设置轴范围
    #ax.set_xlim(0.5, len(data_S_topk[0]) + 0.5)
    ax.set_ylim(0, 1)
    # 设置刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))  # 每个整数显示刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 每10个单位显示刻度
    ax.tick_params(axis='both', which='major', labelsize=16)
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    # 添加图例
    legend = ax.legend(loc='upper right', fontsize=10, frameon=True)
    # legend.get_frame().set_facecolor('#f5f5f5')
    # legend.get_frame().set_edgecolor('#cccccc')
    legend.get_frame().set_linewidth(0.8)
    # 调整布局
    plt.tight_layout(pad=1.2)
    # 保存高质量图片 (用于论文)
    plt.savefig(figname, dpi=900, bbox_inches='tight')
    # 显示图表
    plt.show()

def plot_topk_bar(az_pred, el_pred, az_true, el_true, figname):
    import matplotlib.pyplot as plt
    data_S_topk = []
    find_num_proposed = []
    find_num_proposed = []
    find_num_proposed = []
    for S in range(2,7):
        print("=====当前测试的码本最高等级:", S,"=====")

        top0, top1, top2, top3, top4, top5, top6= 0, 0, 0, 0, 0, 0, 0
        ##=========================== 统计top-k ========================##
        for i in range(len(az_true)):
            #print(az_true[i], az_pred[i], el_true[i], el_pred[i])

            if S>=0 and is_target_in_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top0 += 1
            elif is_target_in_4neighbor_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top1 += 1
            elif is_target_in_8neighbor_beam(S, az_pred[i], el_pred[i], az_true[i], el_true[i], in_degrees=True):
                top2 += 1
            else:
                top6 += 1


        data_S_topk.append([top0/(i+1), top1/(i+1), top2/(i+1), top6/(i+1)])
        print("top-k rate", top0/(i+1), top1/(i+1), top2/(i+1), top6/(i+1))
        print("分层平均搜索次数：", 2.5 * S)
        print("视觉辅助时平均搜索次数(总是逐层往上找)：", top0/(i+1) + top1/(i+1)*3.5 + top2/(i+1)*7.5 + top6/(i+1)* (8+2.5*S))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=900)

    colors = ['#FFA500', '#6B8E23', '#008080', '#6495ED', '#D8BFD8']  # Tableau调色板
    labels = [r'$8\times 8(s=2)$', r'$16\times 16(s=3)$', r'$32\times 32(s=4)$', r'$64\times 64(s=5)$',
              r'$128\times 128(s=6)$']

    # 设置柱状图的宽度和位置
    bar_width = 0.15
    x = np.arange(len(data_S_topk[0]))  # X轴位置 (0, 1, 2, ...)

    for i, series in enumerate(data_S_topk):
        if i == 5:
            break

        # 计算每个柱状图的位置偏移
        offset = (i - 2) * bar_width  # 居中排列

        # 绘制柱状图
        bars = ax.bar(x + offset, series,
                      width=bar_width,
                      color=colors[i],
                      alpha=1,
                      label=labels[i])

        # 添加数据标签 (可选)
        for j, val in enumerate(series):
            if j < 0:
                ax.annotate(f'{val:.2f}%',
                            xy=(x[j] + offset, val),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center',
                            fontsize=10,
                            color=colors[i])

    # 设置轴标签和标题
    ax.set_xlabel(r'$\mathcal{I}^{(i)}$', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)

    # 设置X轴刻度和标签
    custom_labels = ['1', '2', '3', '4']
    ax.set_xticks(range(4))
    ax.set_xticklabels(custom_labels)

    # 设置轴范围
    ax.set_xlim(-0.5, len(data_S_topk[0]) - 0.5)
    ax.set_ylim(0, 1)

    # 设置刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 每0.1个单位显示刻度
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6, which='both', axis='y')

    # 添加图例
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True)
    legend.get_frame().set_linewidth(0.8)

    # 调整布局
    plt.tight_layout(pad=2.0)

    # 保存高质量图片 (用于论文)
    plt.savefig(figname, dpi=600, bbox_inches='tight')

    # 显示图表
    plt.show()
