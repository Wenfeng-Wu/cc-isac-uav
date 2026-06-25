import numpy as np
import torch
import os
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from numpy import mean

try:
    from .functions import plot_topk_line, plot_topk_bar, is_target_in_beam, is_target_in_4neighbor_beam, \
        is_target_in_8neighbor_beam, is_target_in_25neighbor_beam
except ImportError:
    from functions import plot_topk_line, plot_topk_bar, is_target_in_beam, is_target_in_4neighbor_beam, \
        is_target_in_8neighbor_beam, is_target_in_25neighbor_beam
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_and_unpack(file_name):
    """一次性加载并拆包 checkpoint"""
    load_path = os.path.join(parent_dir, 'data_P6', file_name)
    print(load_path)
    ckpt = torch.load(load_path, weights_only=False)
    #model_path = ckpt['model']['model_path']
    #print("data_P6 from", model_path)
    return ckpt['results']


def format_scientific(v):
    return f"{v:.2e}" if v < 1e-3 or v > 1e4 else f"{v:.4f}"

def mse(a, b):
    return sum(((x-y)*0.0174533)**2 for x, y in zip(a, b)) / len(a)

#======================= 画图函数 ========================================================================================

def plot_cdf_focused(ax, data_list, colors, lines, labels, markers, xlabel, title=None, rmse_text=None,
                     focus_percentile=95, x_padding=0.05):
    # 计算所有数据集中最小的95%分位数
    min_p95 = min(np.percentile(data, focus_percentile) for data in data_list)

    # 设置x轴范围：0到最小95%分位数 + 留白
    xmax = min_p95 * (1 + x_padding)
    ax.set_xlim(0, xmax)

    # 保持y轴完整显示
    ax.set_ylim(0, 1.0)

    # 绘制CDF曲线
    for data, color, label, line, marker in zip(data_list, colors, labels, lines, markers):
        # 计算CDF
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # 绘制CDF曲线
        sns.lineplot(x=sorted_data, y=cdf,  linestyle=line, color=color, ax=ax, linewidth=2, legend=False, marker=marker, markevery=400)

    # 设置图表属性
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('CPF', fontsize=16)
    #ax.legend(fontsize=16)
    # 同时设置X轴和Y轴刻度标签字体大小
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)

    if title:
        ax.set_title(f"{title} (Focused on {focus_percentile}% Point)")

    return min_p95  # 返回使用的x轴上限值


def plot_histogram_line(ax, data_list, xmax, colors, linstyle, labels,  xlabel, title=None, rmse_text=None, bins=30):
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
    #ax.set_xlim(0, xmax)
    if title:
        ax.set_title(title)

    if rmse_text:
        ax.text(0.95, 0.95, rmse_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))


def print_topk(az_pred, el_pred, az_true, el_true):
    import matplotlib.pyplot as plt
    data_S_topk = []
    data_Times = []
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


        data_S_topk.append([top0/(i+1), top1/(i+1), top2/(i+1), top3/(i+1)])
        print("top-k rate", top0/(i+1), top1/(i+1), top2/(i+1), top3/(i+1), top6/(i+1))
        #print("分层平均搜索次数：", 2.5 * S)
        eep = top0/(i+1) + top1/(i+1)*3.5 + top2/(i+1)*7.5 + top3/(i+1)*17.5 + top6/(i+1)* 2.5*S
        data_Times.append(eep)
        print("视觉辅助时平均搜索次数(总是逐层往上找)：", eep)
    return data_S_topk, data_Times


if __name__ == "__main__":
    # =======================加载数据=========================================================================================
    # 获取当前脚本文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    r0 = load_and_unpack('Comparison_MMFE_pred_a_snr0.pth')
    az_true = r0['az_true']
    az_pred1 = r0['az_pred']
    az_vision = r0['az_vision']
    az_echoOnly = r0['az_echo']
    az_diff_pred1_true = [abs(x) for x in r0['az_diff_pred_true']]
    az_diff_vision_true = [abs(x) for x in r0['az_diff_vision_true']]
    az_diff_echoOnly_true = [abs(x) for x in r0['az_diff_echo_true']]

    r0 = load_and_unpack('Comparison_KL_pred_ab_snr0_1.pth')
    az_diff_klpred_true = r0['az_diff_pred_true']
    az_klpred = r0['az_pred']
    el_diff_klpred_true = r0['el_diff_pred_true']
    el_klpred = r0['el_pred']

    r0 = load_and_unpack('Comparison_MMFE_noFuse_pred_a_snr0_1.pth')
    az_diff_noFusepred_true = r0['az_diff_pred_true']
    az_noFusepred = r0['az_pred']

    r0 = load_and_unpack('Comparison_MMFE_noFuse_pred_b_snr0_1.pth')
    el_diff_noFusepred_true = r0['el_diff_pred_true']
    el_noFusepred = r0['el_pred']

    r0 = load_and_unpack('Comparison_MMFE_pred_b_snr0.pth')
    el_true = r0['el_true']
    el_pred1 = r0['el_pred']
    el_vision = r0['el_vision']
    el_echoOnly = r0['el_echo']
    el_diff_pred1_true = [abs(x) for x in r0['el_diff_pred_true']]
    el_diff_vision_true = [abs(x) for x in r0['el_diff_vision_true']]
    el_diff_echoOnly_true = [abs(x) for x in r0['el_diff_echo_true']]

    # =======================计算 MSE =========================================================================================

    # 计算 MSE
    az_mse_pred1_true = mse(az_pred1, az_true)
    el_mse_pred1_true = mse(el_pred1, el_true)
    az_mse_vision_true = mse(az_vision, az_true)
    el_mse_vision_true = mse(el_vision, el_true)
    az_mse_echoOnly_true = mse(az_echoOnly, az_true)
    el_mse_echoOnly_true = mse(el_echoOnly, el_true)

    az_mse_klpred_true = mse(az_klpred, az_true)
    el_mse_klpred_true = mse(el_klpred, el_true)
    az_mse_noFusepred_true = mse(az_noFusepred, az_true)
    el_mse_noFusepred_true = mse(el_noFusepred, el_true)

    print("\n".join([
        f"MMFE  Azimuth MSE: {format_scientific(az_mse_pred1_true)}",
        f"MMFE  Elevation MSE: {format_scientific(el_mse_pred1_true)}",
        f"V2EDA Azimuth MSE: {format_scientific(az_mse_vision_true)}",
        f"V2EDA Elevation MSE: {format_scientific(el_mse_vision_true)}",
        f"Echo-Only Azimuth MSE: {format_scientific(az_mse_echoOnly_true)}",
        f"Echo-Only Elevation MSE: {format_scientific(el_mse_echoOnly_true)}",
        f"MMFE wo Fuse Azimuth MSE:{format_scientific(az_mse_noFusepred_true)}",
        f"MMFE wo Fuse Elevation MSE:{format_scientific(el_mse_noFusepred_true)}",
        f"KL Azimuth MSE: {format_scientific(az_mse_klpred_true)}",
        f"KL Elevation MSE: {format_scientific(el_mse_klpred_true)}"

    ]))

    # ================================= 主绘图逻辑 ============================================================================

    # 定义颜色和标签方案
    plot_config = {
        'colors': ['#FFA500', '#6B8E23', '#008080', '#FFB6C1', '#DB7093'],
        'labels': ['MMFE', 'Echo-Only', 'Vision-Only', 'KF-Based', 'MMFE w/o Fuse'],
        'line_styles': ['-', '--', '-.', ':', '--'],
        'markers': ['None', 'None', 'None', 's', 'o']
    }

    az_data = [az_diff_pred1_true, az_diff_echoOnly_true, az_diff_vision_true, az_diff_klpred_true, az_diff_noFusepred_true]
    az_data = [[abs(x * 0.017453293) for x in sublist] for sublist in az_data]
    el_data = [el_diff_pred1_true, el_diff_echoOnly_true, el_diff_vision_true, el_diff_klpred_true, el_diff_noFusepred_true]
    el_data = [[abs(x * 0.017453293) for x in sublist] for sublist in el_data]

    #print("mse : ", mean(az_diff_pred1_true))
    #print("mse : ", mean(el_diff_pred1_true))
    #print("mse : ", mean(az_diff_echoOnly_true))
    #print("mse : ", mean(el_diff_echoOnly_true))
    # =============分开的箱线图

    # 布局参数
    inner_gap = 0.3  # 组内间隔
    box_width = 0.3
    start_pos = 0.5  # 起始位置

    # 分别计算方位角和仰角的箱线图位置与数据
    # 方位角(Azimuth)数据处理
    az_cols = 5
    az_positions = [start_pos + inner_gap * i for i in range(az_cols)]
    az_colors = plot_config['colors'][:az_cols]

    # 仰角(Elevation)数据处理
    el_cols = 5
    el_positions = [start_pos + inner_gap * i for i in range(el_cols)]
    el_colors = plot_config['colors'][:el_cols]

    # 均值点样式设置
    meanprops = dict(marker='o',
                     markerfacecolor='white',
                     markeredgecolor='black',
                     markersize=4)

    # --------------------------
    # 绘制方位角(Azimuth)箱线图
    # --------------------------
    fig_az, ax_az = plt.subplots(figsize=(3, 5))
    bplot_az = ax_az.boxplot(az_data,
                             positions=az_positions,
                             widths=box_width,
                             patch_artist=True,
                             showfliers=False,  # 不显示异常值
                             showmeans=True,  # 显示均值
                             medianprops={'color': 'white', 'linewidth': 1},
                             meanprops=meanprops)

    # 为方位角箱线图着色
    for patch, color in zip(bplot_az['boxes'], az_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 方位角坐标轴设置
    ax_az.tick_params(labelsize=18)
    ax_az.set_xticks([])
    # ax_az.set_xticklabels(plot_config['labels'], fontsize=20)  # 使用组标签
    ax_az.set_xlabel('Azimuth', fontsize=20)  # 方位角标题
    ax_az.set_ylabel('Angle Error (rad)', color='black', fontsize=20)

    # 方位角图例
    legend_elements = [Line2D([0], [0],
                              color=c, label=l, linestyle=ls, alpha=0.7)
                       for c, l, ls in zip(plot_config['colors'][:5],
                                           plot_config['labels'],
                                           plot_config['line_styles'])]
    # ax_az.legend(handles=legend_elements, loc='upper left', fontsize=18)

    # 保存方位角图像
    plt.tight_layout()
    save_az = os.path.join(current_dir, 'paper-fig8_2.png')
    plt.savefig(save_az, dpi=900, bbox_inches='tight')
    plt.show()

    # --------------------------
    # 绘制仰角(Elevation)箱线图
    # --------------------------
    fig_el, ax_el = plt.subplots(figsize=(3, 5))
    bplot_el = ax_el.boxplot(el_data,
                             positions=el_positions,
                             widths=box_width,
                             patch_artist=True,
                             showfliers=False,  # 不显示异常值
                             showmeans=True,  # 显示均值
                             medianprops={'color': 'white', 'linewidth': 1},
                             meanprops=meanprops)

    # 为仰角箱线图着色
    for patch, color in zip(bplot_el['boxes'], el_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 仰角坐标轴设置
    ax_el.tick_params(labelsize=18)
    ax_el.set_xticks([])
    # ax_el.set_xticklabels(plot_config['labels'], fontsize=20)  # 使用组标签
    ax_el.set_xlabel('Elevation', fontsize=20)  # 仰角标题
    ax_el.set_ylabel('Angle Error (rad)', color='black', fontsize=20)

    # 仰角图例（与方位角保持一致）
    # ax_el.legend(handles=legend_elements, loc='upper left', fontsize=18)

    # 保存仰角图像
    plt.tight_layout()
    save_el = os.path.join(current_dir, 'paper-fig8_3.png')
    plt.savefig(save_el, dpi=900, bbox_inches='tight')
    plt.show()

    # ============================== CDF =====================================
    fig, axes = plt.subplots(1, 1, figsize=(4, 5))
    plot_cdf_focused(
        axes,
        az_data,
        plot_config['colors'],
        plot_config['line_styles'],
        plot_config['labels'],
        plot_config['markers'],
        'Azimuth Error (rad)',
        # rmse_text=az_rmse
    )

    # 调整布局
    plt.subplots_adjust(left=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.legend(handles=legend_elements, loc='lower right', fontsize=18)
    axes.legend(handles=legend_elements, loc='lower right', fontsize=12, bbox_to_anchor=(1, 0), frameon=True)
    # 保存图像
    plt.subplots_adjust(
        # left=0.4,      # 左侧边距
        # right=0.9,     # 右侧边距
        # bottom=0.1,    # 底部边距
        # top=0.9,       # 顶部边距
        # hspace=0.4,    # 子图间垂直间距
        wspace=0.3  # 子图间水平间距（对多列子图有用）
    )
    save_path = os.path.join(current_dir, 'paper-fig8_4.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"综合分析图已保存至: {save_path}")
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(4, 5))
    plot_cdf_focused(
        axes,
        el_data,
        plot_config['colors'],
        plot_config['line_styles'],
        plot_config['labels'],
        plot_config['markers'],
        'Elevation Error (rad)',
        # rmse_text=el_rmse
    )

    # 调整布局
    plt.subplots_adjust(left=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.legend(handles=legend_elements, loc='lower right', fontsize=18)
    axes.legend(handles=legend_elements, loc='lower right', fontsize=12, bbox_to_anchor=(1, 0), frameon=True)
    # 保存图像
    plt.subplots_adjust(
        # left=0.4,      # 左侧边距
        # right=0.9,     # 右侧边距
        # bottom=0.1,    # 底部边距
        # top=0.9,       # 顶部边距
        # hspace=0.4,    # 子图间垂直间距
        wspace=0.3  # 子图间水平间距（对多列子图有用）
    )
    save_path = os.path.join(current_dir, 'paper-fig8_5.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"综合分析图已保存至: {save_path}")
    plt.show()
    #plot_topk_line(az_vision, el_vision, az_true, el_true, 'fig10.png')
    #plot_topk_line(az_pred1, el_pred1, az_true, el_true, 'fig101.png')
    #plot_topk_line(az_echoOnly, el_echoOnly, az_true, el_true, 'fig102.png')
    '''
    vision_topk = (plot_topk(az_vision, el_vision, az_true, el_true))
    echo_topk = (plot_topk(az_echoOnly, el_echoOnly, az_true, el_true))
    fused_topk = (plot_topk(az_pred1, el_pred1, az_true, el_true))
    plot_topk_line(az_vision, el_vision, az_true, el_true, 'fig5_1.png')
    plot_topk_line(az_echoOnly, el_echoOnly, az_true, el_true, 'fig5_2.png')
    plot_topk_line(az_pred1, el_pred1, az_true, el_true, 'fig5_3.png')

    for k in range(4):
        #data_S_topk = [vision_topk[k], echo_topk[k],fused_topk[k]]
        data_S_topk = [echo_topk[k],fused_topk[k]]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=900)

        colors = ['#FFA500', '#6B8E23', '#008080']  # Tableau调色板
        markers = ['o', 's', '^', 'D', '+']  # 不同的标记样式
        line_styles = ['-', '-.', '--', ':', (0, (2, 2))]  # 不同的线型
        #labels = ['Vision-Only', 'Echo-Only', 'Multi-Modal']
        labels = ['Echo-Only', 'Multi-Modal']

        for i, series in enumerate(data_S_topk):
            if i == 5:
                break
            x = np.arange(1, len(series) + 1)  # X轴值 (1, 2, 3, ...)

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
        # 设置轴标签和标题
        ax.set_xlabel(r'$\mathcal{I}^{(i)}$', fontsize=16)
        ax.set_ylabel('Accuracy (%)', fontsize=18)

        # 设置横坐标刻度 - 从1开始到数据长度
        ax.set_xticks(range(1, len(data_S_topk[0]) + 1))

        # 设置轴范围
        ax.set_xlim(0.5, len(data_S_topk[0]) + 0.5)
        ax.set_ylim(0, 1)

        # 设置刻度（移除或注释掉这行，因为已经用set_xticks设置了）
        # ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', labelsize=16)
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6, which='both')
        # 添加图例
        legend = ax.legend(loc='upper right', fontsize=14, frameon=True)
        # legend.get_frame().set_facecolor('#f5f5f5')
        # legend.get_frame().set_edgecolor('#cccccc')
        legend.get_frame().set_linewidth(0.8)
        # 调整布局
        plt.tight_layout(pad=1.2)
        # 保存高质量图片 (用于论文)
        plt.savefig('fig4_{}.png'.format(k), dpi=900, bbox_inches='tight')
        # 显示图表
        plt.show()

    print("done")
    '''
