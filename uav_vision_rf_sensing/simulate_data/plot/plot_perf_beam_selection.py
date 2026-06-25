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

    plot_topk_line(az_vision, el_vision, az_true, el_true, 'paper_fig10a.png')
    #plot_topk_line(az_pred1, el_pred1, az_true, el_true, 'fig101.png')
    #plot_topk_line(az_echoOnly, el_echoOnly, az_true, el_true, 'fig102.png')

