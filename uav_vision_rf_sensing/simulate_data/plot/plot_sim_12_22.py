import numpy as np
import pandas as pd
import torch
import os
# 画图设置
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def process_vision_data(load_path):
    """
    处理视觉对准网络的测试数据，计算并打印误差指标（MSE、NMSE、NMSE(dB)）

    参数:
        load_path: 数据文件（checkpoint）的路径
    """

    # 定义科学计数法格式化函数（嵌套在内部，避免全局污染）
    def format_scientific(value):
        """将值格式化为科学计数法或常规小数字符串"""
        return f"{value:.2e}" if value < 1e-3 or value > 1e4 else f"{value:.4f}"

    # 加载数据
    print("视觉对准网络的测试数据的估计值和实际值所在路径：", load_path)
    checkpoint = torch.load(load_path)
    model_data = checkpoint['model']
    results = checkpoint['results']
    model_path = model_data['model_path']
    print("视觉对准网络的模型参数路径：", model_path)

    # 提取并处理方位角数据（调整范围）
    az_tru = results['az_true']
    az_pre = results['az_pred']
    az_true = [a - 90 for a in az_tru]  # 调整为0-90范围
    az_pred = [a - 90 for a in az_pre]

    # 提取仰角数据（无需范围调整）
    el_true = results['el_true']
    el_pred = results['el_pred']

    # 处理误差数据（转换为弧度）
    az_diff = [a * 0.0174533 for a in results['az_diff']]  # 角度转弧度
    el_diff = [a * 0.0174533 for a in results['el_diff']]

    # 转换为numpy数组（高效计算）
    az_pred_np = np.array(az_pred)
    az_true_np = np.array(az_true)
    el_pred_np = np.array(el_pred)
    el_true_np = np.array(el_true)

    # 计算MSE（弧度²）
    az_mse_pred_true = np.mean(((az_pred_np - az_true_np) * 0.0174533) ** 2)
    el_mse_pred_true = np.mean(((el_pred_np - el_true_np) * 0.0174533) ** 2)

    # 计算真实值功率（弧度²，用于NMSE归一化）
    az_true_power = np.mean((az_true_np * 0.0174533) ** 2)
    el_true_power = np.mean((el_true_np * 0.0174533) ** 2)

    # 计算NMSE及dB值（添加极小值避免除零）
    epsilon = 1e-10
    az_nmse = az_mse_pred_true / (az_true_power + epsilon)
    el_nmse = el_mse_pred_true / (el_true_power + epsilon)
    az_nmse_db = 10 * np.log10(az_nmse)
    el_nmse_db = 10 * np.log10(el_nmse)

    # 格式化输出指标
    az_rmse_str = format_scientific(az_mse_pred_true)
    el_rmse_str = format_scientific(el_mse_pred_true)

    # 打印结果
    print(f"\n视觉单模态的估计 Azimuth MSE: {az_rmse_str}（弧度²）")
    print(f"视觉单模态的估计 Elevation MSE: {el_rmse_str}（弧度²）")

    print("\n" + "=" * 50)
    print("误差指标计算结果：")
    print("-" * 50)
    print(f"方位角：")
    print(f"  MSE（弧度²）: {az_rmse_str}")
    print(f"  NMSE: {az_nmse:.6f}")
    print(f"  NMSE(dB): {az_nmse_db:.4f} dB")
    print("-" * 50)
    print(f"仰角：")
    print(f"  MSE（弧度²）: {el_rmse_str}")
    print(f"  NMSE: {el_nmse:.6f}")
    print(f"  NMSE(dB): {el_nmse_db:.4f} dB")
    print("=" * 50 + "\n")
    return az_diff, el_diff, az_nmse_db, el_nmse_db

if __name__ == "__main__":

    ##=========================== 获取数据 ========================##
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    load_path = os.path.join(parent_dir, 'data_P6', 'vision_net_est_ab_new.pth')  # 波束训练阶段的，vision2ab
    #load_path2 = os.path.join(parent_dir, 'data_P6', 'vision_net_est_ab_no_ic.pth')  # 波束训练阶段的，vision2ab
    #load_path2 = os.path.join(parent_dir, 'data_P6', 'vision_net_est_plusInCA_ab_new.pth')  # 波束训练阶段的，vision2ab
    load_path2 = os.path.join(parent_dir, 'data_P6', 'vision_net_est_Concat_ab_new.pth')  # 波束训练阶段的，vision2ab

    az_diff, el_diff, az_nmse_db, el_nmse_db, = process_vision_data(load_path)
    az_diff2, el_diff2, az_nmse_db2, el_nmse_db2, = process_vision_data(load_path2)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    # ==================== 方位角误差直方图 ====================
    plt.figure(figsize=(8, 3))
    bins = np.linspace(-0.1, 0.1, 50)
    #bins=100
    sns.histplot(az_diff, bins=bins, color='#6B8E23', alpha=0.6, kde=True, stat='probability', label='V2EDA')
    sns.histplot(az_diff2, bins=bins, color='#FFA500', alpha=0.3, kde=True, stat='probability',
                 label='V2EDA w/o $i_c$', line_kws={'linestyle': '--'})
    #plt.title(f'Azimuth Error\\\\\\\\n(MSE = {az_rmse_str})')
    plt.xlabel('Error (rad)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(bottom=0.2)
    plt.xlim(-0.1, 0.1)
    plt.ylim(0, 0.2)

    # 保存第一个图像
    save_path1 = os.path.join(current_dir, 'fig1_2.png')
    plt.savefig(save_path1, dpi=900, bbox_inches='tight')
    print(f"方位角误差直方图已保存至: {save_path1}")
    plt.show()

    # ==================== 仰角误差直方图 ====================
    plt.figure(figsize=(8, 3))
    sns.histplot(el_diff, bins=bins, color='#6B8E23', alpha=0.6, kde=True, stat='probability', label='V2EDA')
    sns.histplot(el_diff2, bins=bins, color='#FFA500', alpha=0.3, kde=True, stat='probability',
                 label='V2EDA w/o $i_c$', line_kws={'linestyle': '--'})
    #plt.title(f'Elevation Error\\\\\\\\n(MSE = {el_rmse_str})')
    plt.xlabel('Error (rad)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(bottom=0.2)
    plt.xlim(-0.2, 0.2)
    plt.ylim(0, 0.1)


    # 保存第二个图像
    save_path2 = os.path.join(current_dir, 'fig2_2.png')
    plt.savefig(save_path2, dpi=900, bbox_inches='tight')
    print(f"仰角误差直方图已保存至: {save_path2}")
    plt.show()



