import torch
import os
import matplotlib.pyplot as plt

from simulate_data.plot.polt_fig8_cdf_topk import print_topk

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#=======================加载数据=========================================================================================
# 获取当前脚本文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def load_and_unpack(file_name):
    """一次性加载并拆包 checkpoint"""
    load_path = os.path.join(parent_dir, 'data_P6', file_name)
    print(load_path)
    ckpt = torch.load(file_name, weights_only=False)
    #model_path = ckpt['model']['model_path']
    #print("data_P6 from", model_path)
    return ckpt['results']


r0 = load_and_unpack('C:/FengFeng/wfwuCode/uav_vision_assisted/uav_vision_rf_sensing/simulate_data/data_P6/'
                     'Comparison_KL_pred_ab_snr0_lotLostEcho09.pth')
az_true = r0['az_true']
el_true = r0['el_true']
az_pred = r0['az_pred']
el_pred = r0['el_pred']



print('================================KF================================')
echo_topk, echo_times = (print_topk(az_pred, el_pred, az_true, el_true))



