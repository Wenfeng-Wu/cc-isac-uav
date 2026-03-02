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
    ckpt = torch.load(load_path)
    model_path = ckpt['model']['model_path']
    print("data_P6 from", model_path)
    return ckpt['results']


r0 = load_and_unpack('Comparison_MMFE_pred_a_snr0_lotLostEcho09_echo-only.pth')
az_true = r0['az_true']
#az_pred1 = r0['az_pred']
az_vision= r0['az_vision']
az_echoOnly = r0['az_echo']
#az_diff_pred1_true = [abs(x) for x in r0['az_diff_pred_true']]
az_diff_vision_true = [abs(x) for x in r0['az_diff_vision_true']]
az_diff_echoOnly_true = [abs(x) for x in r0['az_diff_echo_true']]

r0 = load_and_unpack('Comparison_MMFE_pred_a_snr0_lotLostEcho09_multimodal.pth')
#az_true = r0['az_true']
az_pred1 = r0['az_pred']
#az_vision= r0['az_vision']
#az_echoOnly = r0['az_echo']
az_diff_pred1_true = [abs(x) for x in r0['az_diff_pred_true']]
#az_diff_vision_true = [abs(x) for x in r0['az_diff_vision_true']]
#az_diff_echoOnly_true = [abs(x) for x in r0['az_diff_echo_true']]

r0 = load_and_unpack('Comparison_MMFE_pred_a_snr0_lotLostEcho09_echo-only.pth')
az_true = r0['az_true']
#az_pred1 = r0['az_pred']
az_vision= r0['az_vision']
az_echoOnly = r0['az_echo']
#az_diff_pred1_true = [abs(x) for x in r0['az_diff_pred_true']]
az_diff_vision_true = [abs(x) for x in r0['az_diff_vision_true']]
az_diff_echoOnly_true = [abs(x) for x in r0['az_diff_echo_true']]


r0 = load_and_unpack('Comparison_MMFE_pred_b_snr0_lotLostEcho09.pth')
el_true = r0['el_true']
el_pred1 =  r0['el_pred']
el_vision =  r0['el_vision']
el_echoOnly = r0['el_echo']
el_diff_pred1_true = [abs(x) for x in r0['el_diff_pred_true']]
el_diff_vision_true = [abs(x) for x in r0['el_diff_vision_true']]
el_diff_echoOnly_true = [abs(x) for x in r0['el_diff_echo_true']]

print('===============================vision===============================')
vision_topk, vision_times = (print_topk(az_vision, el_vision, az_true, el_true))
print('================================echo================================')
echo_topk, echo_times = (print_topk(az_echoOnly, el_echoOnly, az_true, el_true))
print('================================MMFE================================')
fused_topk, fused_times = (print_topk(az_pred1, el_pred1, az_true, el_true))


