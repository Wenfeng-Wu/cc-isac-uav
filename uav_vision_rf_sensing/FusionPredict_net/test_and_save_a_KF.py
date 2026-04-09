import random
import numpy as np
import torch
import sys
sys.path.append("...")
import matplotlib.pyplot as plt
from Calibration_nets.vision_net_est_ab_light import Vision_Net
from FusionPredict_net.functions import set_INOUT_data, set_INOUT_data_someError, denormalize_azimuth, denormalize_elevation

from data_process import uav_dataset_set_time
# 固定随机种子（自定义一个整数，如42，关键是每次运行用同一个值）
SEED = 42

# 1. 固定Python内置random库的种子
random.seed(SEED)

# 2. 固定numpy的种子
np.random.seed(SEED)

# 3. 固定PyTorch的种子（若用PyTorch生成数据）
torch.manual_seed(SEED)  # CPU上的种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # 单GPU种子
    torch.cuda.manual_seed_all(SEED)  # 多GPU种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作可复现
    torch.backends.cudnn.benchmark = False  # 禁用自动优化（可能影响速度，但保证复现）

# 4. 固定pandas的种子（pandas依赖numpy，通常固定numpy即可）
pd_np = np.random.RandomState(SEED)

def kalman_predict_angle(angle_seq, dt=1.0):
    """
    使用Kalman Filter根据历史角度预测下一时刻角度
    angle_seq: 长度为6的角度序列
    """

    # 状态 [angle, angular_velocity]
    F = np.array([[1, dt],
                  [0, 1]])

    H = np.array([[1, 0]])

    Q = np.array([[1e-4, 0],
                  [0, 1e-4]])

    R = np.array([[1e-2]])

    # 初始状态
    v0 = (angle_seq[1] - angle_seq[0]) / dt
    x = np.array([[angle_seq[0]],
                  [v0]])

    P = np.eye(2)

    for z in angle_seq:

        z = np.array([[z]])

        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        # update
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (z - H @ x)
        P = (np.eye(2) - K @ H) @ P

    # 再预测一步
    x_pred = F @ x

    return x_pred[0,0]

def test_and_save(right_model, test_loader, device, save_path, ):
    # Load trained model
    az_true, az_pred, az_vision, az_echo = [], [], [], []
    el_true, el_pred, el_vision, el_echo = [], [], [], []
    losses1 = []
    losses2 = []
    j = 0
    for batch in test_loader:
        j = j+1
        # input_data_vision:[B,T,4]  vision 中获取的参数，包括0~t时刻的估计
        # input_data_echo:[B,T,4] echo 中获取的参数，包括0~t-1时刻的abdv估计
        # output_data_true: [B,1,4] UAV的GPS中提取的abdv参数，作为真实数据。 包括t时刻的abdv参数
        #input_data_vision, input_data_echo, output_data_true = set_INOUT_data(batch, right_model, device)
        input_data_vision, input_data_echo, output_data_true = set_INOUT_data_someError(batch, right_model, device)
        input_data_vision = input_data_vision[:,:,0:2]
        input_data_echo   = input_data_echo[:,:,0:3]
        output_data_true  = output_data_true[:,:,0:3]

        true_param1 = denormalize_azimuth(output_data_true.squeeze(1))
        true_param2 = denormalize_elevation(output_data_true.squeeze(1))

        az_true.extend(true_param1[:, 0].tolist())
        el_true.extend(true_param2[:, 1].tolist())
        # 用KL算法从历史的input_data_echo预测下一个时间点的方位角

        az_seq = input_data_echo[0,:-1,0].cpu().numpy()
        el_seq = input_data_echo[0,:-1,1].cpu().numpy()

        az_next = kalman_predict_angle(az_seq)
        el_next = kalman_predict_angle(el_seq)

        az_pred.append(denormalize_azimuth(az_next))
        el_pred.append(denormalize_elevation(el_next))


    az_diff_pred_true = [true - pred for true, pred in zip(az_true, az_pred)]
    el_diff_pred_true = [true - pred for true, pred in zip(el_true, el_pred)]


    def calculate_rmse(errors):
        """
        计算RMSE（均方根误差）

        参数：
            errors: 误差列表/数组（格式：真实值 - 预测值）
        返回：
            rmse: 均方根误差（单位与误差一致）
        """
        # 转换为numpy数组便于计算
        errors_np = np.array(errors)

            # 1. 计算MSE（误差平方的均值）
        mse = np.mean(errors_np ** 2)

            # 2. 计算RMSE（MSE的平方根）
        rmse = np.sqrt(mse)

        return rmse

    # ==================== 计算三组数据的RMSE ====================

    rmse_pred_az = calculate_rmse(az_diff_pred_true)  # 误差：true - pred
    rmse_pred_el = calculate_rmse(el_diff_pred_true)  # 误差：true - pred

        # ==================== 打印结果 ====================
    print("方位角RMSE计算结果：")
    print(f"KL算法预测的 vs 真实值: {rmse_pred_az:.4f}")
    print("仰角RMSE计算结果：")
    print(f"KL算法预测的 vs 真实值: {rmse_pred_el:.4f}")
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model structure and weights_P12
    # Save predictions and metadata
    results = {
        'az_true': az_true,
        'az_pred': az_pred,
        'az_diff_pred_true': az_diff_pred_true,
        'el_true': el_true,
        'el_pred': el_pred,
        'el_diff_pred_true': el_diff_pred_true,
    }

    torch.save({
        'model': None,
        'results': results,
    }, save_path)
    print("Saved data_P6 to ", save_path)



if __name__ == "__main__":
    # 查看Calibration_nets模型
    right_model = Vision_Net()
    print(right_model)
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device=torch.device('cpu')
    print("device:", device)
    right_model_path = '../Calibration_nets/weights/vision_net_est_ab_e100.pth'
    right_model.load_state_dict(torch.load(right_model_path, map_location=device))
    right_model.to(device).eval()
    train_loader, test_loader = uav_dataset_set_time(7)
    print("dataset load")

    output_path = '../simulate_data/data_P6/Comparison_KL_pred_ab_snr0_lotLostEcho09.pth'
    test_and_save(right_model,test_loader, device, output_path)

'''
    ================RMSE
    KL预测器：
    方位角RMSE计算结果：
    KL算法预测的 vs 真实值: 0.8405
    仰角RMSE计算结果：
    KL算法预测的 vs 真实值: 0.9369
    Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr0_1.pth
    
    KL预测器：
    方位角RMSE计算结果：
    KL算法预测的 vs 真实值: 0.9943
    仰角RMSE计算结果：
    KL算法预测的 vs 真实值: 1.0568
    Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr-1_1.pth
    
    
     snr0
     方位角RMSE计算结果：
     Vision-Only vs 真实值: 0.5051
     Multi-Modal vs 真实值: 0.3806
     Echo-Only vs 真实值: 0.5236
     Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1.pth
     
     
     snr=0
     方位角RMSE计算结果：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.4459
    Echo-Only vs 真实值: 0.5236
    Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_a_snr0_1.pth

     snr=-1
     方位角RMSE计算结果：
     Vision-Only vs 真实值: 0.5051
     Multi-Modal vs 真实值: 0.4210
     Echo-Only vs 真实值: 0.5823
     Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr-1_1.pth
     
     SNR=-2
     方位角RMSE计算结果：
     Vision-Only vs 真实值: 0.5051
     Multi-Modal vs 真实值: 0.4580
     Echo-Only vs 真实值: 0.6467
    

    snr=-5:
    方位角RMSE计算结果：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.6974
    Echo-Only vs 真实值: 1.0263

    snr=-1:
    echo error 001
    方位角RMSE计算结果：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.4208
    Echo-Only vs 真实值: 0.5818
    
    echo error 002
    方位角RMSE计算结果：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.4212
    Echo-Only vs 真实值: 0.5823
    
    echo error 003
    方位角RMSE计算结果：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.4202
    Echo-Only vs 真实值: 0.5813
    
    echo error 004
    方位角RMSE计算结果：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.4214
    Echo-Only vs 真实值: 0.5816
    
    方位角RMSE计算结果(d保留，角度继承vision)：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.4725
    Echo-Only vs 真实值: 0.5279
    Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_EroorInEcho.pth
    
    方位角RMSE计算结果(d=0.5，角度继承vision)：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.5689
    Echo-Only vs 真实值: 1.6897
    Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_EroorInEcho.pth

    方位角RMSE计算结果(d=0.1，角度继承vision)：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.5356
    Echo-Only vs 真实值: 1.2969
    Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_EroorInEcho.pth
    
    方位角RMSE计算结果(d=0，角度继承vision)：
    Vision-Only vs 真实值: 0.5051
    Multi-Modal vs 真实值: 0.5689
    Echo-Only vs 真实值: 1.6897
    Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_EroorInEcho.pth
    

Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4584
Echo-Only vs 真实值: 0.6472
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_someEroorInEcho_002_echo-only.pth

方位角RMSE计算结果：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4563
Echo-Only vs 真实值: 0.6459
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_someEroorInEcho_02_echo-only.pth

方位角RMSE计算结果：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4594
Echo-Only vs 真实值: 0.5251
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_1_someEroorInEcho_09_echo-only.pth

'''