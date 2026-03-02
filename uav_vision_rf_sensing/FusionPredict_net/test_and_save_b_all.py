import random
import numpy as np
import torch
import sys
sys.path.append("...")
import matplotlib.pyplot as plt
from Calibration_nets.vision_net_est_ab_light import Vision_Net
from FusionPredict_net.functions import set_INOUT_data, set_INOUT_data_someError, denormalize_elevation

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

def test_and_save(right_model, model1, model2, test_loader, device, model1_path, model2_path, save_path, ):
    # Load trained model
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.to(device).eval()

    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.to(device).eval()

    #az_true, az_pred, az_vision, az_echo = [], [], [], []
    el_true, el_pred, el_vision, el_echo = [], [], [], []
    losses1 = []
    losses2 = []
    j = 0
    with torch.no_grad():
        for batch in test_loader:
            j = j+1
            # input_data_vision:[B,T,4]  vision 中获取的参数，包括0~t时刻的估计
            # input_data_echo:[B,T,4] echo 中获取的参数，包括0~t-1时刻的abdv估计
            # output_data_true: [B,1,4] UAV的GPS中提取的abdv参数，作为真实数据。 包括t时刻的abdv参数
            input_data_vision, input_data_echo, output_data_true = set_INOUT_data(batch, right_model, device)
            #input_data_vision, input_data_echo, output_data_true = set_INOUT_data_someError(batch, right_model, device)
            input_data_vision = input_data_vision[:,:,0:2]
            input_data_echo   = input_data_echo[:,:,0:3]
            output_data_true  = output_data_true[:,:,0:2]

            output_pred = pred1_model(input_data_vision, input_data_echo)
            loss = pred1_model.pred_loss(output_pred,output_data_true.squeeze(1))
            pred_param = pred1_model.denormalize_output(output_pred)
            true_param = denormalize_elevation(output_data_true[:,:,1])
            vision_param = denormalize_elevation(input_data_vision[:,-1,1])

            losses1.append(loss.item())

            # 统计t时刻，模型的预测ab，真实的ab，图像估计的ab
            el_pred.extend(pred_param[:, 0].tolist())
            el_true.extend(true_param[:, 0].tolist())
            el_vision.extend(vision_param.tolist())

            output_pred = pred2_model(input_data_echo)
            loss = pred2_model.pred_loss(output_pred,output_data_true.squeeze(1))
            pred_param = pred2_model.denormalize_output(output_pred)
            losses2.append(loss.item())

            #统计t时刻，模型的预测ab，真实的ab，图像估计的ab
            el_echo.extend(pred_param[:, 0].tolist())

            if abs(el_pred[-1]-el_true[-1]) == abs(el_echo[-1]-el_vision[-1]):
                A = ((el_pred[-1]-el_true[-1]) < (el_echo[-1]-el_true[-1]))
                #B = ((el_pred[-1]-el_true[-1]) < (el_echo[-1]-el_true[-1]))
                print("Est Az:" , el_pred[-1]-el_true[-1], el_echo[-1]-el_true[-1], A)
                #print("Est El:" , el_pred[-1]-el_true[-1], el_echo[-1]-el_true[-1], B)

                # 假设你的tensor已经存在，名为images
                # images的形状为 [1, 7, 3, 64, 64]

                # 首先移除batch维度，得到 [7, 3, 64, 64]
                images = images.squeeze(0)

                # 创建1行7列的子图
                fig, axes = plt.subplots(1, 7, figsize=(21, 3))

                # 绘制每个图像
                for i in range(7):
                    # 选择当前子图
                    ax = axes[i]

                    # 调整通道顺序：从 [3, 64, 64] 转换为 [64, 64, 3]
                    img = images[i].permute(1, 2, 0)

                    # 显示图像（RGB图像不需要指定cmap）
                    ax.imshow(img)

                    # 设置标题和关闭坐标轴
                    ax.set_title(f'Image {i + 1}')
                    ax.axis('off')

                # 调整子图间距
                plt.tight_layout()
                plt.show()
                #plt.savefig('C:/FengFeng/Muilty_Modal_fusion/uav_vision_assisted/code/uav_vision_rf_sensing/simulate_data/plot/bad_image/' + str(A)+'_' +'_{}.png'.format(j))


            # Compute differences after collecting all predictions
        el_diff_vision_true = [true - pred for true, pred in zip(el_true, el_vision)]
        el_diff_pred_true = [true - pred for true, pred in zip(el_true, el_pred)]
        el_diff_echo_true = [true - pred for true, pred in zip(el_true, el_echo)]

        print(f"Test Batch MSE Loss: {sum(losses1) / len(losses1):.4f}")
        print(f"Test Batch MSE Loss: {sum(losses2) / len(losses2):.4f}")

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
        # 1. Vision-Only与真实值的RMSE
        rmse_vision = calculate_rmse(el_diff_vision_true)  # 误差：true - vision_pred

        # 2. Multi-Modal与真实值的RMSE
        rmse_pred = calculate_rmse(el_diff_pred_true)  # 误差：true - pred

        # 3. Echo-Only与真实值的RMSE
        rmse_echo = calculate_rmse(el_diff_echo_true)  # 误差：true - echo_pred

        # ==================== 打印结果 ====================
        print("仰角RMSE计算结果：")
        print(f"Vision-Only vs 真实值: {rmse_vision:.4f}")  # 单位与误差一致（如度）
        print(f"Multi-Modal vs 真实值: {rmse_pred:.4f}")
        print(f"Echo-Only vs 真实值: {rmse_echo:.4f}")

        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model structure and weights_P12
        model_data = {
            'model_state_dict': model1.state_dict(),
            'model_path': model1_path,
        }

        # Save predictions and metadata
        results = {
            'el_true': el_true,
            'el_pred': el_pred,
            'el_vision': el_vision,
            'el_echo': el_echo,

            'el_diff_vision_true': el_diff_vision_true,
            'el_diff_pred_true': el_diff_pred_true,
            'el_diff_echo_true': el_diff_echo_true,

        }

        torch.save({
            'model': model_data,
            'results': results,
        }, save_path)
        print("Saved data_P6 to ", save_path)



if __name__ == "__main__":
    # 查看Calibration_nets模型
    right_model = Vision_Net()
    print(right_model)
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    right_model_path = '../Calibration_nets/weights/vision_net_est_ab_e100.pth'
    right_model.load_state_dict(torch.load(right_model_path, map_location=device))
    right_model.to(device).eval()
    train_loader, test_loader = uav_dataset_set_time(7)
    print("dataset load")
    # 创建 FusuinPredict_Net 模型
    from FusionPredict_net.fuse_net_VisionFirstCome_pred_b_1 import PredModel as PredModel_MMFE
    pred1_model = PredModel_MMFE()
    print(pred1_model)
    model1_path = 'weights_P6/fuse_net_timeEst_b_e100_1_snr0.pth'
    from FusionPredict_net.Signal_net_EchoOnly_pred_b_1 import PredModel as PredModel_ECHO
    pred2_model = PredModel_ECHO()
    print(pred2_model)
    model2_path = 'weights_P6/echo_net_pred_b_e100_1_snr0.pth'

    output_path = '../simulate_data/data_P6/Comparison_MMFE_pred_b_snr-3.pth'
    test_and_save(right_model, pred1_model, pred2_model,test_loader, device, model1_path, model2_path, output_path)

    '''
    ## snr=0
    ## 仰角RMSE计算结果：
    ## Vision-Only vs 真实值: 1.6321
    ## Multi-Modal vs 真实值: 0.7396
    ## Echo-Only vs 真实值: 0.7648
    ## Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_p100_1.pth
       
    ## snr=-1
    ## 仰角RMSE计算结果：
    ## Vision-Only vs 真实值: 1.6321
    ## Multi-Modal vs 真实值: 0.7360
    ## Echo-Only vs 真实值: 0.7711
    ## Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr-1_p100_1.pth
    
仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7772
Echo-Only vs 真实值: 0.8377
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr-2_p100_1.pth

    ## snr=-5
    ## 仰角RMSE计算结果：
    ## Vision-Only vs 真实值: 1.6321
    ## Multi-Modal vs 真实值: 1.2237
    ## Echo-Only vs 真实值: 1.3477
    
    仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 1.1097
Echo-Only vs 真实值: 1.2570
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_p100_1_someErrorInEcho_002.pth
    
    Echo失效:
    仰角RMSE计算结果(d角度继承旧记忆，角度继承vision)：
    Vision-Only vs 真实值: 1.6321
    Multi-Modal vs 真实值: 1.2203
    Echo-Only vs 真实值: 1.2672
    Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_p100_ErrorInecho.pth


    Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7991
Echo-Only vs 真实值: 0.8670
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_p100_1_someErrorInEcho_02_echoonly.pth

    
    仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 1.5210
Echo-Only vs 真实值: 1.6216
Saved data_P12 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_p100_1_someErrorInEcho_09_echoonly.pth
'''

