import torch
import matplotlib.pyplot as plt
from Calibration_nets.vision_net_est_ab_light import Vision_Net, denormalize_distance
from FusionPredict_net.functions import set_INOUT_data, denormalize_elevation

from data_process.set_dataset_time import uav_dataset_set


def test_and_save(right_model, model1, model2, test_loader, device, model1_path, model2_path, save_path, ):
    # Load trained model
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.to(device).eval()

    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.to(device).eval()

    #az_true, az_pred, az_vision, az_echo = [], [], [], []
    d_true, d_pred, d_echo = [], [], []
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
            input_data_vision = input_data_vision[:,:,0:2]
            input_data_echo   = input_data_echo[:,:,0:3]
            output_data_true  = output_data_true[:,:,0:3]

            output_pred = pred1_model(input_data_vision, input_data_echo)
            loss = pred1_model.pred_loss(output_pred,output_data_true.squeeze(1))
            pred_param = denormalize_distance(output_pred)
            true_param = denormalize_distance(output_data_true[:,:,2])


            losses1.append(loss.item())

            # 统计t时刻，模型的预测ab，真实的ab，图像估计的ab
            d_pred.extend(pred_param[:, 0].tolist())
            d_true.extend(true_param[:, 0].tolist())

            output_pred = pred2_model(input_data_echo)
            loss = pred2_model.pred_loss(output_pred,output_data_true.squeeze(1))
            pred_param = denormalize_distance(output_pred)
            losses2.append(loss.item())

            #统计t时刻，模型的预测ab，真实的ab，图像估计的ab
            d_echo.extend(pred_param[:, 0].tolist())

            if abs(d_pred[-1]-d_true[-1]) == 0:
                A = ((d_pred[-1]-d_true[-1]) < (d_echo[-1]-d_true[-1]))
                #B = ((el_pred[-1]-el_true[-1]) < (el_echo[-1]-el_true[-1]))
                print("Est Az:" , d_pred[-1]-d_true[-1], d_echo[-1]-d_true[-1], A)
                #print("Est El:" , el_pred[-1]-el_true[-1], el_echo[-1]-el_true[-1], B)


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
        d_diff_pred_true = [true - pred for true, pred in zip(d_true, d_pred)]
        d_diff_echo_true = [true - pred for true, pred in zip(d_true, d_echo)]

        print(f"Test Batch MSE Loss: {sum(losses1) / len(losses1):.4f}")
        print(f"Test Batch MSE Loss: {sum(losses2) / len(losses2):.4f}")

        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model structure and weights_P12
        model_data = {
            'model_state_dict': model1.state_dict(),
            'model_path': model1_path,
        }

        # Save predictions and metadata
        results = {
            'd_true': d_true,
            'd_pred': d_pred,
            'd_echo': d_echo,

            'd_diff_pred_true': d_diff_pred_true,
            'd_diff_echo_true': d_diff_echo_true,

        }

        torch.save({
            'model': model_data,
            'results': results,
        }, save_path)
        print("Saved data_P12 to ", save_path)



if __name__ == "__main__":
    # 查看Calibration_nets模型
    right_model = Vision_Net()
    print(right_model)
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    right_model_path = '../Calibration_nets/weights/vision_net_est_ab_e100.pth'
    right_model.load_state_dict(torch.load(right_model_path, map_location=device))
    right_model.to(device).eval()
    train_loader, test_loader = uav_dataset_set(7)
    print("dataset load")

    # 创建 FusuinPredict_Net 模型
    from FusionPredict_net.fuse_net_VisionFirstCome_pred_d_1 import PredModel as PredModel_MMFE
    pred1_model = PredModel_MMFE()
    print(pred1_model)

    model1_path = 'weights_P6/fuse_net_timeEst_d_e100_1_snr0.pth'

    from FusionPredict_net.Signal_net_EchoOnly_pred_d_1 import PredModel as PredModel_ECHO

    pred2_model = PredModel_ECHO()
    print(pred2_model)

    model2_path = 'weights_P6/echo_net_pred_d_e100_1_snr0.pth'

    output_path = '../simulate_data/data_P6/Comparison_MMFE_pred_d_snr0_p100_1.pth'
    test_and_save(right_model, pred1_model, pred2_model,test_loader, device, model1_path, model2_path, output_path)