import numpy as np
import torch
from Calibration_nets.vision_net_est_ab_light import Vision_Net
from data_process.set_dataset_uavonly import uav_dataset_set

# 画图设置
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 查看Calibration_nets模型
right_model = Vision_Net()
print(right_model)
# 设置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
right_model_path = 'C:/FengFeng/Muilty_Modal_fusion/uav_vision_assisted/code/uav_vision_rf_sensing/Calibration_nets/weights/vision_net_est_ab_e100.pth'
right_model.load_state_dict(torch.load(right_model_path, map_location=device))
right_model.to(device).eval()
train_loader, test_loader = uav_dataset_set()
print("dataset load")


for epoch in range(1, 2):
    right_model.eval()
    running_loss = 0.0
    for i, batch in enumerate(test_loader, 1):
        # 提取batch数据
        images = batch['mask_image'].to(device)
        boxs = batch['target_box'].to(device)
        bs_coors_true = batch['uav_bs_coor'].to(device)
        echo_coors_esti = batch['uav_echo_coor'].to(device)

        # 获取 batch size 和时间步数
        B, N = images.shape[0], images.shape[1]

        # Flatten 操作，将 (B, N) 合并为一个维度，得到 (B*N, ...)
        #images_flat = torch.flatten(images, start_dim=0, end_dim=1)  # shape: (B*N, C, H, W)
        #box_flat = torch.flatten(boxs, start_dim=0, end_dim=1)  # shape: (B*N, 4)
        #bs_coor_flat = torch.flatten(bs_coors_true, start_dim=0, end_dim=1)  # shape: (B*N, 4)
        #echo_coord_flat = torch.flatten(echo_coors_esti, start_dim=0, end_dim=1)  # shape: (B*N, 4)

        # 模型推理,计算vision预测的uav参数
        with torch.no_grad():
            outputs = right_model(images, boxs[:, 0:2], boxs[:, 2:4])
            est_vision_output = right_model.denormalize_output(outputs)

            # plot #print(est_vision_output)
            import matplotlib.pyplot as plt

            # Az.
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Tableau调色板
            length = 32

            start1 = 0
            x_indices1 = torch.arange(start1, start1 + length)

            Adata_out1 = torch.abs(est_vision_output[start1:start1 + length, 0] - bs_coors_true[start1:start1 + length, 0] ) * 0.01745329
            Adata_ori1 = torch.abs(echo_coors_esti[start1:start1 + length, 0] - bs_coors_true[start1:start1 + length, 0])  * 0.01745329
            Edata_out1 = torch.abs(est_vision_output[start1:start1 + length, 1] - bs_coors_true[start1:start1 + length, 1])  * 0.01745329
            Edata_ori1 = torch.abs(echo_coors_esti[start1:start1 + length, 1] - bs_coors_true[start1:start1 + length, 1])  * 0.01745329

            good1 = Adata_out1 > Adata_ori1
            good2 = Edata_out1 > Edata_ori1
            bad1 = Adata_out1 < Adata_ori1
            bad2 = Edata_out1 < Edata_ori1

            check = Adata_out1 > 0.025 #(0.01745329*0.25)

            indices = torch.where(check == True)

            if indices[0].numel() > 0 :
                import torch
                import matplotlib.pyplot as plt
                import numpy as np

                # 转换为numpy数组并调整维度顺序
                if isinstance(images, torch.Tensor):
                    images = images.permute(0, 2, 3, 1).cpu().numpy()
                else:
                    images = np.transpose(images, (0, 2, 3, 1))

                # 将good1转换为numpy数组（如果是torch tensor）
                if isinstance(check, torch.Tensor):
                    check_mask = check.cpu().numpy()
                else:
                    check_mask = check

                # 计算需要多少行和列来显示所有True的图像
                num_true = np.sum(check_mask)
                num_cols = 5  # 每行显示5张图片
                num_rows = (num_true + num_cols - 1) // num_cols  # 计算需要的行数

                # 创建画布
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

                # 如果只有一行，确保axes是二维数组
                if num_rows == 1:
                    axes = axes.reshape(1, -1)

                # 创建空白白色图像（用于填充False的位置）
                blank_image = np.ones((64, 64, 3), dtype=np.float32)

                # 计数器，用于跟踪当前显示的是第几个True图像
                true_count = 0

                for i in range(32):
                    if check_mask[i]:
                        row = true_count // num_cols
                        col = true_count % num_cols

                        # 显示图像
                        axes[row, col].imshow(images[i])
                        axes[row, col].set_title(f'Sample {i}', fontsize=8)
                        axes[row, col].axis('off')

                        true_count += 1

                # 如果True图像的数量不是num_cols的整数倍，隐藏多余的子图
                for i in range(true_count, num_rows * num_cols):
                    row = i // num_cols
                    col = i % num_cols
                    axes[row, col].axis('off')

                plt.tight_layout()
                plt.show()

                # 打印统计信息
                print(f"总共32个样本，其中{num_true}个为True，{32 - num_true}个为False")



            ## 中文图
'''
            plt.figure(figsize=(9, 3))
            plt.rcParams['xtick.labelsize'] = 14
            plt.rcParams['ytick.labelsize'] = 14
            plt.plot(x_indices1.numpy(),
                     Adata_out1.numpy(),
                     linestyle='-',
                     color='#6B8E23',
                     linewidth=3,
                     label='本文算法')

            plt.plot(x_indices1.numpy(),
                     Adata_ori1.numpy(),
                     linestyle='-.',
                     color='#FFA500',
                     linewidth=3,
                     label='回波',
                     )

            plt.xlabel('时间', fontsize=18)
            plt.ylabel('方位角（弧度）', fontsize=18)

            plt.legend(loc='best',
                       fontsize=16)
            plt.subplots_adjust(left=0.1)
            plt.subplots_adjust(bottom=0.2)
            plt.show()

            plt.figure(figsize=(9, 3))
            plt.plot(x_indices1.numpy(),
                     Edata_out1.numpy(),
                     linestyle='-',
                     color='#6B8E23',
                     linewidth=3,
                     label='本文算法')

            plt.plot(x_indices1.numpy(),
                     Edata_ori1.numpy(),
                     linestyle='-.',
                     color='#FFA500',
                     linewidth=3,
                     label='回波',
                     )

            plt.xlabel('时间', fontsize=18)
            plt.ylabel('仰角（弧度）', fontsize=18)

            plt.legend(loc='best',
                       fontsize=16)
            plt.subplots_adjust(left=0.1)
            plt.subplots_adjust(bottom=0.2)
            plt.show()
'''






