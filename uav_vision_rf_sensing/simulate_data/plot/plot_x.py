# 获取项目根目录（假设脚本在 simulate_data/plot/ 下，根目录是上两级）
import sys
sys.path.append("...")
import torch
from Calibration_nets import Vision_Net
from data_process import uav_dataset_set_time

# 画图设置
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

length = 128

# 查看Calibration_nets模型
right_model = Vision_Net()
print(right_model)
# 设置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
right_model_path = ('C:/FengFeng/wfwuCode/uav_vision_assisted/uav_vision_rf_sensing/Calibration_nets/weights'
                    '/vision_net_est_ab_e100.pth')
right_model.load_state_dict(torch.load(right_model_path, map_location=device))
right_model.to(device).eval()
train_loader, test_loader = uav_dataset_set_time(n_timesteps=length)
print("dataset load")


for epoch in range(1, 2):
    right_model.eval()
    running_loss = 0.0
    for i, batch in enumerate(test_loader, 1):
        # 提取batch数据
        images = batch['mask_images'].to(device)
        boxs = batch['target_boxs'].to(device)
        bs_coors_true = batch['uav_bs_coors_true'].to(device)
        echo_coors_esti = batch['uav_echo_coors_esti'].to(device)

        # 获取 batch size 和时间步数
        B, N = images.shape[0], images.shape[1]

        # Flatten 操作，将 (B, N) 合并为一个维度，得到 (B*N, ...)
        images_flat = torch.flatten(images, start_dim=0, end_dim=1)  # shape: (B*N, C, H, W)
        box_flat = torch.flatten(boxs, start_dim=0, end_dim=1)  # shape: (B*N, 4)
        bs_coor_flat = torch.flatten(bs_coors_true, start_dim=0, end_dim=1)  # shape: (B*N, 4)
        echo_coord_flat = torch.flatten(echo_coors_esti, start_dim=0, end_dim=1)  # shape: (B*N, 4)

        # 模型推理,计算vision预测的uav参数
        with torch.no_grad():
            outputs = right_model(images_flat, box_flat[:, 0:2], box_flat[:, 2:4])
            est_vision_output = right_model.denormalize_output(outputs)

            # plot #print(est_vision_output)
            import matplotlib.pyplot as plt

            # Az.
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Tableau调色板


            start1 = 0
            x_indices1 = torch.arange(start1, start1 + length)

            Adata_out1 = torch.abs(est_vision_output[start1:start1 + length, 0] - bs_coor_flat[start1:start1 + length, 0] ) * 0.01745329
            Adata_ori1 = torch.abs(echo_coord_flat[start1:start1 + length, 0] - bs_coor_flat[start1:start1 + length, 0])  * 0.01745329
            Edata_out1 = torch.abs(est_vision_output[start1:start1 + length, 1] - bs_coor_flat[start1:start1 + length, 1])  * 0.01745329
            Edata_ori1 = torch.abs(echo_coord_flat[start1:start1 + length, 1] - bs_coor_flat[start1:start1 + length, 1])  * 0.01745329

            plt.figure(figsize=(5, 4))
            plt.rcParams['xtick.labelsize'] = 14
            plt.rcParams['ytick.labelsize'] = 14
            plt.plot(x_indices1.numpy(),
                     Adata_out1.numpy(),
                     linestyle='-',
                     color='#6B8E23',
                     linewidth=3,
                     label='V2EDA')

            plt.plot(x_indices1.numpy(),
                     Adata_ori1.numpy(),
                     linestyle='-.',
                     color='#FFA500',
                     linewidth=3,
                     label='Echo',
                     )

            plt.xlabel('Time', fontsize=18)
            plt.ylabel('Azimuth(rad)', fontsize=18)

            plt.legend(loc='best',
                       fontsize=16)
            plt.subplots_adjust(left=0.1)
            plt.subplots_adjust(bottom=0.2)
            plt.show()



            plt.figure(figsize=(5, 4))
            plt.plot(x_indices1.numpy(),
                     Edata_out1.numpy(),
                     linestyle='-',
                     color='#6B8E23',
                     linewidth=3,
                     label='V2EDA')

            plt.plot(x_indices1.numpy(),
                     Edata_ori1.numpy(),
                     linestyle='-.',
                     color='#FFA500',
                     linewidth=3,
                     label='Echo',
                     )

            plt.xlabel('Time', fontsize=18)
            plt.ylabel('Elevation(rad)', fontsize=18)

            plt.legend(loc='best',
                       fontsize=16)
            plt.subplots_adjust(left=0.1)
            plt.subplots_adjust(bottom=0.2)
            plt.show()
            '''
            if isinstance(images_flat, torch.Tensor):
                images = images_flat.permute(0, 2, 3, 1).cpu().numpy()
            else:
                images = np.transpose(images_flat, (0, 2, 3, 1))

            # 处理图像值范围
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)

            # 创建画布
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))

            for i, ax in enumerate(axes.flat):
                if i < 25:
                    ax.imshow(images[i])
                    ax.axis('off')
                else:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
            '''


            if i == 20:
                break
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






