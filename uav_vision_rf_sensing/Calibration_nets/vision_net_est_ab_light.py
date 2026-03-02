import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 数据对齐：视觉语义自动对齐到目标射频语义


def normalize_azimuth(azimuth):
    # 110-150
    range_strat = 110
    range_end = 150
    azimuth = (azimuth - range_strat) / (range_end - range_strat)
    return azimuth


def denormalize_azimuth(azimuth):
    range_strat = 110
    range_end = 150
    azimuth = azimuth * ( range_end - range_strat ) + range_strat
    return azimuth


def normalize_elevation(elevation):
    range_strat = 10
    range_end = 80
    elevation = ( elevation - range_strat) / (range_end - range_strat)
    return elevation


def denormalize_elevation(elevation):
    range_strat = 10
    range_end = 80
    elevation = elevation * ( range_end - range_strat ) + range_strat
    return elevation


def normalize_distance(distance):
    range_strat = 100
    range_end = 300
    distance = ( distance - range_strat) / (range_end - range_strat)
    return distance


def denormalize_distance(distance):
    range_strat = 100
    range_end = 300
    distance = distance * ( range_end - range_strat ) + range_strat
    return distance


class ResMLP(nn.Module):

    def __init__(self, dim):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, dim]
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class CrossAttention(nn.Module):
    """
    Scaled dot-product cross-attention (single head).
    """
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # query, key, value: [B, dim]
        Q = self.w_q(query)  # [B, dim]
        K = self.w_k(key)
        V = self.w_v(value)
        # compute scores: [B, 1, dim] x [B, dim, 1] -> [B,1,1]
        # for batch-wise, compute per-sample
        scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / (self.dim ** 0.5)
        attn = self.softmax(scores)  # [B,1,1]
        out = attn * V.unsqueeze(1)   # [B,1,dim]
        return out.squeeze(1)


class Vision_Net(nn.Module):
    def __init__(self):
        super(Vision_Net, self).__init__()
        # 1.1 Convolutional feature extractor for img
        self.img_embed = nn.Sequential(
            # first conv group
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.AvgPool2d(kernel_size=4, stride=4),  # downsample once
            # second conv group
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.img_for_box = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.resMLP_img = ResMLP(16)

        # 2.1 Embedding for box
        self.box_embed = nn.Linear(4, 16)

        # 2.2 Res-MLP for box
        self.resMLP_box = ResMLP(16)

        # 3.1 Cross-attention modules
        # one head, embed_dim=16
        self.attn_image = CrossAttention(16)
        # 3.2 Cross-attention modules
        self.attn_box = CrossAttention(16)

        # 4. Final output
        self.output_fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )


    def forward(self, images, box_center, box_size):
        # images: [B, C, H, W]
        # box_center: [B, 2]
        # box_size: [B, 2]

        # image embedding:
        B, C, H, W = images.shape
        x = self.img_embed(images)  # [B, 16, 7, 7]

        x_for_box = self.img_for_box(x)
        img_feature = x_for_box.view(B, -1)  #[B,16]


        # box embedding:
        box = torch.cat((box_center, box_size), dim=1)
        box_feature = self.box_embed(box)  # [B,16]

        # resMLP:
        img_feature = self.resMLP_img(img_feature)   # [B,16]
        box_feature = self.resMLP_box(box_feature)       # [B,16]

        # cross-attention:
        box_cross= self.attn_box(query=box_feature, key=box_feature, value=box_feature)
        img_cross= self.attn_image(query=img_feature, key=img_feature, value=img_feature)

        angel_feature = box_cross * img_cross

        out = self.output_fc(angel_feature)  # [B,2]


        return out  #  [B, 2]

    def vision_loss(self, pred, true):
        # pred : [B,2]; az, el;0-1
        # true : [B,3]; az, 0-360;el,70-90;dis,30-390
        true_az = normalize_azimuth(true[:,0])
        true_el = normalize_elevation(true[:,1])
        pred_az = pred[:,0]   # 0-1
        pred_el = pred[:,1]   # 0-1
        a = 0.5
        b = 0.5
        criterion = nn.MSELoss()
        loss1 = criterion(true_az, pred_az)
        loss2 = criterion(true_el, pred_el)


        return a* loss1+ b*loss2

    def denormalize_output(self, x):
        az = denormalize_azimuth(x[:,0]).unsqueeze(1)
        el = denormalize_elevation(x[:,1]).unsqueeze(1)
        return torch.cat((az, el), dim=1)


def train_model(model, train_loader, test_loader, device, save_path, epochs=10, lr=1e-3):
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Model loaded successfully. re training.")
    else:
        print(f"Model path does not exist: {save_path}. start training.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/vision_net_light_est_ab_new/{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs saved to: {log_dir}")

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 1):
            # set input:
            images = batch['mask_image'].to(device)
            box = batch['target_box'].to(device)
            labels = batch['uav_bs_coor'].to(device)
            echo = batch['uav_echo_coor'].to(device)

            optimizer.zero_grad()
            outputs = model(images, box[:, 0:2], box[:, 2:4])
            loss = model.vision_loss(outputs, labels)

            output_pred = model.denormalize_output(outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                avg_loss = running_loss / 10
                writer.add_scalar('Train/Loss', avg_loss, (epoch - 1) * len(train_loader) + i)
                pred_az_out = output_pred[0,0].detach().cpu().numpy()
                pred_el_out = output_pred[0,1].detach().cpu().numpy()

                true_az_out = labels[0,0].detach().cpu().numpy()
                true_el_out = labels[0,1].detach().cpu().numpy()

                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {avg_loss:.4f}, "
                      f"Pred: {pred_az_out}, {pred_el_out}"
                      f"Label: {true_az_out}, {true_el_out}")
                writer.add_text('Sample/Pred_vs_Label',
                                f"Pred: {pred_az_out},{pred_el_out}"
                                f"Label: {true_az_out}, {true_el_out}", epoch)

                running_loss = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in test_loader:
                images = batch['mask_image'].to(device)
                box = batch['target_box'].to(device)
                coor = batch['uav_bs_coor'].to(device)

                outputs = model(images, box[:, 0:2], box[:, 2:4])
                loss = model.vision_loss(outputs, coor)
                val_loss += loss.item()
            val_loss /= len(test_loader)
            print(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")
            writer.add_scalar('Val/Loss', val_loss, epoch)

            # save best model
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path} (val_loss={best_val:.4f})")

    writer.close()


def test_and_save(model, test_loader, device, model_path, save_path):
    # Load trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    az_true, az_pred = [], []
    el_true, el_pred = [], []
    dis_true, dis_pred = [], []
    az_echo, el_echo, dis_echo = [], [], []
    images_list = []
    losses = []
    speeds = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['mask_image'].to(device)
            box = batch['target_box'].to(device)
            coor = batch['uav_bs_coor'].to(device)
            #speed = batch['uav_speed'].to(device)
            echo = batch['uav_echo_coor'].to(device)

            # Ground truth in degrees

            outputs = model(images, box[:,0:2], box[:,2:4])
            loss= model.vision_loss(outputs, coor)
            output_pred = model.denormalize_output(outputs)

            losses.append(loss.item())

            # Convert to degrees for plotting
            images_list.extend(images)
            az_true.extend(output_pred[:,0].tolist())
            el_true.extend(output_pred[:,1].tolist())
            az_pred.extend(coor[:,0].tolist())
            el_pred.extend(coor[:,1].tolist())
            az_echo.extend(echo[:,0].tolist())
            el_echo.extend(echo[:,1].tolist())
            dis_echo.extend(echo[:,2].tolist())

            #speeds.extend(speed.tolist())

            #break

    # Compute differences after collecting all predictions
    az_diff = [true - pred for true, pred in zip(az_true, az_pred)]
    el_diff = [true - pred for true, pred in zip(el_true, el_pred)]
    print(f"Test Batch MSE Loss: {sum(losses)/len(losses):.4f}")

    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model structure and weights_P12
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_path': model_path,
    }

    # Save predictions and metadata
    results = {
        'image': images_list,
        'az_true': az_true,
        'el_true': el_true,
        'az_pred': az_pred,
        'el_pred': el_pred,
        'az_diff': az_diff,
        'el_diff': el_diff,
        'az_echo': az_echo,
        'el_echo': el_echo,
        'dis_echo': dis_echo
    }

    torch.save({
        'model': model_data,
        'results': results,
    }, save_path)
    print("Saved data_P12 to ", save_path)


if __name__ == "__main__":
    # 查看模型
    model = Vision_Net()
    print(model)
    images = torch.randn(8, 3, 64, 64)
    box1 = torch.randn(8, 2)
    box2 = torch.randn(8, 2)
    out = model(images, box1, box2)
    print("model output shape :", out.shape)  # expect [8,3]

    # 设置参数
    device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'weights/vision_net_est_ab_e100.pth'
    output_path = '../simulate_data/data_P6/vision_net_est_ab_new.pth'
    epoch = 100

    from uav_vision_assisted.uav_vision_rf_sensing.data_process import uav_dataset_set_uavonly
    train_loader, test_loader = uav_dataset_set_uavonly()
    print("dataset load")

    # 训练并保存最佳模型
    import time
    s1 = time.time()
    #train_model(model, train_loader, test_loader, device, save_path=model_path, epochs=epoch, lr=0.5e-4)
    e1= time.time()
    print("training time : ", (e1-s1)/60)
    # 载入保存的模型进行测试
    test_and_save(model, train_loader, device, model_path, output_path)

