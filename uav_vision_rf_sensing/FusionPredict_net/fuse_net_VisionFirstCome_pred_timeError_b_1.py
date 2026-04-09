import os
import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import sys
sys.path.append("...")
from Calibration_nets import Vision_Net
from FusionPredict_net.functions import denormalize_elevation, set_INOUT_data, FusionNet, \
    PositionalEncoding, set_INOUT_data_someError
from data_process import uav_dataset_set_timeAdd


class PredModel(nn.Module):
    def __init__(self, input_dim=5, d_model=16, d_aux=4):
        """
                预测网络
                参数:
                input_dim: 每个时间步的特征维度 (默认4)
                d_model: Transformer编码器维度 (默认1628)
                d_aux: 辅助信号特征维度 (默认4)

        """
        super(PredModel, self).__init__()
        self.fuse = FusionNet()
        # Transformer编码器
        self.hist_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.hist_pool = nn.AdaptiveAvgPool1d(1)  # 序列池化

        # 辅助分支
        self.mlp_aux = nn.Sequential(
            nn.Linear(2, d_aux),
            nn.ReLU())

        self.mlp_out = nn.Sequential(
            nn.Linear(d_model+d_aux, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid())


    def forward(self, vision_ori, echo_ori):
        # vision_ori: [B, N, 4]
        # echo_ori: [B, N, 4]

        fused_input, additional_input = self.fuse(vision_ori, echo_ori)

        # 历史分支处理
        hist_emb = self.hist_linear(fused_input)  # [B, N, d_model]
        hist_emb = self.pos_encoder(hist_emb)     # [B, N, d_model]
        hist_out = self.transformer_encoder(hist_emb) # [B, N, d_model]
        hist_feat = self.hist_pool(hist_out.permute(0, 2, 1)).squeeze(-1)  # [B, d_model]
        # 条件分支处理
        aux_feat = self.mlp_aux(additional_input)  # [B,  d_model]
        decoder_feat = torch.cat((hist_feat, aux_feat), dim=1)
        out = self.mlp_out(decoder_feat)

        return out

    def pred_loss(self, output, labels):
        # [B,3]
        loss = F.mse_loss(output, labels[:,1:2])
        return loss

    def denormalize_output(self,x):
        el = denormalize_elevation(x[:,0]).unsqueeze(1)
        return el


def train_model(right_model, model, train_loader, test_loader, device, save_path, epochs=10, lr=1e-3):
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("Model loaded successfully. re training.")
    else:
        print(f"Model path does not exist: {save_path}. start training.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/Net_pred_ab_Fuse/Fuse_net_timeEst_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs saved to: {log_dir}")

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 1):
            # input_data_vision:[B,T,4]  vision 中获取的参数，包括0~t时刻的估计
            # input_data_echo:[B,T,4] echo 中获取的参数，包括0~t-1时刻的abdv估计
            # output_data_true: [B,1,4] UAV的GPS中提取的t时刻的abdv参数，作为真实数据。 包括t时刻的abdv参数
            input_data_vision, input_data_echo, output_data_true = set_INOUT_data(batch, right_model, device)
            input_data_vision = input_data_vision[:,:,0:2]
            input_data_echo = input_data_echo[:,:,0:3]
            output_data_true = output_data_true[:,:,0:2]
            optimizer.zero_grad()
            output_pred = pred_model(input_data_vision, input_data_echo)
            loss = pred_model.pred_loss(output_pred, output_data_true.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                avg_loss = running_loss / 10
                writer.add_scalar('Train/Loss', avg_loss, (epoch - 1) * len(train_loader) + i)
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {avg_loss:.4f}, "
                )

                running_loss = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in test_loader:
                input_data_vision, input_data_echo, output_data_echo = set_INOUT_data(batch, right_model, device)
                input_data_vision = input_data_vision[:, :, 0:2]
                input_data_echo = input_data_echo[:, :, 0:3]
                output_data_true = output_data_true[:, :, 0:2]
                output_pred = pred_model(input_data_vision, input_data_echo)
                loss = pred_model.pred_loss(
                    output_pred, output_data_echo.squeeze(1))

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


if __name__ == "__main__":
    # 查看Calibration_nets模型
    right_model = Vision_Net()
    print(right_model)
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    right_model_path = '../Calibration_nets/weights/vision_net_est_ab_e100.pth'
    right_model.load_state_dict(torch.load(right_model_path, map_location=device))
    right_model.to(device).eval()
    train_loader, test_loader = uav_dataset_set_timeAdd(7)
    print("dataset load")

    # 创建 FusuinPredict_Net 模型
    pred_model = PredModel()
    print(pred_model)

    model_path = 'weights_P6/fuse_net_timeEst_b_e100_1_snr0_timeAdd.pth'
    epoch = 100

    train_model(right_model, pred_model, train_loader, test_loader, device, save_path=model_path, epochs=epoch, lr=0.5e-4)


