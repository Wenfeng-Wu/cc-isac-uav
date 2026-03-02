import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.ExifTags import GPSTAGS
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


def set_INOUT_data(batch, model, device):

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
        outputs = model(images_flat, box_flat[:,0:2], box_flat[:,2:4])
        est_vision_output = model.denormalize_output(outputs)

    # 处理图像预测的abdv参数 到 BN4, 并归一化
    est_vision_az = est_vision_output[:,0].view(B, N, -1)  # shape: (B, N, Z)
    est_vision_el = est_vision_output[:,1].view(B, N, -1)
    est_vision_dis = torch.zeros_like(est_vision_az)
    est_vision_v = torch.zeros_like(est_vision_az)

    est_vision_az = normalize_azimuth(est_vision_az)
    est_vision_el = normalize_elevation(est_vision_el)


    # 处理真实的abdv参数到 BN4， 并归一化
    true_az = bs_coor_flat[:,0].view(B, N, -1)  # shape: (B, N, Z)
    true_el = bs_coor_flat[:,1].view(B, N, -1)
    true_dis = bs_coor_flat[:,2].view(B, N, -1)
    true_v = bs_coor_flat[:,3].view(B, N, -1)

    true_az = normalize_azimuth(true_az)   # 110-150
    true_el = normalize_elevation(true_el)  # 10-80
    true_dis = normalize_distance(true_dis)  # 0-1

    # 处理echo估计的abdv参数，并归一化
    est_echo_az = normalize_azimuth(echo_coord_flat[:,0])
    est_echo_el = normalize_elevation(echo_coord_flat[:,1])
    est_echo_dis = normalize_distance(echo_coord_flat[:,2])

    est_echo_az = est_echo_az.view(B, N, -1)
    est_echo_el = est_echo_el.view(B, N, -1)
    est_echo_dis = est_echo_dis.view(B, N, -1)
    est_echo_v = echo_coord_flat[:,3].view(B, N, -1)

    mask_2 = torch.ones(B, N, 4)
    mask_2[:, -1, :] = 0

    output_vision_est = torch.cat([est_vision_az, est_vision_el, est_vision_dis, est_vision_v], dim=-1)
    output_true = torch.cat([true_az, true_el, true_dis, true_v], dim=-1)
    output_echo_esti = torch.cat([est_echo_az, est_echo_el, est_echo_dis, est_echo_v], dim=-1) * mask_2

    return output_vision_est, output_echo_esti, output_true[:,-1,:].unsqueeze(1)

def set_INOUT_data_someError(batch, model, device):

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
        outputs = model(images_flat, box_flat[:,0:2], box_flat[:,2:4])
        est_vision_output = model.denormalize_output(outputs)

    Etype = 'echo_error_is_last_echo'

    if Etype == 'echo_error_is_last_echo':
        mask_prob = 0.9
        if torch.rand(1) < mask_prob:
            # 随机选择一个索引（与原逻辑一致）
            mask_idx = torch.randint(0, echo_coord_flat.shape[0], (1,))  # 修正索引范围为回声数据长度

            # 将回声估计的对应位置替换为视觉模型的预测结果
            # 假设视觉输出est_vision_output[:,0]是az，[:,1]是el（与回声的前两维对应）
            echo_coord_flat[mask_idx, 0] = echo_coord_flat[0, 0]  # 回声az = 视觉az
            echo_coord_flat[mask_idx, 1] = echo_coord_flat[0, 1]  # 回声el = 视觉el

            # 处理修改后的回声估计结果
        est_echo_az = normalize_azimuth(echo_coord_flat[:, 0])
        est_echo_el = normalize_elevation(echo_coord_flat[:, 1])
        est_echo_dis = normalize_distance(echo_coord_flat[:, 2])
    elif Etype == 'echo_error_is_vision':
        mask_prob = 0.2  # 2%的概率触发误差
        if torch.rand(1) < mask_prob:
            # 随机选择一个索引（与原逻辑一致）
            mask_idx = torch.randint(0, echo_coord_flat.shape[0], (1,))  # 修正索引范围为回声数据长度

            # 将回声估计的对应位置替换为视觉模型的预测结果
            # 假设视觉输出est_vision_output[:,0]是az，[:,1]是el（与回声的前两维对应）
            echo_coord_flat[mask_idx, 0] = est_vision_output[mask_idx, 0]  # 回声az = 视觉az
            echo_coord_flat[mask_idx, 1] = est_vision_output[mask_idx, 1]  # 回声el = 视觉el

            # 处理修改后的回声估计结果
        est_echo_az = normalize_azimuth(echo_coord_flat[:, 0])
        est_echo_el = normalize_elevation(echo_coord_flat[:, 1])
        est_echo_dis = normalize_distance(echo_coord_flat[:, 2])
    elif Etype == 'echo_is_all_vision':
        est_echo_az = normalize_azimuth(est_vision_output[:, 0])
        est_echo_el = normalize_elevation(est_vision_output[:, 1])
        est_echo_dis = normalize_distance(echo_coord_flat[:, 2]) * 0 + normalize_distance(echo_coord_flat[0, 2]).tolist()

    elif Etype == 'echo_is_all_last_echo':
        est_echo_az = normalize_azimuth(est_vision_output[:, 0])* 0 + normalize_azimuth(
            echo_coord_flat[0, 0]).tolist()
        est_echo_el = normalize_elevation(est_vision_output[:, 1])* 0 + normalize_elevation(
            echo_coord_flat[0, 1]).tolist()
        est_echo_dis = normalize_distance(echo_coord_flat[:, 2]) * 0 + normalize_distance(
            echo_coord_flat[0, 2]).tolist()

    else:
        est_echo_az = normalize_azimuth(echo_coord_flat[:, 0])
        est_echo_el = normalize_elevation(echo_coord_flat[:, 1])
        est_echo_dis = normalize_distance(echo_coord_flat[:, 2])

    if Etype == 'vision_is_all_echo':
        # 处理图像预测的abdv参数 到 BN4, 并归一化
        est_vision_az = torch.cat([echo_coord_flat[:-1, 0], echo_coord_flat[-2, 0].unsqueeze(-1)],dim=-1).view(B, N, -1)  # shape: (B, N, Z)
        est_vision_el = torch.cat([echo_coord_flat[:-1, 1], echo_coord_flat[-2, 1].unsqueeze(-1)],dim=-1).view(B, N, -1)
        #est_vision_az = echo_coord_flat[:, 0].view(B, N, -1)  # shape: (B, N, Z)
        #est_vision_el = echo_coord_flat[:, 1].view(B, N, -1)
    else:
        # 处理图像预测的abdv参数 到 BN4, 并归一化
        est_vision_az = est_vision_output[:,0].view(B, N, -1)  # shape: (B, N, Z)
        est_vision_el = est_vision_output[:,1].view(B, N, -1)
    est_vision_dis = torch.zeros_like(est_vision_az)
    est_vision_v = torch.zeros_like(est_vision_az)

    est_vision_az = normalize_azimuth(est_vision_az)
    est_vision_el = normalize_elevation(est_vision_el)

    # 处理真实的abdv参数到 BN4， 并归一化
    true_az = bs_coor_flat[:,0].view(B, N, -1)  # shape: (B, N, Z)
    true_el = bs_coor_flat[:,1].view(B, N, -1)
    true_dis = bs_coor_flat[:,2].view(B, N, -1)
    true_v = bs_coor_flat[:,3].view(B, N, -1)

    true_az = normalize_azimuth(true_az)   # 110-150
    true_el = normalize_elevation(true_el)  # 10-80
    true_dis = normalize_distance(true_dis)  # 0-1

    est_echo_az = est_echo_az.view(B, N, -1)
    est_echo_el = est_echo_el.view(B, N, -1)
    est_echo_dis = est_echo_dis.view(B, N, -1)
    est_echo_v = echo_coord_flat[:,3].view(B, N, -1)

    mask_2 = torch.ones(B, N, 4, device=device)
    mask_2[:, -1, :] = 0

    output_vision_est = torch.cat([est_vision_az, est_vision_el, est_vision_dis, est_vision_v], dim=-1)
    output_true = torch.cat([true_az, true_el, true_dis, true_v], dim=-1)
    output_echo_esti = torch.cat([est_echo_az, est_echo_el, est_echo_dis, est_echo_v], dim=-1) * mask_2

    return output_vision_est, output_echo_esti, output_true[:,-1,:].unsqueeze(1)


class FusionNetMini(nn.Module):
    def __init__(self):
        super(FusionNetMini, self).__init__()
        C = 1
        self.conv = nn.Conv1d(in_channels=2 * C, out_channels=2, kernel_size=1)

    def forward(self, part1, part2):
        # part : [B,N,C]
        combined = torch.cat((part1, part2), dim=-1) # shape: [B, N, 2C]
        combined = combined.permute(0, 2, 1)  # shape: [B, 2C, N]
        conv_output = self.conv(combined)   # shape: [B, 2, N]
        param_part1 = conv_output[:, 0, :]  # 第一个参数  [B,N]
        param_part2 = conv_output[:, 1, :]  # 第二个参数  [B,N]
        weighted_part1 = param_part1.unsqueeze(-1) * part1  # shape: [B, N, C]
        weighted_part2 = param_part2.unsqueeze(-1) * part2  # shape: [B, N, C]
        fused_part = weighted_part1 + weighted_part2        # shape: [B, N, C]
        normalization_factor = param_part1 + param_part2    # shape: [B, N]

        # 防止除以0，避免数值错误，使用一个小的 epsilon 值
        epsilon = 1e-8
        fused_ = fused_part / (normalization_factor.unsqueeze(-1) + epsilon)  # shape: [B, N, C]
        return fused_

class FusionNetMini_1(nn.Module):
    def __init__(self):
        super(FusionNetMini_1, self).__init__()
        C = 1
        #self.conv = nn.Conv1d(in_channels=2 * C, out_channels=2, kernel_size=1)
        self.lir_1 = nn.Linear(in_features=32, out_features=6)
        self.lir_2 = nn.Linear(in_features=32, out_features=6)

    def forward(self, images_feature, part1, part2):
        # part : [B,N,C]
        #combined = torch.cat((part1, part2), dim=-1) # shape: [B, N, 2C]
        #combined = combined.permute(0, 2, 1)  # shape: [B, 2C, N]
        #conv_output = self.conv(combined)   # shape: [B, 2, N]
        #param_part1 = conv_output[:, 0, :]  # 第一个参数  [B,N]
        #param_part2 = conv_output[:, 1, :]  # 第二个参数  [B,N]
        param_part1 = self.lir_1(images_feature)
        param_part2 = self.lir_2(images_feature)
        weighted_part1 = param_part1.unsqueeze(-1) * part1  # shape: [B, N, C]
        weighted_part2 = param_part2.unsqueeze(-1) * part2  # shape: [B, N, C]
        fused_part = weighted_part1 + weighted_part2        # shape: [B, N, C]
        normalization_factor = param_part1 + param_part2    # shape: [B, N]

        # 防止除以0，避免数值错误，使用一个小的 epsilon 值
        epsilon = 1e-8
        fused_ = fused_part / (normalization_factor.unsqueeze(-1) + epsilon)  # shape: [B, N, C]
        return fused_


class FusionNet_1(nn.Module):
    def __init__(self):
        super(FusionNet_1, self).__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(5, 5, 5),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.down_1 = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2))


        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.down_2 = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2))

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )



        # 1x1卷积层，将输入拼接后的2C维度映射到1x1的两个参数
        self.fuseMini1 = FusionNetMini_1()
        self.fuseMini2 = FusionNetMini_1()
        self.fuseMini3 = FusionNetMini_1()

    def forward(self, images, vision_ori, echo_ori):
        # vision_ori:  shape: [B, N+1, 4]
        # echo_ori:  shape: [B, N+1, 4]

        # fused_output: [B, N, 4]
        # vision_add: [B,2]

        B,N,C,H,W = images.shape

        images = images.permute(0, 2, 1, 3, 4)

        images_feture = self.conv3d_1(images)
        images_feture = self.down_1(images_feture)
        images_feture = self.conv3d_2(images_feture)
        images_feture = self.down_2(images_feture)
        images_feture = self.conv3d_3(images_feture)
        images_feture = images_feture.reshape(B,-1)
        # 融合前2个参数
        vision_old = vision_ori[:,:-1,:-1]   # shape: [B, N, C]
        echo_old = echo_ori[:,:-1,:-1]   # shape: [B, N, C]

        fuse_parm1 = self.fuseMini1(images_feture, vision_old[:,:,0].unsqueeze(-1), echo_old[:,:,0].unsqueeze(-1))  # shape: [B, N, 1]
        fuse_parm2 = self.fuseMini2(images_feture, vision_old[:,:,1].unsqueeze(-1), echo_old[:,:,1].unsqueeze(-1))  # shape: [B, N, 1]
        #fuse_parm3 = self.fuseMini1(vision_old[:,:,2].unsqueeze(-1), echo_old[:,:,2].unsqueeze(-1))  # shape: [B, N, 1]

        fused_ = torch.cat([fuse_parm1, fuse_parm2], dim=2)   # shape: [B, N, 2]

        # 把各模态的补充信息加上
        echo_add_d = echo_ori[:,:-1,-2].unsqueeze(-1)
        echo_add_v = echo_ori[:,:-1,-1].unsqueeze(-1)
        fused_output = torch.cat((fused_, echo_add_d, echo_add_v), dim=2)

        # 假设图像能先回波一步提取的信息
        vision_add = vision_ori[:, -1, 0:2]
        return fused_output, vision_add, images_feture


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()


        # 1x1卷积层，将输入拼接后的2C维度映射到1x1的两个参数
        self.fuseMini1 = FusionNetMini()
        self.fuseMini2 = FusionNetMini()
        self.fuseMini3 = FusionNetMini()

    def forward(self, vision_ori, echo_ori):
        # vision_ori:  shape: [B, N+1, 4]
        # echo_ori:  shape: [B, N+1, 4]

        # fused_output: [B, N, 4]
        # vision_add: [B,2]

        b,n,c =echo_ori.shape

        # 融合前2个参数
        fuse_parm1 = self.fuseMini1(vision_ori[:,:-1,0].unsqueeze(-1), echo_ori[:,:-1, 0].unsqueeze(-1))  # shape: [B, N, 1]
        fuse_parm2 = self.fuseMini2(vision_ori[:,:-1,1].unsqueeze(-1), echo_ori[:,:-1, 1].unsqueeze(-1))  # shape: [B, N, 1]

        fused_ = torch.cat([fuse_parm1, fuse_parm2], dim=2)   # shape: [B, N, 2]
        # 添加原始echo参数
        echo_add_a = echo_ori[:, :-1, 0].unsqueeze(-1)
        echo_add_b = echo_ori[:, :-1, 1].unsqueeze(-1)

        if c == 2:
            fused_output = torch.cat((fused_, echo_add_a, echo_add_b), dim=2)

        if c == 3:
            echo_add_d = echo_ori[:, :-1, 2].unsqueeze(-1)
            fused_output = torch.cat((fused_, echo_add_a, echo_add_b, echo_add_d), dim=2)

        # 假设图像能先回波一步提取的信息
        vision_add = vision_ori[:, -1, :]
        return fused_output, vision_add



class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class CrossAttention(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        # 历史特征的线性变换
        self.hist_proj = nn.Linear(d_model, d_model)
        # 辅助特征的线性变换
        self.aux_proj = nn.Linear(d_model, d_model)
        # 注意力权重计算
        self.attn = nn.Linear(d_model, 1)
        # 输出层
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hist_feat, aux_feat):
        """
        输入:
        hist_feat: 历史特征 [B, d_model]
        aux_feat: 辅助特征 [B, d_model]
        """
        # 投影变换
        hist_proj = self.hist_proj(hist_feat)  # [B, d_model]
        aux_proj = self.aux_proj(aux_feat)  # [B, d_model]

        # 计算注意力分数
        # 拼接特征并计算注意力权重
        combined = torch.cat([hist_proj.unsqueeze(1), aux_proj.unsqueeze(1)], dim=1)  # [B, 2, d_model]
        attn_scores = self.attn(combined).squeeze(-1)  # [B, 2]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 2]

        # 加权融合
        weighted_hist = hist_proj * attn_weights[:, 0].unsqueeze(-1)  # [B, d_model]
        weighted_aux = aux_proj * attn_weights[:, 1].unsqueeze(-1)  # [B, d_model]
        fused = weighted_hist + weighted_aux

        # 输出投影
        output = self.out_proj(fused)  # [B, d_model]
        return output
