import os
import re
from glob import glob
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class UAVDataset(Dataset):
    def __init__(
        self,
        mask_dir: str,
        bs_coord_base_dir: str,      # e.g. ".../uav_bs_coordinate_bearing{bearing}_dis{dis}"
        cam_coord_dir: str,          # e.g. ".../uav_cam_coordinate"
        box_dir: str,                # e.g. ".../bbox_labels_final"
        echo_dir: str,                # e.g. ".../uav_echo_coor_bearing{bearing}_dis{dis}"
        device_envs: List[Tuple[float, float, float, float]],
        transform: transforms.Compose = None
    ):
        """
        Args:
            mask_dir:            存放 mask_image 的文件夹路径
            bs_coord_base_dir:   格式化路径模版，包含 {bearing} 和 {dis}
            cam_coord_dir:       存放 uav_cam_coordinate 的路径
            box_dir:             存放 bbox 文本标签的文件夹路径，文件名与 mask_image 对应且带额外后缀
            device_envs:         device_env 列表，每项为 (bearing, dis, za, el)
            transform:           mask 图像预处理
        """
        self.mask_dir = mask_dir
        self.bs_coord_base_dir = bs_coord_base_dir
        self.cam_coord_dir = cam_coord_dir
        self.box_dir = box_dir
        self.echo_dir = echo_dir
        self.device_envs = device_envs
        self.transform = transform or transforms.ToTensor()

        # 收集 mask 文件路径和索引
        mask_paths = sorted(glob(os.path.join(self.mask_dir, "image_BS1_*.jpg")))
        idx_pattern = re.compile(r"image_BS1_(\d+)_")

        self.samples = []  # 存储每个样本的信息
        for env in self.device_envs:
            bearing, Ele, dis, za, el = env
            bs_dir = bs_coord_base_dir.format(bearing=int(bearing), dis=dis)  #这里需要单设，因为跟数据产生的不一致了
            echo_dir = echo_dir.format(bearing=int(bearing), dis=dis)  #这里需要单设，因为跟数据产生的不一致了
            for mp in mask_paths:
                m = idx_pattern.search(os.path.basename(mp))
                if not m:
                    continue
                idx = int(m.group(1))

                # 查找对应的 box 文件
                box_path =  os.path.join(self.box_dir, f"image_BS1_{idx}_")+mp[-12:]
                box_path = box_path.replace(".jpg", ".txt")

                #speed_path = os.path.join(self.echo_dir, f"speed_{idx}.txt")

                self.samples.append({
                    "device_env": env,
                    "mask_path": mp,
                    "mask_index": idx,
                    "bs_coord_dir": bs_dir,
                    "box_path": box_path,
                    "echo_coord_dir": echo_dir
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 1. device_env
        bearing, Ele, dis, za, el = item["device_env"]
        device_env_tensor = torch.tensor([bearing, Ele, dis, za, el], dtype=torch.float32)

        # 2. mask_index
        mask_index = item["mask_index"]

        # 3. mask_image
        img = Image.open(item["mask_path"]).convert("RGB")
        mask_tensor = self.transform(img)

        # 4. uav_bs_coor
        bs_coord_file = os.path.join(item["bs_coord_dir"], f"coordinate_{mask_index}.txt")
        with open(bs_coord_file, 'r') as f:
            bs_vals = [float(v) for v in f.read().split()]
        uav_bs_coor = torch.tensor(bs_vals, dtype=torch.float32)

        # 5. uav_cam_coor
        cam_coord_file = os.path.join(self.cam_coord_dir, f"coordinate_{mask_index}.txt")
        with open(cam_coord_file, 'r') as f:
            cam_vals = [float(v) for v in f.read().split()]
        uav_cam_coor = torch.tensor(cam_vals, dtype=torch.float32)

        echo_coord_file = os.path.join(item["echo_coord_dir"], f"coordinate_{mask_index}.txt")
        with open(echo_coord_file, 'r') as f:
            echo_vals = [float(v) for v in f.read().split()]
        uav_echo_coor = torch.tensor(echo_vals, dtype=torch.float32)

        # 6. target_box: 文件中有5个数字，取后4个 (xcenter, ycenter, w, ymax)
        with open(item["box_path"], 'r') as f:
            vals = f.read().split()
        if len(vals) < 5:
            raise ValueError(f"目标框文件 {item['box_path']} 内容不足 5 个值: {vals}")
        box_vals = [float(v) for v in vals[-4:]]
        box_tensor = torch.tensor(box_vals, dtype=torch.float32)

        #with open(item["speed_path"], 'r') as f:
        #    vals = f.read().split()
        #speed_vals = [float(v) for v in vals[:]]
        #speed_tensor = torch.tensor(speed_vals, dtype=torch.float32)


        return {
            "device_env": device_env_tensor,  # [5]
            "mask_index": mask_index,         # int
            "mask_image": mask_tensor,        # [C,H,W]
            "uav_bs_coor": uav_bs_coor,        # [4]
            "uav_cam_coor": uav_cam_coor,      # [4]
            "target_box": box_tensor,           # [4]
            "uav_echo_coor": uav_echo_coor          # [4]
        }


def uav_dataset_set():
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # 设备环境参数（训练和测试共用）
    device_envs = [(315, 0, 100, 233.026899, 88.027546), ]

    def create_loader(data_type):
        """辅助函数：创建指定类型（train/test）的数据集和数据加载器"""
        # 基础路径（根据类型切换train/test）
        base_dir = f"C:/FengFeng/wfwuCode/uav_vision_assisted/dataset"
        type_dir = f"{base_dir}_{data_type}"
        # 构造各路径（统一基于base_dir）
        paths = {
            "mask_dir": f"{type_dir}"+"/uav_gps_beam_process/uav_only",
            "bs_coord_base_dir": f"{base_dir}"+"/uav_gps_beam_process/uav_bs_coordinate_bearing{bearing}_dis{dis}",
            "cam_coord_dir": f"{base_dir}"+"/uav_gps_beam_process/uav_cam_coordinate",
            "box_dir": f"{base_dir}" + "/uav_gps_beam/scenario23_dev/resources/bbox_labels_final",
            "echo_dir": f"{base_dir}"+"/uav_gps_beam_process/uav_bs_echo_coor_bearing{bearing}_dis{dis}_snr0_p100",
            "cam_coord_base_dir": f"{base_dir}"+"/uav_gps_beam_process/uav_cam_coordinate"
            }

        dataset = UAVDataset(
            mask_dir=paths["mask_dir"],
            bs_coord_base_dir=paths["bs_coord_base_dir"],
            cam_coord_dir=paths["cam_coord_dir"],
            box_dir=paths["box_dir"],
            echo_dir=paths["echo_dir"],
            device_envs=device_envs,
            transform=transform
        )

        # 根据数据类型设置DataLoader参数
        batch_size = 32 if data_type == "train" else 1
        shuffle = True if data_type == "train" else False

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 创建训练和测试数据加载器
    train_loader = create_loader("train")
    test_loader = create_loader("test")

    # 验证数据加载（可选，用于确认数据格式）
    def verify_loader(loader):
        for batch in loader:
            # 仅做格式验证，不输出具体数据
            assert "device_env" in batch
            assert "mask_image" in batch
            assert "uav_bs_coor" in batch
            break  # 验证第一个batch即可

        for batch in loader:
            device_envs = batch["device_env"]  # Tensor [B, n, 5]
            mask_index = batch["mask_index"]  # Tensor [B, n, C, H, W]

            print("mask_images:", mask_index.shape)



            break

    verify_loader(train_loader)
    verify_loader(test_loader)

    return train_loader, test_loader

uav_dataset_set()