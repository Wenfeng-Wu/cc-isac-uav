import os
import re
from glob import glob
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class UAVDatasetLSTM(Dataset):
    def __init__(self, mask_dir: str,
                 bs_coord_base_dir: str,
                 bs_echo_dir: str,
                 cam_coord_base_dir: str,
                 box_dir: str,
                 device_envs: List[Tuple[float, float, float, float]],
                 n_timesteps: int = 5,  # 每个包包含n个时间戳
                 transform: transforms.Compose = None):
        """
        Args:
            n_timesteps: 每个包包含n个时间戳的数据。
        """
        self.mask_dir = mask_dir
        self.bs_coord_base_dir = bs_coord_base_dir
        self.echo_coord_dir = bs_echo_dir
        self.cam_coord_base_dir = cam_coord_base_dir
        self.box_dir = box_dir
        self.device_envs = device_envs
        self.n_timesteps = n_timesteps
        self.transform = transform or transforms.ToTensor()

        # 收集 mask 文件路径和索引
        mask_files = glob(os.path.join(self.mask_dir, "image_BS1_*.jpg"))
        idx_pattern = re.compile(r"image_BS1_(\d+)_")
        mask_info = []  # 存储元组(index, path)
        seen_indices = set()
        for mp in mask_files:
            base_mp = os.path.basename(mp)
            m = idx_pattern.search(base_mp)
            if not m:
                continue
            idx = int(m.group(1))
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            mask_info.append((idx, mp))
        # 按索引排序
        mask_info.sort(key=lambda x: x[0])
        self.samples = []  # 存储每个样本的信息
        for env in self.device_envs:
            bearing, Ele, dis, za, el = env
            bs_dir = bs_coord_base_dir.format(bearing=int(bearing), dis=dis)
            echo_dir = bs_echo_dir.format(bearing=int(bearing), dis=dis)
            for idx, mp in mask_info:
                base_mp = os.path.basename(mp)
                # 构建box_path: 同名的txt文件
                box_file = base_mp.replace(".jpg", ".txt")
                box_path = os.path.join(self.box_dir, box_file)
                self.samples.append({
                    "device_env": env,
                    "mask_path": mp,
                    "mask_index": idx,
                    "bs_coord_base_dir": bs_dir,
                    "echo_coord_dir": echo_dir,
                    "box_path": box_path
                })

    def __len__(self):
        # 返回可用的样本数量，每个样本包含n个时间戳
        return len(self.samples) - self.n_timesteps

    def __getitem__(self, idx):
        # 获取一个样本的过去n个时间戳数据
        input_data = []
        target_data = []

        # 获取连续n个时间戳的数据
        for i in range(self.n_timesteps):
            sample_idx = idx + i
            item = self.samples[sample_idx]

            # 1. device_env
            bearing, Ele, dis, za, el = item["device_env"]
            device_env_tensor = torch.tensor([bearing, Ele, dis, za, el], dtype=torch.float32)

            # 2. mask_image
            img = Image.open(item["mask_path"]).convert("RGB")
            mask_tensor = self.transform(img)

            # 3. uav_bs_coor
            bs_coord_base_true_file = os.path.join(item["bs_coord_base_dir"], f"coordinate_{item['mask_index']}.txt")
            with open(bs_coord_base_true_file, 'r') as f:
                bs_vals = [float(v) for v in f.read().split()]
            uav_bs_coor_true = torch.tensor(bs_vals, dtype=torch.float32)

            # 4. uav_cam_coor
            cam_coord_base_true_file = os.path.join(self.cam_coord_base_dir, f"coordinate_{item['mask_index']}.txt")
            with open(cam_coord_base_true_file, 'r') as f:
                cam_vals = [float(v) for v in f.read().split()]
            uav_cam_coor_true = torch.tensor(cam_vals, dtype=torch.float32)

            # 4. echo_coor
            echo_coord_esti_file = os.path.join(item["echo_coord_dir"], f"coordinate_{item['mask_index']}.txt")
            with open(echo_coord_esti_file, 'r') as f:
                echo_vals = [float(v) for v in f.read().split()]
            uav_echo_coor_esti = torch.tensor(echo_vals, dtype=torch.float32)

            # 5. target_box: 文件中有5个数字，取后4个 (xcenter, ycenter, w, ymax)
            with open(item["box_path"], 'r') as f:
                vals = f.read().split()
            if len(vals) < 5:
                raise ValueError(f"目标框文件 {item['box_path']} 内容不足 5 个值: {vals}")
            box_vals = [float(v) for v in vals[-4:]]
            box_tensor = torch.tensor(box_vals, dtype=torch.float32)

            # 每个时间戳的数据
            input_data.append({
                "mask_index": torch.tensor([item["mask_index"]]),
                "device_env": device_env_tensor,
                "mask_image": mask_tensor,
                "uav_bs_coor_true": uav_bs_coor_true,
                "uav_cam_coor_true": uav_cam_coor_true,
                "uav_echo_coor_esti": uav_echo_coor_esti,
                "target_box": box_tensor
            })

        # 将输入数据（过去n个时间戳）拼接成合适的输入格式
        input_data = {
            "mask_index": torch.stack([x["mask_index"] for x in input_data]),
            "device_envs": torch.stack([x["device_env"] for x in input_data]),
            "mask_images": torch.stack([x["mask_image"] for x in input_data]),
            "uav_bs_coors_true": torch.stack([x["uav_bs_coor_true"] for x in input_data]),
            "uav_cam_coors_true": torch.stack([x["uav_cam_coor_true"] for x in input_data]),
            "uav_echo_coors_esti": torch.stack([x["uav_echo_coor_esti"] for x in input_data]),
            "target_boxs": torch.stack([x["target_box"] for x in input_data])
        }

        return input_data


def uav_dataset_set(n_timesteps=7):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    device_envs = [
        (315, 0, 100, 233.026899, 88.027546),
    ]

    dataset = UAVDatasetLSTM(
        mask_dir=r"C:/FengFeng/wfwuCode/uav_vision_assisted/dataset\uav_gps_beam_process\uav_only",
        bs_coord_base_dir=r"C:/FengFeng/wfwuCode/uav_vision_assisted/dataset\uav_gps_beam_process\uav_bs_coordinate_bearing{bearing}_dis{dis}",
        bs_echo_dir=r"C:/FengFeng/wfwuCode/uav_vision_assisted/dataset\uav_gps_beam_process\uav_bs_echo_timeAdd_coor_bearing{bearing}_dis{dis}",
        cam_coord_base_dir=r"C:/FengFeng/wfwuCode/uav_vision_assisted/dataset\uav_gps_beam_process\uav_cam_coordinate",
        box_dir=r"C:/FengFeng/wfwuCode/uav_vision_assisted/dataset\uav_gps_beam\scenario23_dev\resources\bbox_labels_final",
        device_envs=device_envs,
        n_timesteps=n_timesteps,  # 每个包包含n个时间戳
        transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
    )

    import torch
    from torch.utils.data import random_split, DataLoader

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # 固定 Generator 的种子
    generator = torch.Generator().manual_seed(42)

    # 划分数据集
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    # 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for batch in train_loader:
        device_envs = batch["device_envs"]  # Tensor [B, n, 5]
        mask_images = batch["mask_images"]  # Tensor [B, n, C, H, W]
        target_box = batch["target_boxs"]  # Tensor [B, n, 4]
        uav_bs_coors_true = batch["uav_bs_coors_true"]  # Tensor [B, n, 4]
        uav_cam_coors_true = batch["uav_cam_coors_true"]  # Tensor [B, n, 4]
        uav_echo_coors_esti = batch["uav_echo_coors_esti"]  # Tensor [B, n, 4]
        #print(mask_images)
        break
    return train_loader, test_loader

uav_dataset_set()