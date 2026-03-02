import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
# 保存查看box显示在图像目标上的展示
def image_segment(dataset="",device="cpu"):
    for img in os.listdir(dataset):
        if not img.endswith(".jpg"):
            continue
        img_path = os.path.join(dataset,img)
        box_json_path = img_path.replace(".jpg", ".txt").replace("unit1", "resources").replace("camera_data", "bbox_labels_final")
        dst_img_path = img_path.replace("uav_gps_beam", "uav_gps_beam_process").replace(
            "scenario23_dev/unit1/camera_data", "uav_box")
        dirs = os.path.sep.join(dst_img_path.split(os.path.sep)[:-1])
        os.makedirs(dirs, exist_ok=True)

        with open(box_json_path, "r")as f:
            numbers = list(map(float, f.read().strip().split()))

        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]

        cls_type, x_center_norm, y_center_norm, width_norm, height_norm = numbers

        # 计算像素坐标（与之前相同）
        x_center_pixel = x_center_norm * img_width
        y_center_pixel = y_center_norm * img_height
        box_width = width_norm * img_width
        box_height = height_norm * img_height

        x_min = int(x_center_pixel - box_width / 2)
        y_min = int(y_center_pixel - box_height / 2)
        x_max = int(x_center_pixel + box_width / 2)
        y_max = int(y_center_pixel + box_height / 2)

        input_boxes = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 绘制边界框
        BOX_COLOR = (0, 255, 0)  # 框颜色 (BGR格式)
        TEXT_COLOR = (255, 0, 0)  # 文本颜色
        FONT_SCALE = 0.8  # 字体大小
        THICKNESS = 2  # 框线粗细

        # 在图像副本上绘制（保护原图）
        annotated_image = image.copy()
        cv2.rectangle(
            annotated_image,
            (x_min, y_min),
            (x_max, y_max),
            color=BOX_COLOR,
            thickness=THICKNESS
        )
        cv2.imwrite(dst_img_path, image)
        # 保存带标注的图像
        cv2.imwrite(dst_img_path, annotated_image)



if __name__ == '__main__':
    dataset = 'D:/Feng/NJU/CODE_WorkCode/dataset/uav_gps_beam/scenario23_dev/unit1/camera_data'
    image_segment(dataset, "cpu")
