import os
import math
import numpy as np

# 找哪一个编号的图像，无人机正好在图像中央

base_dir = r"D:\Feng\NJU\CODE_WorkCode\dataset\uav_gps_beam\scenario23_dev"
box_dir = os.path.join(base_dir, "resources", "bbox_labels_final")

for img in os.listdir(box_dir):
    bbox_path = os.path.join(box_dir, img)
    #print("1")
    with open(bbox_path, "r") as f:
        numbers = list(map(float, f.read().strip().split()))

    cls_type, x_center_norm, y_center_norm, width_norm, height_norm = numbers

    if x_center_norm>0.48 and x_center_norm<0.52 and y_center_norm>0.48 and y_center_norm<0.52:
        print(bbox_path)
        #break



# D:\Feng\NJU\CODE_WorkCode\dataset\uav_gps_beam\scenario23_dev\resources\bbox_labels_final\image_BS1_8153_17_49_59.txt