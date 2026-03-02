import os
import torch
import matplotlib.pyplot as plt
import cv2


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 这个代码文件是把目标检测框到的无人机作为主体，把图像进行分割后，输出包含无人机的整个检测框

def image_segment(dataset="",device="cpu"):

    dst_img_dir = 'D:/Feng/NJU/CODE_WorkCode/dataset/uav_gps_beam_process/uav_only'
    for img in os.listdir(dataset):
        dst_img_path = os.path.join(dst_img_dir,img)
        if os.path.exists(dst_img_path):
            print("已存在：", img)
            continue
        img_path = os.path.join(dataset,img)
        dst_img_path = img_path.replace("uav_gps_beam", "uav_gps_beam_process").replace("scenario23_dev/unit1/camera_data", "uav_only")
        box_json_path = img_path.replace(".jpg", ".txt").replace("unit1", "resources").replace("camera_data", "bbox_labels_final")
        with open(box_json_path, "r")as f:
            numbers = list(map(float, f.read().strip().split()))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cls_type, x_center_norm, y_center_norm, width_norm, height_norm = numbers
        # 计算框坐标（与之前相同）
        img_height, img_width = image.shape[:2]
        x_center_pixel = x_center_norm * img_width
        y_center_pixel = y_center_norm * img_height
        box_width = width_norm * img_width
        box_height = height_norm * img_height

        x_min = int(x_center_pixel - box_width / 2)
        y_min = int(y_center_pixel - box_height / 2)
        x_max = int(x_center_pixel + box_width / 2)
        y_max = int(y_center_pixel + box_height / 2)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)

        cropped_image = image[y_min:y_max, x_min:x_max]
        #print(x_min,y_min,x_max,y_max)
        # 保存裁剪图像（推荐直接用 cv2）
        dst_dirs = os.path.sep.join(dst_img_path.split(os.path.sep)[:-1])
        os.makedirs(dst_dirs, exist_ok=True)
        save_img = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_img_path, save_img)

        # 可视化（可选）
        #plt.imshow(cropped_image)
        #plt.axis('off')
        #plt.show()

        print(img_path, "process success!")



if __name__ == '__main__':
    dataset = 'D:/Feng/NJU/CODE_WorkCode/dataset/uav_gps_beam/scenario23_dev/unit1/camera_data'
    image_segment(dataset, "cpu")
