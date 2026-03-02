import os
import math
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt


# ----------------------------- #
# 计算无人机相对于相机和基站的坐标信息并存储
# $\theta$：方位角（azimuth），即朝北顺时针旋转的角度
# $\phi$：仰角（elevation），即向上看的角度
# $r$：直线距离
# $v$：径向速度
# 基站位置自己设置
# ----------------------------- #
# 地理坐标转换为 ECEF 坐标： 使用 WGS84 椭球模型将经纬度和高度转换为 ECEF 坐标。
# 计算相对位置向量： 通过目标点的 ECEF 坐标减去参考点的 ECEF 坐标，得到相对位置向量。
# 转换为 ENU 坐标系： 将相对位置向量从 ECEF 坐标系转换为以参考点为原点的 ENU 坐标系。
# GIS Stack Exchange
# 计算仰角、方位角、距离、径向速度： 使用 ENU 坐标计算仰角、方位角和直线距离。

def geodetic_to_ecef(lat, lon, alt):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x, y, z = transformer.transform(lon, lat, alt)
    return np.array([x, y, z])


def ecef_to_enu(delta_ecef, lat_ref, lon_ref):
    lat_rad = math.radians(lat_ref)
    lon_rad = math.radians(lon_ref)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # 构建旋转矩阵
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])

    enu = R @ delta_ecef
    return enu


def calculate_aer(enu, x_speed, y_speed, z_speed):
    # 通过相对位置ENU，转换为相对球坐标系(角度值)，并计算目标相对球心的径向速度
    east, north, up = enu
    distance = np.linalg.norm(enu)
    azimuth = math.degrees(math.atan2(east, north)) % 360
    elevation = math.degrees(math.asin(up / distance))

    v_radial = (x_speed * np.cos(elevation) * np.sin(azimuth) +
                y_speed * np.cos(elevation) * np.cos(azimuth) +
                z_speed * np.sin(elevation))
    return azimuth, elevation, distance, v_radial


# ----------------------------- #
# Path Setup (use raw strings)
# ----------------------------- #

base_dir = r"D:\Feng_Feng\dataset\uav_gps_beam"
gps_cam_file = os.path.join(base_dir, "scenario23_dev", "unit1", "GPS_data", "gps_location.txt")
uav_gps_dir = os.path.join(base_dir, "scenario23_dev", "unit2", "GPS_data")
uav_height_dir = os.path.join(base_dir, "scenario23_dev", "unit2", "height")
uav_xspeed_dir = os.path.join(base_dir, "scenario23_dev", "unit2", "x_speed")
uav_yspeed_dir = os.path.join(base_dir, "scenario23_dev", "unit2", "y_speed")
uav_zspeed_dir = os.path.join(base_dir, "scenario23_dev", "unit2", "z_speed")
output_dir_cam = r"D:\Feng_Feng\dataset\uav_gps_beam_process\uav_cam_coordinate"
output_dir_bs = r"D:\Feng_Feng\dataset\uav_gps_beam_process\uav_bs_coordinate"

os.makedirs(output_dir_cam, exist_ok=True)
os.makedirs(output_dir_bs, exist_ok=True)

# ----------------------------- #
# Load Camera GPS
# ----------------------------- #
# 提取摄像头的经纬度，计算摄像头的ECEF坐标
with open(gps_cam_file, 'r') as f:
    lat_cam = float(f.readline().strip())
    lon_cam = float(f.readline().strip())
alt_cam = 0  # 摄像头高度，假设基站和摄像头等高，方便计算
ecef_cam = geodetic_to_ecef(lat_cam, lon_cam, alt_cam)


# ----------------------------- #
# Load BS GPS
# ----------------------------- #
def calculate_destination(lat1_deg, lon1_deg, distance_m, bearing_deg):
    R = 6371000  # 地球半径，单位：米
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    bearing = math.radians(bearing_deg)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_m / R) +
                     math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat1),
                             math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2))

    # 将结果转换为度
    lat2_deg = math.degrees(lat2)
    lon2_deg = math.degrees(lon2)
    return lat2_deg, lon2_deg


#bearings = [0, 45, 90, 135, 180, 225, 270, 315]
bearings = [315]
#distance_cam_bss = [100, 90, 80, 70, 60, 50, 40, 30]
distance_cam_bss = [100]
alt_bs = 0
# 用于收集仰角（el_bs）、方位角（az_bs）或距离（dist_bs）的值
az_list = []
el_list = []
dist_list = []
v_list = []

for i, bearing in enumerate(bearings):
    distance_cam_bs = distance_cam_bss[i]

    # 计算摄像头相对于基站的相对位置向量ecef
    lat_bs, lon_bs = calculate_destination(lat_cam, lon_cam, distance_cam_bs, bearing)
    ecef_bs = geodetic_to_ecef(lat_bs, lon_bs, alt_bs)

    # ----------------------------- #
    # Process UAV Files
    # ----------------------------- #

    step = 0
    for filename in os.listdir(uav_gps_dir):
        step = step + 1
        if filename.startswith("gps_location_") and filename.endswith(".txt"):
            # print("ready:",step)
            index = filename.split("_")[-1].split(".")[0]
            gps_path = os.path.join(uav_gps_dir, f"gps_location_{index}.txt")
            height_path = os.path.join(uav_height_dir, f"height_{index}.txt")
            xspeed_path = os.path.join(uav_xspeed_dir, f"x_speed_{index}.txt")
            yspeed_path = os.path.join(uav_yspeed_dir, f"y_speed_{index}.txt")
            zspeed_path = os.path.join(uav_zspeed_dir, f"z_speed_{index}.txt")
            output_path1 = os.path.join(output_dir_bs + f'_bearing{bearing}_dis{distance_cam_bs}',
                                        f"coordinate_{index}.txt")
            output_path2 = os.path.join(output_dir_cam, f"coordinate_{index}.txt")

            # try:
            with (open(gps_path, 'r') as f_gps,
                  open(height_path, 'r') as f_alt,
                  open(xspeed_path, 'r') as x_v,
                  open(yspeed_path, 'r') as y_v,
                  open(zspeed_path, 'r') as z_v):
                lat_uav = float(f_gps.readline().strip())  # 经纬度
                lon_uav = float(f_gps.readline().strip())
                alt_uav = float(f_alt.readline().strip())  # 高度
                x_speed = float(x_v.readline().strip())  # 速度
                y_speed = float(y_v.readline().strip())
                z_speed = float(z_v.readline().strip())

                ecef_uav = geodetic_to_ecef(lat_uav, lon_uav, alt_uav)

                # 计算相对位置向量：后者为观察点，前者的相对位置
                # 转换为 ENU 坐标
                # 计算仰角、方位角和距离并保存
                if i == 0:
                    delta_ecef_cam = ecef_uav - ecef_cam
                    enu_cam = ecef_to_enu(delta_ecef_cam, lat_cam, lon_cam)
                    az_cam, el_cam, dist_cam, v_radial_cam = calculate_aer(enu_cam, x_speed, y_speed, z_speed)
                    with open(output_path2, 'w') as fout:
                        fout.write(f"{az_cam:.6f} {el_cam:.6f} {dist_cam:.6f} {v_radial_cam:.6f}\n")

                delta_ecef_bs = ecef_uav - ecef_bs
                enu_bs = ecef_to_enu(delta_ecef_bs, lat_bs, lon_bs)
                az_bs, el_bs, dist_bs, v_radial_bs = calculate_aer(enu_bs, x_speed, y_speed, z_speed)
                with open(output_path1, 'w') as fout:
                    fout.write(f"{az_bs:.6f} {el_bs:.6f} {dist_bs:.6f} {v_radial_cam:.6f}\n")
                # 记录用于统计的数据
                az_list.append(az_bs)
                el_list.append(el_bs)
                dist_list.append(dist_bs)
                v_list.append(v_radial_bs)
                if step  == 1:
                    # 画相对位置图，把bs的xy坐标作为原点0，0
                    delta_ecef_cam = ecef_cam - ecef_bs
                    delta_ecef_uav = ecef_uav - ecef_bs

                    enu_cam = ecef_to_enu(delta_ecef_cam, lat_bs, lon_bs)
                    enu_uav = ecef_to_enu(delta_ecef_uav, lat_bs, lon_bs)

                    # ----------------------------- #
                    # 绘制二维图
                    # ----------------------------- #

                    plt.figure(figsize=(8, 6))
                    plt.scatter(0, 0, color='blue', label='BS')  # 基站作为原点
                    plt.scatter(enu_cam[0], enu_cam[1], color='red', label='CAM')
                    plt.scatter(enu_uav[0], enu_uav[1], color='green', label='UAV')

                    # 添加标签
                    plt.text(0, 0, 'BS', fontsize=9, ha='right')
                    plt.text(enu_cam[0], enu_cam[1], 'CAM', fontsize=9, ha='right')
                    plt.text(enu_uav[0], enu_uav[1], 'UAV', fontsize=9, ha='right')

                    # 设置图形属性
                    #plt.title('相对于基站的相对位置图（单位：米）')
                    plt.xlabel('东向（米）')
                    plt.ylabel('北向（米）')
                    # plt.xlim(-5,60)
                    # plt.ylim(-15,25)
                    plt.grid(True)
                    plt.axis('equal')  # 保持比例一致
                    plt.legend()
                    plt.show()


                    def cartesian_to_polar(x, y):
                        r = np.sqrt(x ** 2 + y ** 2)
                        theta = np.arctan2(y, x)  # 弧度
                        return theta, r


                    plt.figure(figsize=(8, 6))
                    ax = plt.subplot(111, polar=True)

                    # BS 作为极点 (r=0)
                    theta_bs, r_bs = 0, 0
                    theta_cam, r_cam = cartesian_to_polar(enu_cam[0], enu_cam[1])
                    theta_uav, r_uav = cartesian_to_polar(enu_uav[0], enu_uav[1])

                    # 画点
                    ax.scatter(theta_bs, r_bs, color='blue', label='BS')
                    ax.scatter(theta_cam, r_cam, color='red', label='CAM')
                    ax.scatter(theta_uav, r_uav, color='green', label='UAV')

                    # 添加标签
                    ax.annotate("BS", (theta_bs, r_bs), ha='right')
                    ax.annotate("CAM", (theta_cam, r_cam), ha='right')
                    ax.annotate("UAV", (theta_uav, r_uav), ha='right')

                    # 设置图属性
                    #ax.set_title('极坐标视角下的相对位置图（单位：米）', va='bottom')
                    ax.legend(loc='upper right')
                    plt.show()
    # ---------- 可视化：绘制直方图 ---------- #
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.hist(az_list, bins=30, color='skyblue', edgecolor='black')
    plt.title('Azimuth Distribution')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Count')

    plt.subplot(1, 4, 2)
    plt.hist(el_list, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Elevation Distribution')
    plt.xlabel('Elevation (degrees)')

    plt.subplot(1, 4, 3)
    plt.hist(dist_list, bins=30, color='salmon', edgecolor='black')
    plt.title('Distance Distribution')
    plt.xlabel('Distance (meters)')

    plt.subplot(1, 4, 4)
    plt.hist(v_list, bins=30, color='salmon', edgecolor='black')
    plt.title('V Distribution')
    plt.xlabel('V (m/s)')

    plt.tight_layout()
    plt.show()
    # except FileNotFoundError:
    #    print(f"跳过缺失的文件: {filename} 或其对应的 height 文件")

    print("✅ 所有坐标已计算完成,", i)
