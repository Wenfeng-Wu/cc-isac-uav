# 参数定义示例
T_frame = 100  # ms，一帧的总时长
R_max = 100    # Mbps，单个地面用户在理想条件下（占用整个T_comm）可达到的最大速率

# 假设参数
#N = 20   # 全扫描波束数
M = 10   # 纯回波平均扫描波束数
L = 3    # 多模态平均扫描波束数
t_beam = 1 # ms，测试一个波束的时长

# 为不同感知方法标定其所需的感知时隙
#T_sense_FS = N * t_beam  # 20 ms
T_sense_EO = M * t_beam  # 10 ms
T_sense_MM = L * t_beam  # 3 ms

# 根据步骤2的结果，计算每种方法剩余的可用于通信的时间。
#T_comm_FS = T_frame - T_sense_FS  # 100 - 20 = 80 ms
T_comm_EO = T_frame - T_sense_EO  # 100 - 10 = 90 ms
T_comm_MM = T_frame - T_sense_MM  # 100 - 3  = 97 ms

# 建模地面用户通信与性能计算
K = 5  # 地面用户数量

# 每个用户分到的通信时间
#T_per_user_FS = T_comm_FS / K  # 80 / 5 = 16 ms
T_per_user_EO = T_comm_EO / K  # 90 / 5 = 18 ms
T_per_user_MM = T_comm_MM / K  # 97 / 5 = 19.4 ms

# 系统总吞吐量 (正比于 T_comm)
# 模型A：固定用户数，计算总和速率提升
#Total_Throughput_FS = R_max * (T_comm_FS / T_frame)  # 100 * (80/100) = 80 Mbps
Total_Throughput_EO = R_max * (T_comm_EO / T_frame)  # 100 * (90/100) = 90 Mbps
Total_Throughput_MM = R_max * (T_comm_MM / T_frame)  # 100 * (97/100) = 97 Mbps

# 模型B：固定单用户需求，计算可服务的用户数
T_req = 10  # ms，每个用户需要的最低通信时长

# 可服务的用户数
#Num_Users_FS = T_comm_FS // T_req  # 80 // 10 = 8 个用户
Num_Users_EO = T_comm_EO // T_req  # 90 // 10 = 9 个用户
Num_Users_MM = T_comm_MM // T_req  # 97 // 10 = 9 个用户 (注意：这里和EO一样，因为97/10=9.7，向下取整为9)

# 当T_req=15ms时，差异更明显：
# Num_Users_FS = 80 // 15 = 5
# Num_Users_EO = 90 // 15 = 6
# Num_Users_MM = 97 // 15 = 6