
##======================test_and_save_a_all.py============================##
##================================方位角===================================##
##=====通过调整set_dataset_time.py中的测试集文件夹，指定不同SNR的回波感知数据=====##
##========================================================================##

SNR=0,无数据丢失情况:
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3806
Echo-Only vs 真实值: 0.5236
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0.pth

SNR=-1,无数据丢失情况:
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4210
Echo-Only vs 真实值: 0.5823
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr-1.pth

SNR=-2,无数据丢失情况:
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4580
Echo-Only vs 真实值: 0.6467
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr-2.pth

SNR=-3,无数据丢失情况：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.5070
Echo-Only vs 真实值: 0.7303
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr-3.pth

# 下面是Multi-Modal w/o Fuse方法

SNR=0,无数据丢失情况:
Vision-Only vs 真实值: 0.5051
Multi-Modal w/o Fuse vs 真实值: 0.4459 *这是有效值
Echo-Only vs 真实值: 0.5236
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_a_snr0_1.pth

SNR=-1,无数据丢失情况:
Vision-Only vs 真实值: 0.5051
Multi-Modal w/o Fuse  vs 真实值: 0.5061    *这是有效值
Echo-Only vs 真实值: 0.5823
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_a_snr-1_1.pth

SNR=-2,无数据丢失情况：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.5683    *这是有效值
Echo-Only vs 真实值: 0.6467
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_a_snr-2_1.pth

#下面是KF-BASED 方法：

SNR=0,无数据丢失情况:
方位角RMSE计算结果：
KL算法预测的 vs 真实值: 0.8405
仰角RMSE计算结果：
KL算法预测的 vs 真实值: 0.9369
Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr0_1.pth

SNR=-1,无数据丢失情况:
方位角RMSE计算结果：
KL算法预测的 vs 真实值: 0.9943
仰角RMSE计算结果：
KL算法预测的 vs 真实值: 1.0568
Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr-1_1.pth


SNR=-2,无数据丢失情况：
方位角RMSE计算结果：
KL算法预测的 vs 真实值: 1.0806
仰角RMSE计算结果：
KL算法预测的 vs 真实值: 1.1561
Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr-2_1.pth

# 下面是由时间偏差的MMFE：用set_INOUT_data_timeOffset处理输出数据
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4209
Echo-Only vs 真实值: 0.5762
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset_pred_a_snr0_1.pth
echo落后偏差最大是前后的一半

方位角RMSE计算结果：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4029
Echo-Only vs 真实值: 0.5520
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset3_pred_a_snr0_1.pth
echo落后偏差最大是前后的三分之一

Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3955
Echo-Only vs 真实值: 0.5422
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset4_pred_a_snr0_1.pth
echo落后偏差最大是前后的四分之一


现在考虑偏差是回波提前了：
方位角RMSE计算结果：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3728
Echo-Only vs 真实值: 0.5179
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset-2_pred_a_snr0_1.pth


现在考虑偏差是回波提前延后都有了：
方位角RMSE计算结果：
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3952
Echo-Only vs 真实值: 0.5466
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset2-2_pred_a_snr0_1.pth
也就是回波偏差范围是【-1/fps/2, 1/fps/2】

##------通过set_INOUT_data_someError函数的Etype,指定不同丢失和填充数据方式------##
##-------------------------------echo-only--------------------------------##
##--------------------数据丢失只能Etype='echo_error_is_last_echo'-----------##
##---------------为了仿真方便，最近的echo数据取的是第一个时隙的数据---------------##
少丢失情况：

SNR=0
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3831  *这是有效值
Echo-Only vs 真实值: 0.5315  *这是有效值
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_fewLostEcho02_echo-only.pth

SNR=-2
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4563  *这是有效值
Echo-Only vs 真实值: 0.6459  *这是有效值
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr-2_fewLostEcho02_echo-only.pth

多丢失情况：
SNR=0
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3887  *这是有效值
Echo-Only vs 真实值: 0.5388 *这是有效值
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_lotLostEcho09_echo-only.pth

SNR=-2
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4574  *这是有效值
Echo-Only vs 真实值: 0.6638  *这是有效值
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr-2_lotLostEcho09_echo-only.pth


少丢失情况：
SNR=0
方位角RMSE计算结果：
KL算法预测的 vs 真实值: 0.8300
仰角RMSE计算结果：
KL算法预测的 vs 真实值: 0.9270
Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr0_fewLostEcho02.pth


多丢失情况：
SNR=0
方位角RMSE计算结果：
KL算法预测的 vs 真实值: 0.8023
仰角RMSE计算结果：
KL算法预测的 vs 真实值: 0.9231
Saved data_P6 to  ../simulate_data/data_P6/Comparison_KL_pred_ab_snr0_lotLostEcho09.pth


##------------------------------------------------------------------------##
##-----------Multi-Modal数据丢失采用Etype='echo_error_is_vision'------------##

少丢失情况：
SNR=0
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3832 *这是有效值
Echo-Only vs 真实值: 0.5230
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_fewLostEcho02_multimodal.pth

多丢失情况：
SNR=0
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.3832 *这是有效值
Echo-Only vs 真实值: 0.5230
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_a_snr0_lotLostEcho09_multimodal.pth

#下面是Multi-Modal w/o Fuse方法
少丢失情况：
SNR=0
Vision-Only vs 真实值: 0.5051
Multi-Modal noFuse vs 真实值: 0.4473  *这是有效值
Echo-Only vs 真实值: 0.5236
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_a_snr0_fewLostEcho02_multimodalNofuse.pth

多丢失情况：
SNR=0
Vision-Only vs 真实值: 0.5051
Multi-Modal vs 真实值: 0.4411 *这是有效值
Echo-Only vs 真实值: 0.5073
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_a_snr0_fewLostEcho09_multimodalNofuse.pth


用回波补充丢失的：


##======================test_and_save_b_all.py============================##
##================================仰角=====================================##
##=====通过调整set_dataset_time.py中的测试集文件夹，指定不同SNR的回波感知数据=====##
##========================================================================##

SNR=0,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7396
Echo-Only vs 真实值: 0.7648
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0.pth

SNR=-1,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7360
Echo-Only vs 真实值: 0.7711
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr-1.pth

SNR=-2,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7772
Echo-Only vs 真实值: 0.8377
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr-2.pth

SNR=-3,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.8317
Echo-Only vs 真实值: 0.9062
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr-3.pth


# 下面是Multi-Modal noFuse
SNR=0,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal noFuse vs 真实值: 0.8276  *这是有效值
Echo-Only vs 真实值: 0.7648
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_b_snr0_1.pth

SNR=-1,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.8261 *这是有效值
Echo-Only vs 真实值: 0.7711
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_b_snr-1_1.pth

SNR=-2,无数据丢失情况:
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.8598 *这是有效值
Echo-Only vs 真实值: 0.8377
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_b_snr-2_1.pth

考虑回波延后
仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7256
Echo-Only vs 真实值: 0.7663
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset3_pred_b_snr0_1.pth

仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7230
Echo-Only vs 真实值: 0.7761
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset2_pred_b_snr0_1.pth

仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7280
Echo-Only vs 真实值: 0.7636
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset4_pred_b_snr0_1.pth

考虑回波提前：
仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7702
Echo-Only vs 真实值: 0.7813
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset-2_pred_b_snr0_1.pth

现在考虑偏差是回波提前延后都有了：
仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7523
Echo-Only vs 真实值: 0.7886
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_timeOffset2-2_pred_b_snr0_1.pth


##------通过set_INOUT_data_someError函数的Etype,指定不同丢失和填充数据方式------##
##------------------------------------------------------------------------##
##----------------------Etype='echo_error_is_last_echo'-------------------##
##---------------为了仿真方便，最近的echo数据取的是第一个时隙的数据---------------##
少丢失情况：
SNR=0
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7569  *这是有效值
Echo-Only vs 真实值: 0.7896  *这是有效值
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_fewLostEcho02.pth
多丢失情况：
SNR=0
仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7918
Echo-Only vs 真实值: 0.8216  *这是有效值
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_lotLostEcho09.pth

但其实这个MMFE应该用vision补充：echo_error_is_vision：求出来的MMFE如下
仰角RMSE计算结果：
Vision-Only vs 真实值: 1.6321
Multi-Modal vs 真实值: 0.7523  *这是有效值
Echo-Only vs 真实值: 0.7797  #这个值使用vision补充的，不能算
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_pred_b_snr0_lotLostEcho09_2.pth


Vision-Only vs 真实值: 1.6321
Multi-Modal noFuse vs 真实值: 0.8422  *这是有效值
Echo-Only vs 真实值: 0.7755
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_b_snr0_lotLostEcho09.pth

Vision-Only vs 真实值: 1.6321
Multi-Modal noFuse vs 真实值: 0.8290 *这是有效值
Echo-Only vs 真实值: 0.7661
Saved data_P6 to  ../simulate_data/data_P6/Comparison_MMFE_noFuse_pred_b_snr0_fewLostEcho02.pth