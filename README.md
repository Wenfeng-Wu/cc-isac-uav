# A Camera-Cooperative ISAC Framework for Multimodal Non-Cooperative UAV Sensing

This repository contains the code and plotting scripts for the paper:

**A Camera-Cooperative ISAC Framework for Multimodal Non-Cooperative UAVs Sensing**  

Abstract: The detection of non-cooperative unmanned aerial vehicles (UAVs) presents significant challenges for Integrated Sensing and Communication (ISAC) systems due to the inherent limitations of single-modal perception and the competition for shared communication and sensing resources. To address these challenges, this paper proposes a novel Camera-Cooperative ISAC (CC-ISAC) framework that employs multimodal sensing to enable efficient UAV beam steering and tracking. The proposed framework employs cameras for coarse-grained airspace monitoring and utilizes ISAC for fine-grained, high-precision sensing, forming a complementary perception loop that enhances both sensing accuracy and resource efficiency. Within this frame work, two key modules are developed: (1) a Vision-to-Echo Data Alignment (V2EDA) model that aligns visual and echo domain features through cross-attention mechanisms, and (2) a Multimodal Fusion-Based Estimation (MMFE) model that inte grates historical multimodal data with current observations for robust state estimation. Extensive evaluations conducted on the DeepSense6G dataset demonstrate that the proposed framework achieves an average reduction of 71% in beam steering over head and 1.69–11.15% in tracking overhead while maintaining high angular estimation accuracy. The CC-ISAC framework effectively mitigates resource contention between sensing and communication, enabling reliable UAV surveillance while freeing substantial system resources for additional communication tasks, thereby representing a practical advancement in ISAC system design.

Paper: https://arxiv.org/abs/2605.22090

The project implements two main modules:

- **V2EDA**: a vision-to-echo data alignment model for estimating UAV angle parameters from camera observations.
- **MMFE**: a multimodal fusion-based estimation model for robust UAV tracking with visual and echo sensing inputs.

## Directory Structure

```text
uav_vision_rf_sensing/
|-- Calibration_nets/
|   |-- vision_net_est_ab_light.py           # V2EDA main model
|   |-- vision_net_est_ab_*.py               # V2EDA ablation models
|   `-- weights/                             # trained V2EDA weights
|-- data_process/
|   |-- set_dataset_uavonly.py               # single-frame V2EDA dataset loader
|   |-- set_dataset_time.py                  # time-sequence MMFE dataset loader
|   `-- set_dataset_timeAdd.py               # time-offset dataset loader
|-- FusionPredict_net/
|   |-- fuse_net_VisionFirstCome_pred_*.py   # MMFE models
|   |-- fuseNo_net_VisionFirstCome_pred_*.py # MMFE without-fusion ablations
|   |-- Signal_net_EchoOnly_pred_*.py        # Echo-only baselines
|   |-- test_and_save_*.py                   # evaluation and result export
|   `-- weights_P6/                          # trained MMFE/Echo-only weights
`-- simulate_data/
    |-- data_P6/                             # saved evaluation results used by plots
    `-- plot/                                # paper figure/table reproduction scripts
```

## Dateset, Data and Weights

Dateset source: https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23

The plotting scripts read precomputed results from:

```text
simulate_data/data_P6/
```

Training and evaluation scripts expect trained weights in:

```text
Calibration_nets/weights/
FusionPredict_net/weights_P6/
```

For a clean open-source release, large datasets and model checkpoints should be distributed separately, for example through a release page, cloud storage, Zenodo, or Hugging Face. Keep the same relative directory layout after downloading.

## Environment

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Recommended working directory for reproduction:

```bash
cd uav_vision_rf_sensing
```

## Reproducing Paper Figures and Tables

Run the following commands from `uav_vision_rf_sensing/`.

| Paper result | Script | Output |
|---|---|---|
| Fig. 4(a), Fig. 4(b) | `python simulate_data/plot/plot_v-Ts.py` | `paper-fig4a.png`, `paper-fig4b.png` |
| Fig. 7(a), Fig. 7(b) | `python simulate_data/plot/plot_angle_error_in_V2EDA.py` | `paper-fig7a.png`, `paper-fig7b.png` |
| Fig. 8 angle-error boxplots and CDFs | `python simulate_data/plot/plot_MMFE_angle_error.py` | `paper-fig8_2.png`, `paper-fig8_3.png`, `paper-fig8_4.png`, `paper-fig8_5.png` |
| Fig. 9 fault-tolerance RMSE | `python simulate_data/plot/plot_fault_tolerance_MMFE.py` | `paper-fig9.png` |
| Fig. 10(a) beam-selection performance | `python simulate_data/plot/plot_perf_beam_selection.py` | `paper_fig10a.png` |
| Fig. 11 end-to-end communication overhead | `python simulate_data/plot/plot_e2e.py` | `paper_fig11.png` |
| Table 4 model complexity | `python simulate_data/plot/table4_model_param.py` | printed table |
| Table 5 tracking overhead | `python simulate_data/plot/table5_cul_Overhead.py` and `python simulate_data/plot/table5_cul_Overhead_KF.py` | printed top-k / overhead statistics |


## Citation

If this repository is useful for your research, please cite the paper:

```@ARTICLE{11570999,
  author={Wu, Wenfeng and Xiang, Luping and Yang, Kun},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={A Camera-Cooperative ISAC Framework for Multimodal Non-Cooperative UAVs Sensing}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
  keywords={Beams;Modeling;Integrated sensing and communication;Autonomous aerial vehicles;Tracking;Timing;Visual systems;Cameras;Visualization;Signal detection;Integrated Sensing and Communication (ISAC);Camera-cooperative ISAC;Non-cooperative UAV detection;Multimodal fusion;Beam steering and tracking},
  doi={10.1109/JSTSP.2026.3705654}}
```

## Contact

For questions, contact: wenfengwu@smail.nju.edu.cn
