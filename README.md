# ES-MEOT

If you use the code, please cite our paper:

```text
@ARTICLE{10753434,
  author={Deng, Jiayin and Hu, Zhiqun and Lu, Zhaoming and Wen, Xiangming},
  journal={IEEE Sensors Journal}, 
  title={3D Multiple Extended Object Tracking by Fusing Roadside Radar and Camera Sensors}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Radar tracking;Trajectory;Three-dimensional displays;Visualization;Sensors;Radar;Probability density function;Radio frequency;Cameras;Symbols;Multiple extended object tracking;random finite sets;radar and camera fusion;roadside perception},
  doi={10.1109/JSEN.2024.3493952}
}
```

## Pipline

| Description |       Step             | Input              | Output                | Method              |
| ---- | ---- | ---- | ---- | ---- |
| **Preprocess**    | 1.1 Keypoints Detection | 1.1 Video **(output.avi)**| 1.1 Pixel Keypoints **(output-keypoints.npy)** | 1.1 Yolov8 |
|              | 1.2 Track Keypoints | 1.2 Pixel Keypoints (output-keypoints.npy) | 1.2 Tracked Keypoints **(output-keypoints-ByteTrack.npy)** | 1.2 ByteTrack |
|              | 1.3 Calculate Plane Points | 1.3 Tracked Keypoints **(output-keypoints-ByteTrack.npy)**, Radar Track **(radarTrack.npy)** | 1.3 Plane Points **(output-PlanePoint.npy)** | 1.3 FusionCalib |
|              | 1.4 Train MDN | 1.4 Plane Points **(output-PlanePoint.npy)** | 1.4 MDN (mdn.pt) | 1.4 MDN |
| **Realtime**    | 2.1 Keypoints Detection | 2.1 Same as 1.1 | 2.1 Same as 1.1 | 2.1 Same as 1.1 |
|                 | 2.2 Back-projected Pose and Shape | 2.2 Pixel Keypoints **(output-keypoints.npy)** | 2.2 Back-projected Pose and Shape **(pose_shape_from_kps.npy)** | 2.2 EPnP |
|                 | 2.3 MTPMB Tracker | 2.3 Radar Point Clouds **(radarPcl.npy)**, Pixel Keypoints **(output-keypoints.npy)**, Back-projected Pose and Shape **(pose_shape_from_kps.npy)**, mdn network **(mdn.pt)** | 2.3 Result **(result.npy)** | 2.3 MTPMB Tracker |


## Installation

* Clone the repository and cd to it.

```bash
git clone --recursive https://github.com/RadarCameraFusionTeam-BUPT/ES-MEOT.git
cd ES-MEOT
```

* Create and activate virtual environment using anaconda (tested on python 3.8 and 3.10).

```bash
conda create -n ES_MEOT python=3.10
conda activate ES_MEOT
```

* Install dependencies for yolov8.

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

* Install other dependencies.

```bash
pip install scikit-learn
pip install lap
pip install cython_bbox
pip install filterpy
pip install third-party/murty
```

## Prepare test data

* Download experimental data from [data link](https://data.mendeley.com/datasets/k2kggnrtxg/1), and unzip into the `data` folder.

* The data folder will be like:

```text
data
├─ output.avi
├─ config
├─ annotation.npy
├─ radarPcl.npy
└─ radarTrack.npy
```

## Usage

### 1. Visual Keypoints Detection

```bash
python keypoints_det.py data/output.avi --model assets/best.pt --render
mv output/output-keypoints.npy data
```

* !!! Note: the ``.pt`` file in ``assets`` is a small model trained on our dataset. If you want to test the code in another scene, you can replace the model using a larger one [link](https://drive.google.com/file/d/1vnJbfMzvKxIPGX49Lkmc9Tlr9XrGiv-I/view?usp=drive_link).

## 2. Visual Back-projection from Keypoints

### 2.1 Back-projection Pose and Shape

```bash
python get_pose_and_shape_from_kps.py data/output-keypoints.npy --config data/config
mv output/pose_shape_from_kps.npy data
```

### 2.2 Train MDN for Position Back-projection

* Track pixel keypoints

```bash
python track_keypoints.py data
mv output/output-keypoints-ByteTrack.npy data
```

* Calculate Plane Points using FusionCalib (radar camera fusion)

```bash
python CalPlanePoint.py data
mv output/output-PlanePoint.npy data
```

* Train MDN

```bash
python train_mdn.py data
mv output/mdn.pt data
```

### 2.3 TPMBM Tracker

```bash
python main.py data
```

* The detections are written in ``output/result.npy``.

## Show

* Save results and annotations in BEV view. Before processing, please modify the file path and parameters in save_res_anno_in_bev.py

```bash
python save_res_anno_in_bev.py
```

* Save results in BEV view. Before processing, please modify the file path and parameters in save_res_in_bev.py

```bash
python save_res_in_bev.py
```

* Save results in image video. Before processing, please modify the file path and parameters in save_res_in_video.py

```bash
python save_res_in_video.py
```
