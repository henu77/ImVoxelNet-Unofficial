# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

## 创新点
1. **端到端的多视角优化**：首次将多视角RGB图像的3D物体检测任务定义为端到端的优化问题，支持任意数量输入（单目或多视角），且在训练和推理中均可灵活处理不同数量的视图。
2. **通用全卷积架构**：提出了一种全卷积3D检测框架（ImVoxelNet），通过将2D图像特征投影到3D体素空间，结合3D卷积网络提取特征，并复用点云检测器的头部结构，无需额外修改。
3. **跨场景通用性**：通过领域特定的检测头（室内/室外）实现统一的架构，在室内外场景（如KITTI、ScanNet）中均取得最优性能，成为首个通用型RGB-based 3D检测方法。

## 方法

1. **数据预处理**：
   - **特征提取**：使用预训练的2D卷积网络（如ResNet-50）提取多尺度特征，并通过FPN融合。
   - **体素投影**：将2D特征按相机位姿投影到3D体素空间，通过平均聚合多视角特征，构建3D体素表示。
2. **3D特征提取**：
   - **编码器-解码器结构**：针对室内场景设计轻量化的3D卷积网络，降低计算复杂度；室外场景则将体素压缩到BEV平面，使用2D卷积处理。
3. **检测头设计**：
   - **室外头**：基于BEV平面，采用2D锚框回归3D边界框（位置、尺寸、角度）。
   - **室内头**：扩展FCOS到3D，通过多尺度3D卷积预测边界框，引入旋转3D IoU损失。
   - **额外任务头**：联合估计相机位姿和房间布局（仅用于部分室内数据集）。

![image-20250312195144750](https://raw.githubusercontent.com/henu77/typoryPic/main/2025/image-20250312195144750.png)

## 实验结果

### 指标

- 在 `KITTI` 数据集上的结果

![image-20250312195349843](https://raw.githubusercontent.com/henu77/typoryPic/main/2025/image-20250312195349843.png)

- 在 `SUN RGB-D` 数据集上的结果

![image-20250312195450152](https://raw.githubusercontent.com/henu77/typoryPic/main/2025/image-20250312195450152.png)

- 在 `ScanNet` 数据集上的结果

![image-20250312195512125](https://raw.githubusercontent.com/henu77/typoryPic/main/2025/image-20250312195512125.png)

### 可视化

![image-20250312195612339](https://raw.githubusercontent.com/henu77/typoryPic/main/2025/image-20250312195612339.png)



![image-20250312195240782](https://raw.githubusercontent.com/henu77/typoryPic/main/2025/image-20250312195240782.png)

## 声明
- 本项目不是原创，是基于[ImVoxelNet](https://github.com/SamsungLabs/imvoxelnet）进行的修改，主要是为了适配更高的版本的 pytorch 和 mmcv 。
- 采用 [OpenMMLab](https://openmmlab.com/) 的 [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 进行实现，仅能用于预训练模型的推理。

## 环境配置

### 硬件信息

```shell
nvcc -V
```
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_03:03:05_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
```

```shell
nvidia-smi
```

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.16                 Driver Version: 572.16         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   46C    P3             N/A /  120W |    3805MiB /   8188MiB |     11%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
### 依赖库
- python 3.8.20
- torch 2.0.0+cu118
- torchvision 0.15.1+cu118
- mmcv 2.0.0
- mmengine 0.10.6
- mmdet 3.3.0

### 创建环境
```shell
# 创建虚拟环境
conda create -n imvoxelnet python=3.8.20
# 激活虚拟环境
conda activate imvoxelnet

# 安装CUDA 11.1版本的torch和torchvision
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://mirrors.aliyun.com/pytorch-wheels/cu118

# 安装mmcv、mmengine、mmdet
pip install -U opemmim
mim install mmcv==2.0.0
mim install mmengine==0.10.6
mim install mmdet==3.3.0

# 本地安装mmdetection3d
pip install -e .

# 安装PyQT5
pip install PyQt5 pygrabber==0.1
# 安装Gradio
pip install gradio==4.44.1
```

### 可视化