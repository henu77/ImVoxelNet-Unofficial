import sys
import cv2
import numpy as np
import torch
import skimage.io
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QGroupBox, QGridLayout,QLineEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from pygrabber.dshow_graph import FilterGraph

from mmcv.transforms import Compose
from mmdet3d.structures import DepthInstance3DBoxes
import matplotlib.pyplot as plt
from mmengine.runner import load_checkpoint
from mmdet3d.models.detectors import ImVoxelNet_v2
from mmengine.registry import TRANSFORMS
from mmdet3d.datasets import MultiViewPipeline

from checkpoints.imvoxelnet_total_sunrgbd_fast import (
    model as model_cfg,
    test_cfg,
    train_cfg,
    class_names,
    test_pipeline
)
from util import draw_corners, get_extrinsics

TRANSFORMS.register_module(name='MultiViewPipeline', module=MultiViewPipeline)
label_colors = np.multiply([
    plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
], 255).astype(np.uint8).tolist()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model()
        self.transform = Compose(test_pipeline)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_camera_running = False

    def initUI(self):
        self.setWindowTitle('3D检测演示')
        main_layout = QVBoxLayout()

        # 图像显示区域
        img_layout = QHBoxLayout()
        self.input_label = QLabel('摄像头输入')
        self.pred_label = QLabel('预测结果')
        self.input_label.setFixedSize(640, 480)
        self.pred_label.setFixedSize(640, 480)
        img_layout.addWidget(self.input_label)
        img_layout.addWidget(self.pred_label)
        main_layout.addLayout(img_layout)

        # 摄像头控制区域
        control_group = QGroupBox("摄像头设置")
        control_layout = QVBoxLayout()

        # 摄像头选择
        cam_layout = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.scan_cameras()
        cam_layout.addWidget(QLabel("选择摄像头:"))
        cam_layout.addWidget(self.cam_combo)
        control_layout.addLayout(cam_layout)

        # 内参输入
        intrinsic_layout = QGridLayout()
        self.fx_edit = QLineEdit("529.5")
        self.fy_edit = QLineEdit("529.5")
        self.cx_edit = QLineEdit("365")
        self.cy_edit = QLineEdit("265")
        self.s_edit = QLineEdit("0")
        intrinsic_layout.addWidget(QLabel("fx:"), 0, 0)
        intrinsic_layout.addWidget(self.fx_edit, 0, 1)
        intrinsic_layout.addWidget(QLabel("fy:"), 1, 0)
        intrinsic_layout.addWidget(self.fy_edit, 1, 1)
        intrinsic_layout.addWidget(QLabel("cx:"), 2, 0)
        intrinsic_layout.addWidget(self.cx_edit, 2, 1)
        intrinsic_layout.addWidget(QLabel("cy:"), 3, 0)
        intrinsic_layout.addWidget(self.cy_edit, 3, 1)
        intrinsic_layout.addWidget(QLabel("倾斜系数 s:"), 4, 0)
        intrinsic_layout.addWidget(self.s_edit, 4, 1)
        control_layout.addLayout(intrinsic_layout)

        # 控制按钮
        self.cam_btn = QPushButton("启动摄像头")
        self.cam_btn.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.cam_btn)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        self.setLayout(main_layout)

    def scan_cameras(self):
        self.cam_combo.clear()
        graph = FilterGraph()
        devices = graph.get_input_devices()
        for index, name in enumerate(devices):
            self.cam_combo.addItem(f'Camera {index} {name}', index)

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        cam_idx = self.cam_combo.currentData()
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        self.timer.start(30)  # 约30ms更新一次
        self.cam_btn.setText("停止摄像头")
        self.is_camera_running = True

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.setText("启动摄像头")
        self.is_camera_running = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 显示原始帧
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.input_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.input_label.size(), aspectRatioMode=1))

            # 进行预测
            temp_path = "temp.jpg"
            cv2.imwrite(temp_path, frame)
            self.predict_image(temp_path)

    def load_model(self):
        model = ImVoxelNet_v2(
            backbone=model_cfg["backbone"],
            neck=model_cfg["neck"],
            neck_3d=model_cfg["neck_3d"],
            bbox_head=model_cfg["bbox_head"],
            n_voxels=model_cfg["n_voxels"],
            voxel_size=model_cfg["voxel_size"],
            head_2d=model_cfg["head_2d"],
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        load_checkpoint(model, "checkpoints/20211007_105247.pth", map_location="cpu")
        model.eval()
        return model

    def get_intrinsic_matrix(self):
        try:
            fx = float(self.fx_edit.text())
            fy = float(self.fy_edit.text())
            cx = float(self.cx_edit.text())
            cy = float(self.cy_edit.text())
            s = float(self.s_edit.text())
        except ValueError:
            fx, fy, cx, cy, s = 529.5, 529.5, 365.0, 265.0, 0.0

        return np.array([
            [fx,  s, cx, 0],
            [0,  fy, cy, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])

    def predict_image(self, image_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        inputs = {
            "img_info": [{"filename": image_path}],
            "img_prefix": [""],
            "lidar2img": {
                "extrinsic": [np.eye(4)],
                "intrinsic": [self.get_intrinsic_matrix()],
                "origin": [np.array([0, 3, -1])]
            },
        }

        data = self.transform(inputs)
        data['data_samples'].lidar2img['intrinsic'] = data['data_samples'].lidar2img['intrinsic'][0]
        data['data_samples'].lidar2img['extrinsic'] = data['data_samples'].lidar2img['extrinsic'][0]
        data['data_samples'].lidar2img['origin'] = data['data_samples'].lidar2img['origin'][0]

        img_metas = [{
            "filename": image_path,
            "lidar2img": data['data_samples'].lidar2img,
            "img_shape": data['data_samples'].img_shape,
            "ori_shape": data['data_samples'].ori_shape,
            "box_type_3d": DepthInstance3DBoxes,
        }]

        with torch.no_grad():
            results = self.model.simple_test(data['inputs']['img'].to(device), img_metas)
        
        # 如果没有检测到物体，直接返回, 则显示原始图像
        if len(results[0]['bboxes_3d']) == 0:
            q_img = QImage(image_path)
            self.pred_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.pred_label.size(), aspectRatioMode=1))
        else:
        # 结果可视化
            print('检测到物体')
            img = self.visualize_results(image_path, results[0], img_metas[0])
            q_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.pred_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.pred_label.size(), aspectRatioMode=1))

    def visualize_results(self, image_path, result, img_meta):
        img = skimage.io.imread(image_path)
        intrinsic = img_meta['lidar2img']['intrinsic'][:3, :3]
        extrinsic = get_extrinsics(result['angles'])
        projection = intrinsic @ extrinsic[:3]

        # NMS处理
        keep_indices = self.apply_nms(result['bboxes_3d'], result['scores_3d'], result['labels_3d'], dist_threshold=10.0)
        if len(keep_indices) == 0:
            return img
        bboxes_3d = [result['bboxes_3d'][i] for i in keep_indices]
        labels_3d = [result['labels_3d'][i] for i in keep_indices]

        # 打印检测到的所有类别
        print(f"Class: {', '.join([class_names[label] for label in labels_3d])}")
        # 绘制结果
        for bbox, label in zip(bboxes_3d, labels_3d):
            corners = bbox.corners.cpu().numpy()[0]
            draw_corners(img, corners, label_colors[label], projection, class_names[label])
        return img
    def apply_nms(self, bboxes, scores, label, dist_threshold=2.0):
        # 新增NMS处理逻辑
        detections = []
        for i in range(len(bboxes)):
            corners = bboxes[i].corners.cpu().numpy()[0]
            center = corners.mean(axis=0)  # 计算3D边界框中心点
            detections.append({
                'score': scores[i],
                'label': label[i],
                'center': center,
                'index': i
            })
        
        # 按类别分组处理
        from collections import defaultdict
        detections_by_label = defaultdict(list)
        for det in detections:
            detections_by_label[det['label']].append(det)
        
        keep_indices = []
        dist_threshold = 20.0  # 距离阈值，可根据场景调整
        
        for label, group in detections_by_label.items():
            # 按分数降序排序
            sorted_group = sorted(group, key=lambda x: x['score'], reverse=True)
            
            current_keep = []
            for current_det in sorted_group:
                # 检查与已保留框的距离
                keep = True
                for kept_det in current_keep:
                    distance = np.linalg.norm(current_det['center'] - kept_det['center'])
                    if distance < dist_threshold:
                        keep = False
                        break
                if keep:
                    current_keep.append(current_det)
            
            keep_indices.extend([det['index'] for det in current_keep])
        return keep_indices

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())