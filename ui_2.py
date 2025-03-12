import sys
import os
import cv2
import numpy as np
import torch
import skimage.io
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QComboBox, QGroupBox, QGridLayout, QLineEdit,
                            QMessageBox, QProgressBar)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from pygrabber.dshow_graph import FilterGraph

from mmcv.transforms import Compose
from mmdet3d.structures import DepthInstance3DBoxes
import matplotlib.pyplot as plt
from mmengine.runner import load_checkpoint
from mmdet3d.models.detectors import ImVoxelNet_v2
from mmengine.registry import TRANSFORMS
from mmdet3d.datasets import MultiViewPipeline

from checkpoints.imvoxelnet_total_sunrgbd_fast import (
    model as default_model_cfg,
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

class ErrorManager:
    """统一的错误管理类"""
    @staticmethod
    def show_error(parent, title, message):
        QMessageBox.critical(parent, title, message)
    
    @staticmethod
    def show_warning(parent, title, message):
        QMessageBox.warning(parent, title, message)
    
    @staticmethod
    def show_info(parent, title, message):
        QMessageBox.information(parent, title, message)

class PredictionThread(QThread):
    """预测线程，与主线程分离"""
    prediction_complete = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, model, transform, image_path=None, 
                 intrinsic_matrix=None, frame=None, temp_path=None):
        super().__init__()
        self.model = model
        self.transform = transform
        self.image_path = image_path
        self.intrinsic_matrix = intrinsic_matrix
        self.frame = frame
        self.temp_path = temp_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def run(self):
        try:
            # 如果传入的是帧而不是图像路径
            if self.frame is not None and self.temp_path:
                # 如果是BGR格式的帧，需要特别注意保存格式与加载格式一致
                cv2.imwrite(self.temp_path, self.frame)  # OpenCV保存的是BGR格式
                self.image_path = self.temp_path
            
            if not self.image_path or not os.path.exists(self.image_path):
                self.error_occurred.emit("预测错误", "图像路径不存在")
                return
                
            # 预测逻辑
            self.model.to(self.device)
            
            inputs = {
                "img_info": [{"filename": self.image_path}],
                "img_prefix": [""],
                "lidar2img": {
                    "extrinsic": [np.eye(4)],
                    "intrinsic": [self.intrinsic_matrix],
                    "origin": [np.array([0, 3, -1])]
                },
            }

            data = self.transform(inputs)
            data['data_samples'].lidar2img['intrinsic'] = data['data_samples'].lidar2img['intrinsic'][0]
            data['data_samples'].lidar2img['extrinsic'] = data['data_samples'].lidar2img['extrinsic'][0]
            data['data_samples'].lidar2img['origin'] = data['data_samples'].lidar2img['origin'][0]

            img_metas = [{
                "filename": self.image_path,
                "lidar2img": data['data_samples'].lidar2img,
                "img_shape": data['data_samples'].img_shape,
                "ori_shape": data['data_samples'].ori_shape,
                "box_type_3d": DepthInstance3DBoxes,
            }]

            with torch.no_grad():
                results = self.model.simple_test(data['inputs']['img'].to(self.device), img_metas)
            
            self.prediction_complete.emit(results[0], img_metas[0])
        
        except Exception as e:
            self.error_occurred.emit("预测错误", f"预测过程中发生错误: {str(e)}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_camera_running = False
        self.is_video_running = False
        self.video_path = None
        self.prediction_thread = None
        self.temp_path = "temp.jpg"
        
        # 初始化UI
        self.initUI()
        
        # 尝试加载默认模型
        self.load_default_model()
        
    def load_default_model(self):
        """尝试加载默认模型"""
        try:
            checkpoint_path = "checkpoints/imvoxelnet_total_sunrgbd_fast.pth"
            if os.path.exists(checkpoint_path):
                self.load_model_with_config(default_model_cfg, checkpoint_path)
                self.status_bar.setText("默认模型加载成功")
            else:
                self.status_bar.setText("默认模型检查点不存在，请手动加载模型")
        except Exception as e:
            ErrorManager.show_warning(self, "模型加载警告", f"无法加载默认模型: {str(e)}")
            self.status_bar.setText("默认模型加载失败")

    def initUI(self):
        """初始化UI界面"""
        self.setWindowTitle('3D检测系统')
        self.setGeometry(100, 100, 1300, 800)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 图像显示区域
        display_layout = QVBoxLayout()
        
        # 添加标题标签
        image_titles_layout = QHBoxLayout()
        self.input_title = QLabel('输入图像')
        self.pred_title = QLabel('预测结果')
        self.input_title.setAlignment(Qt.AlignCenter)
        self.pred_title.setAlignment(Qt.AlignCenter)
        image_titles_layout.addWidget(self.input_title)
        image_titles_layout.addWidget(self.pred_title)
        display_layout.addLayout(image_titles_layout)
        
        # 图像标签
        image_layout = QHBoxLayout()
        self.input_label = QLabel()
        self.pred_label = QLabel()
        self.input_label.setFixedSize(640, 480)
        self.pred_label.setFixedSize(640, 480)
        self.input_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.input_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        self.pred_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        image_layout.addWidget(self.input_label)
        image_layout.addWidget(self.pred_label)
        display_layout.addLayout(image_layout)
        
        main_layout.addLayout(display_layout, 2)
        
        # 控制面板区域
        control_layout = QHBoxLayout()
        
        # 相机内参设置
        intrinsic_group = QGroupBox("相机内参设置")
        # 或者指定最大高度
        intrinsic_group.setMaximumHeight(180)

        intrinsic_layout = QGridLayout()
        intrinsic_layout.setHorizontalSpacing(2)
        intrinsic_layout.setVerticalSpacing(2)  # 进一步减小垂直间距
        intrinsic_layout.setContentsMargins(5, 5, 5, 5)

        # 内参输入框 - 进一步减小宽度
        self.fx_edit = QLineEdit("529.5")
        self.fx_edit.setFixedWidth(60)
        self.fy_edit = QLineEdit("529.5")
        self.fy_edit.setFixedWidth(60)
        self.cx_edit = QLineEdit("365")
        self.cx_edit.setFixedWidth(60)
        self.cy_edit = QLineEdit("265")
        self.cy_edit.setFixedWidth(60)
        self.s_edit = QLineEdit("0")
        self.s_edit.setFixedWidth(60)

        # 使用更紧凑的布局 - 将fx/fy和cx/cy放在同一行
        fx_label = QLabel("fx:")
        fx_label.setFixedWidth(20)
        fy_label = QLabel("fy:")
        fy_label.setFixedWidth(20)
        cx_label = QLabel("cx:")
        cx_label.setFixedWidth(20)
        cy_label = QLabel("cy:")
        cy_label.setFixedWidth(20)
        s_label = QLabel("s:")
        s_label.setFixedWidth(20)

        # 两列布局，更加紧凑
        intrinsic_layout.addWidget(fx_label, 0, 0)
        intrinsic_layout.addWidget(self.fx_edit, 0, 1)
        intrinsic_layout.addWidget(cx_label, 0, 2)
        intrinsic_layout.addWidget(self.cx_edit, 0, 3)
        intrinsic_layout.addWidget(fy_label, 1, 0)
        intrinsic_layout.addWidget(self.fy_edit, 1, 1)
        intrinsic_layout.addWidget(cy_label, 1, 2)
        intrinsic_layout.addWidget(self.cy_edit, 1, 3)
        intrinsic_layout.addWidget(s_label, 2, 0)
        intrinsic_layout.addWidget(self.s_edit, 2, 1)

        intrinsic_group.setLayout(intrinsic_layout)

        
        # 摄像头设置
        camera_group = QGroupBox("摄像头设置")
        camera_layout = QVBoxLayout()
        
        # 摄像头选择
        cam_select_layout = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.scan_cameras()
        cam_select_layout.addWidget(QLabel("选择摄像头:"))
        cam_select_layout.addWidget(self.cam_combo)
        camera_layout.addLayout(cam_select_layout)
        
        # 摄像头控制按钮
        cam_btn_layout = QHBoxLayout()
        self.cam_btn = QPushButton("启动摄像头")
        self.cam_btn.clicked.connect(self.toggle_camera)
        cam_btn_layout.addWidget(self.cam_btn)
        camera_layout.addLayout(cam_btn_layout)
        
        camera_group.setLayout(camera_layout)
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        # 模型加载按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        
        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("上传图片")
        self.load_image_btn.clicked.connect(self.browse_image)
        
        self.load_video_btn = QPushButton("上传视频")
        self.load_video_btn.clicked.connect(self.browse_video)
        
        file_layout.addWidget(self.load_image_btn)
        file_layout.addWidget(self.load_video_btn)
        
        file_group.setLayout(file_layout)
        
        # 操作控制区域
        operation_group = QGroupBox("操作控制")
        operation_layout = QVBoxLayout()
        
        self.stop_btn = QPushButton("停止预测")
        self.stop_btn.clicked.connect(self.stop_all)
        self.stop_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("清空显示")
        self.clear_btn.clicked.connect(self.clear_display)
        
        self.exit_btn = QPushButton("退出程序")
        self.exit_btn.clicked.connect(self.close)
        
        operation_layout.addWidget(self.stop_btn)
        operation_layout.addWidget(self.clear_btn)
        operation_layout.addWidget(self.exit_btn)
        
        operation_group.setLayout(operation_layout)
        
        # 添加所有控制组到控制布局
        control_layout.addWidget(intrinsic_group,1)
        control_layout.addWidget(camera_group,1)
        control_layout.addWidget(model_group,1)
        control_layout.addWidget(file_group,1)
        control_layout.addWidget(operation_group,1)
        
        # 添加控制布局到主布局
        main_layout.addLayout(control_layout, 1)
        
        # 状态栏
        self.status_bar = QLabel("就绪")
        main_layout.addWidget(self.status_bar)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.setLayout(main_layout)

    def scan_cameras(self):
        """扫描并加载可用的摄像头"""
        self.cam_combo.clear()
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for index, name in enumerate(devices):
                self.cam_combo.addItem(f'Camera {index} {name}', index)
        except Exception as e:
            ErrorManager.show_warning(self, "摄像头扫描警告", f"扫描摄像头失败: {str(e)}")
    
    def get_intrinsic_matrix(self):
        """从UI中获取相机内参矩阵"""
        try:
            fx = float(self.fx_edit.text())
            fy = float(self.fy_edit.text())
            cx = float(self.cx_edit.text())
            cy = float(self.cy_edit.text())
            s = float(self.s_edit.text())
        except ValueError:
            ErrorManager.show_warning(self, "参数警告", "内参格式错误，将使用默认值")
            fx, fy, cx, cy, s = 529.5, 529.5, 365.0, 265.0, 0.0

        return np.array([
            [fx,  s, cx, 0],
            [0,  fy, cy, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])
    
    def load_model_with_config(self, model_cfg, checkpoint_path):
        """加载模型和配置"""
        try:
            # 创建模型
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
            
            # 加载检查点
            load_checkpoint(model, checkpoint_path, map_location="cpu")
            model.eval()
            
            # 设置模型和转换器
            self.model = model
            self.transform = Compose(test_pipeline)
            
            return True
        
        except Exception as e:
            ErrorManager.show_error(self, "模型加载错误", f"加载模型失败: {str(e)}")
            return False
    
    def browse_model(self):
        """浏览并加载模型检查点"""
        checkpoint_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型检查点", "", "Model Checkpoint (*.pth)"
        )
        
        if checkpoint_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            success = self.load_model_with_config(default_model_cfg, checkpoint_path)
            
            self.progress_bar.setValue(100)
            if success:
                self.status_bar.setText(f"模型加载成功: {os.path.basename(checkpoint_path)}")
                ErrorManager.show_info(self, "模型加载", "模型加载成功")
            
            # 延迟隐藏进度条
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
    
    def browse_image(self):
        """选择图片并进行预测"""
        if not self.check_model():
            return
            
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not image_path:
            return
            
        # 显示原始图像
        self.display_image(image_path, self.input_label)
        self.status_bar.setText(f"已加载图片: {os.path.basename(image_path)}")
        
        # 开始预测
        self.start_prediction(image_path=image_path)
    
    def browse_video(self):
        """选择视频进行处理"""
        if not self.check_model():
            return
            
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if not video_path:
            return
            
        self.video_path = video_path
        self.start_video(video_path)
    
    def start_video(self, video_path):
        """开始处理视频"""
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            ErrorManager.show_error(self, "视频错误", "无法打开视频文件")
            return
            
        self.is_video_running = True
        self.is_camera_running = False
        self.timer.start(30)  # 约30fps
        
        self.stop_btn.setEnabled(True)
        self.status_bar.setText(f"正在播放视频: {os.path.basename(video_path)}")
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.check_model():
            return
            
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """启动摄像头"""
        # 关闭可能正在运行的视频
        if self.is_video_running:
            self.stop_video()
        
        # 获取选中的摄像头索引
        cam_idx = self.cam_combo.currentData()
        if cam_idx is None:
            ErrorManager.show_error(self, "摄像头错误", "请先选择一个摄像头")
            return
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            ErrorManager.show_error(self, "摄像头错误", "无法打开摄像头")
            return
        
        self.is_camera_running = True
        self.is_video_running = False
        self.timer.start(30)
        
        self.cam_btn.setText("停止摄像头")
        self.stop_btn.setEnabled(True)
        self.status_bar.setText("摄像头已启动，正在进行实时预测")
    
    def stop_camera(self):
        """停止摄像头"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.is_camera_running = False
        self.cam_btn.setText("启动摄像头")
        self.stop_btn.setEnabled(False)
        self.status_bar.setText("摄像头已停止")
    
    def stop_video(self):
        """停止视频播放"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.is_video_running = False
        self.stop_btn.setEnabled(False)
        self.status_bar.setText("视频播放已停止")
    
    def update_frame(self):
        """更新视频帧"""
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            # 视频播放完毕
            if self.is_video_running:
                self.stop_video()
                ErrorManager.show_info(self, "视频播放", "视频播放完毕")
            elif self.is_camera_running:
                self.stop_camera()
                ErrorManager.show_error(self, "摄像头错误", "无法获取摄像头画面")
            return
        
        # 转换为RGB以便显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 显示原始帧
        self.display_frame(frame_rgb, self.input_label)
        
        # 进行预测 - 传入原始BGR帧，预测线程内部会正确处理颜色空间
        if not (self.prediction_thread and self.prediction_thread.isRunning()):
            self.start_prediction(frame=frame)
    
    def display_image(self, image_path, label):
        """在指定标签上显示图像"""
        if not os.path.exists(image_path):
            return
            
        # 读取图像并适应标签大小
        img = cv2.imread(image_path)
        if img is None:
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_frame(img_rgb, label)
    
    def display_frame(self, frame, label):
        """在标签上显示帧"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # 创建QImage并显示在标签上
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(
            label.size(), aspectRatioMode=Qt.KeepAspectRatio))
    
    def start_prediction(self, image_path=None, frame=None):
        """开始预测流程"""
        # 检查上一个预测是否仍在运行
        if self.prediction_thread and self.prediction_thread.isRunning():
            # 不要强制终止，而是跳过这一帧
            return
        
        # 为每一帧创建唯一的临时文件名，避免冲突
        if frame is not None:
            import time
            temp_path = f"temp_{time.time()}.jpg"
        else:
            temp_path = self.temp_path
        
        # 创建并启动预测线程
        self.prediction_thread = PredictionThread(
            model=self.model,
            transform=self.transform,
            image_path=image_path,
            intrinsic_matrix=self.get_intrinsic_matrix(),
            frame=frame,
            temp_path=temp_path
        )
        
        # 连接信号
        self.prediction_thread.prediction_complete.connect(self.handle_prediction_result)
        self.prediction_thread.error_occurred.connect(self.handle_prediction_error)
        self.prediction_thread.finished.connect(lambda: self.cleanup_temp(temp_path))
        
        # 启动线程
        self.prediction_thread.start()
    def cleanup_temp(self, temp_path):
        """清理临时文件"""
        if temp_path != self.temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    def handle_prediction_result(self, result, img_meta):
        """处理预测结果"""
        # 加载原始图像
        image_path = img_meta["filename"]
        
        # 使用OpenCV读取图像并转换为RGB (更一致的方法)
        img = cv2.imread(image_path)
        if img is None:
            ErrorManager.show_error(self, "图像错误", f"无法加载图像: {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 确保转换为RGB
        
        # 如果没有检测到物体，直接显示原图
        if len(result['bboxes_3d']) == 0:
            self.display_frame(img, self.pred_label)
            self.status_bar.setText("没有检测到物体")
            return
        
        # 可视化结果
        try:
            vis_img = self.visualize_results(img, result, img_meta)
            self.display_frame(vis_img, self.pred_label)
            self.status_bar.setText(f"检测到 {len(result['bboxes_3d'])} 个物体")
        except Exception as e:
            ErrorManager.show_error(self, "可视化错误", f"可视化结果时出错: {str(e)}")
    
    def handle_prediction_error(self, title, message):
        """处理预测过程中的错误"""
        ErrorManager.show_error(self, title, message)
        self.status_bar.setText("预测失败")
    
    def visualize_results(self, img, result, img_meta):
        """可视化预测结果"""
        # 确保使用的是RGB格式图像
        intrinsic = img_meta['lidar2img']['intrinsic'][:3, :3]
        extrinsic = get_extrinsics(result['angles'])
        projection = intrinsic @ extrinsic[:3]

        # NMS处理
        keep_indices = self.apply_nms(result['bboxes_3d'], result['scores_3d'], result['labels_3d'])
        if len(keep_indices) == 0:
            return img
            
        bboxes_3d = [result['bboxes_3d'][i] for i in keep_indices]
        labels_3d = [result['labels_3d'][i] for i in keep_indices]

        # 绘制结果
        img_copy = img.copy()  # 创建副本以避免修改原始图像
        for bbox, label in zip(bboxes_3d, labels_3d):
            corners = bbox.corners.cpu().numpy()[0]
            # 确保颜色是RGB格式
            color = label_colors[label]
            draw_corners(img_copy, corners, color, projection, class_names[label])
        
        return img_copy
    
    def apply_nms(self, bboxes, scores, labels, dist_threshold=20.0):
        """应用非极大值抑制"""
        # 如果没有检测到物体，返回空列表
        if len(bboxes) == 0:
            return []
            
        # 构建检测结果列表
        detections = []
        for i in range(len(bboxes)):
            corners = bboxes[i].corners.cpu().numpy()[0]
            center = corners.mean(axis=0)  # 计算3D边界框中心点
            detections.append({
                'score': scores[i],
                'label': labels[i],
                'center': center,
                'index': i
            })
        
        # 按类别分组处理
        from collections import defaultdict
        detections_by_label = defaultdict(list)
        for det in detections:
            detections_by_label[det['label']].append(det)
        
        keep_indices = []
        
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
    
    def stop_all(self):
        """停止所有正在进行的操作"""
        if self.is_camera_running:
            self.stop_camera()
        
        if self.is_video_running:
            self.stop_video()
            
        if self.prediction_thread and self.prediction_thread.isRunning():
            self.prediction_thread.terminate()
            self.prediction_thread.wait()
            
        self.stop_btn.setEnabled(False)
        self.status_bar.setText("已停止所有操作")
    
    def clear_display(self):
        """清空显示的图像"""
        # 创建空白图像
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 240
        self.display_frame(blank, self.input_label)
        self.display_frame(blank, self.pred_label)
        self.status_bar.setText("已清空显示")
    
    def check_model(self):
        """检查模型是否已加载"""
        if self.model is None or self.transform is None:
            ErrorManager.show_error(self, "模型错误", "请先加载模型")
            return False
        return True
    
    def closeEvent(self, event):
        """关闭窗口时的事件"""
        self.stop_all()
        # 清理所有temp_开头的临时文件
        for file in os.listdir('.'):
            if file.startswith('temp_') and file.endswith('.jpg'):
                try:
                    os.remove(file)
                except:
                    pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())