import os
import cv2
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from mmcv.transforms import Compose
from mmdet3d.structures import DepthInstance3DBoxes
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

class Model3DDetector:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化后自动加载默认模型
        self.load_default_model()

    def load_default_model(self):
        """尝试加载默认模型"""
        try:
            checkpoint_path = "checkpoints/imvoxelnet_total_sunrgbd_fast.pth"
            if os.path.exists(checkpoint_path):
                self.load_model_with_config(default_model_cfg, checkpoint_path)
                return "默认模型加载成功"
            else:
                return "默认模型检查点不存在，请手动加载模型"
        except Exception as e:
            return f"无法加载默认模型: {str(e)}"

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
            print(f"加载模型失败: {str(e)}")
            return False

    def get_intrinsic_matrix(self, fx, fy, cx, cy, s):
        """构建相机内参矩阵"""
        return np.array([
            [fx,  s, cx, 0],
            [0,  fy, cy, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])

    def predict(self, image, fx=529.5, fy=529.5, cx=365.0, cy=265.0, s=0.0):
        """处理图像并进行预测"""
        if self.model is None or self.transform is None:
            return image, "请先加载模型"
            
        try:
            # 保存临时图像文件
            temp_path = "temp_gradio.jpg"
            if isinstance(image, np.ndarray):
                cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                image.save(temp_path)
                image = np.array(image)
            
            intrinsic_matrix = self.get_intrinsic_matrix(fx, fy, cx, cy, s)
            
            # 准备预测输入
            inputs = {
                "img_info": [{"filename": temp_path}],
                "img_prefix": [""],
                "lidar2img": {
                    "extrinsic": [np.eye(4)],
                    "intrinsic": [intrinsic_matrix],
                    "origin": [np.array([0, 3, -1])]
                },
            }

            # 运行预处理流程
            data = self.transform(inputs)
            data['data_samples'].lidar2img['intrinsic'] = data['data_samples'].lidar2img['intrinsic'][0]
            data['data_samples'].lidar2img['extrinsic'] = data['data_samples'].lidar2img['extrinsic'][0]
            data['data_samples'].lidar2img['origin'] = data['data_samples'].lidar2img['origin'][0]

            img_metas = [{
                "filename": temp_path,
                "lidar2img": data['data_samples'].lidar2img,
                "img_shape": data['data_samples'].img_shape,
                "ori_shape": data['data_samples'].ori_shape,
                "box_type_3d": DepthInstance3DBoxes,
            }]

            # 模型预测
            self.model.to(self.device)
            with torch.no_grad():
                results = self.model.simple_test(data['inputs']['img'].to(self.device), img_metas)
            
            result = results[0]
            img_meta = img_metas[0]
            
            # 可视化结果
            vis_result = self.visualize_results(image, result, img_meta)
            
            # 生成结果信息
            detection_info = f"检测到 {len(result['bboxes_3d'])} 个物体"
            if len(result['bboxes_3d']) > 0:
                detection_info += "\n\n类别数量统计:"
                from collections import Counter
                label_counts = Counter(result['labels_3d'].tolist())
                for label_id, count in label_counts.items():
                    detection_info += f"\n- {class_names[label_id]}: {count}个"
            
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
                
            return vis_result, detection_info
            
        except Exception as e:
            import traceback
            error_msg = f"预测过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return image, error_msg

    def visualize_results(self, img, result, img_meta):
        """可视化预测结果"""
        # 确保图像是RGB格式的numpy数组
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # 如果没有检测到物体，直接返回原图
        if len(result['bboxes_3d']) == 0:
            return img
            
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

    def load_custom_model(self, model_path):
        """加载自定义模型"""
        if not model_path:
            return "请选择模型文件"
        
        try:
            success = self.load_model_with_config(default_model_cfg, model_path)
            if success:
                return f"模型加载成功: {os.path.basename(model_path)}"
            else:
                return "模型加载失败"
        except Exception as e:
            return f"加载模型时出错: {str(e)}"

# 创建Gradio界面
def create_gradio_interface():
    detector = Model3DDetector()
    
    with gr.Blocks(title="3D目标检测系统") as demo:
        gr.Markdown("# 3D目标检测系统")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="输入图像")
                
                with gr.Accordion("相机内参设置", open=False):
                    with gr.Row():
                        fx = gr.Number(value=529.5, label="fx")
                        fy = gr.Number(value=529.5, label="fy")
                    with gr.Row():
                        cx = gr.Number(value=365.0, label="cx")
                        cy = gr.Number(value=265.0, label="cy")
                    with gr.Row():
                        s = gr.Number(value=0.0, label="s")
                
                with gr.Accordion("模型设置", open=False):
                    model_path = gr.File(label="加载自定义模型")
                    load_model_btn = gr.Button("加载模型")
                    model_status = gr.Textbox(label="模型状态", value=detector.load_default_model())
                
                detect_btn = gr.Button("执行检测", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(label="检测结果")
                detection_info = gr.Textbox(label="检测信息")
        
        # 事件处理
        detect_btn.click(
            fn=detector.predict, 
            inputs=[input_image, fx, fy, cx, cy, s], 
            outputs=[output_image, detection_info]
        )
        
        load_model_btn.click(
            fn=detector.load_custom_model,
            inputs=[model_path],
            outputs=[model_status]
        )
        
    return demo

if __name__ == "__main__":
    # 启动Gradio应用
    demo = create_gradio_interface()
    demo.launch(share=True)  # share=True允许生成一个公开可访问的链接