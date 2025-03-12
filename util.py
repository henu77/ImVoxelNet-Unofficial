import numpy as np
import torch
import cv2

def draw_corners(img, corners, color, projection, label_name):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T
        z_mask = corners_2d_3[:, 2] > 0
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int32)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            if z_mask[i] and z_mask[j]:
                img = cv2.line(
                    img=img,
                    pt1=tuple(corners_2d[i]),
                    pt2=tuple(corners_2d[j]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        # 设置文本和背景
        text = label_name
        org = tuple(corners_2d[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 0)  # 黑色文字
        
        # 计算文本框大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # 绘制背景矩形
        padding = 2  # 文本周围的内边距
        cv2.rectangle(
            img,
            (org[0] - padding, org[1] - text_height - padding),
            (org[0] + text_width + padding, org[1] + padding),
            color,  # 使用与边框相同的颜色
            -1  # 填充矩形
        )
        
        # 绘制文本
        img = cv2.putText(
            img=img,
            text=text,
            org=org,
            fontFace=font,
            fontScale=font_scale,
            color=text_color,  # 黑色文字
            thickness=font_thickness,
            lineType=cv2.LINE_AA
        )
        
        return img
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic.cpu().detach().numpy()