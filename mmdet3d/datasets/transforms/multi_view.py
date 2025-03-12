import numpy as np
import os
from mmdet3d.registry import TRANSFORMS

from mmcv.transforms import BaseTransform, Compose, RandomFlip, LoadImageFromFile

@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images

    def transform(self, results):
        imgs = []
        extrinsics = []
        ids = np.arange(len(results['img_info']))
        replace = True if self.n_images > len(ids) else False
        ids = np.random.choice(ids, self.n_images, replace=replace)
        for i in ids.tolist():
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results['img_path'] = _results['img_info']['filename']
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        results['lidar2img']['extrinsic'] = extrinsics
        return results


@TRANSFORMS.register_module()
class RandomShiftOrigin(BaseTransform):
    def __init__(self, std):
        self.std = std

    def transform(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results


@TRANSFORMS.register_module()
class KittiSetOrigin(BaseTransform):
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def transform(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@TRANSFORMS.register_module()
class KittiRandomFlip(BaseTransform):
    def transform(self, results):
        if results['flip']:
            results['lidar2img']['intrinsic'][0, 2] = -results['lidar2img']['intrinsic'][0, 2] + \
                                                      results['ori_shape'][1]
            flip_matrix_0 = np.eye(4, dtype=np.float32)
            flip_matrix_0[0, 0] *= -1
            flip_matrix_1 = np.eye(4, dtype=np.float32)
            flip_matrix_1[1, 1] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0]
            extrinsic = flip_matrix_0 @ extrinsic @ flip_matrix_1.T
            results['lidar2img']['extrinsic'][0] = extrinsic
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 0], -center[:, 1]) - alpha
            center_flip = center
            center_flip[:, 1] *= -1
            alpha_flip = np.arctan2(center_flip[:, 0], -center_flip[:, 1]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@TRANSFORMS.register_module()
class SunRgbdSetOrigin(BaseTransform):
    def transform(self, results):
        intrinsic = results['lidar2img']['intrinsic'][:3, :3]
        extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
        projection = intrinsic @ extrinsic
        h, w, _ = results['ori_shape']
        center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
        center_2d_3 *= 3
        origin = np.linalg.inv(projection) @ center_2d_3
        results['lidar2img']['origin'] = origin
        return results


@TRANSFORMS.register_module()
class SunRgbdTotalLoadImageFromFile(LoadImageFromFile):
    def transform(self, results):
        file_name = results['img_info']['filename']
        flip = file_name.endswith('_flip.jpg')
        if flip:
            results['img_info']['filename'] = file_name.replace('_flip.jpg', '.jpg')
        results = super().__call__(results)
        if flip:
            results['img'] = results['img'][:, ::-1]
        return results


@TRANSFORMS.register_module()
class SunRgbdRandomFlip(BaseTransform):
    def transform(self, results):
        if results['flip']:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
            results['lidar2img']['extrinsic'][0][:3, :3] = flip_matrix @ extrinsic @ flip_matrix.T
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 1], center[:, 0]) - alpha
            center_flip = center @ flip_matrix
            alpha_flip = np.arctan2(center_flip[:, 1], center_flip[:, 0]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results
