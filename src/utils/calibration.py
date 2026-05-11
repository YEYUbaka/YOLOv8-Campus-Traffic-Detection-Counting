"""
标定管理模块
用于管理像素-实际距离的标定
"""

from typing import Tuple, Optional
import json
import os


class CalibrationManager:
    """
    标定管理器
    用于设置和管理像素与实际距离的转换关系
    """
    
    def __init__(self):
        """初始化标定管理器"""
        # 默认标定参数
        self.pixels_per_meter: float = 90.0
        self.reference_distance: float = 5.0  # 参考距离（米）
        self.reference_pixels: float = 50.0   # 参考像素数
        
        # 标定点
        self.calibration_points = []
        
        # 标定文件路径
        self.config_file = None
        
    def set_calibration(self, pixels_per_meter: float):
        """
        设置标定参数
        
        Args:
            pixels_per_meter: 每米对应的像素数
        """
        self.pixels_per_meter = pixels_per_meter
        
    def set_reference(self, pixel_distance: float, real_distance: float):
        """
        通过参考距离设置标定
        
        Args:
            pixel_distance: 像素距离
            real_distance: 实际距离（米）
        """
        self.reference_pixels = pixel_distance
        self.reference_distance = real_distance
        
        if real_distance > 0:
            self.pixels_per_meter = pixel_distance / real_distance
            
    def pixel_to_meter(self, pixels: float) -> float:
        """
        像素距离转换为米
        
        Args:
            pixels: 像素距离
            
        Returns:
            米
        """
        return pixels / self.pixels_per_meter
        
    def meter_to_pixel(self, meters: float) -> float:
        """
        米转换为像素距离
        
        Args:
            meters: 距离（米）
            
        Returns:
            像素距离
        """
        return meters * self.pixels_per_meter
        
    def add_calibration_point(self, point: Tuple[int, int]):
        """
        添加标定点
        
        Args:
            point: 点坐标
        """
        self.calibration_points.append(point)
        
        # 当有两个点时，自动计算标定
        if len(self.calibration_points) == 2:
            self._calculate_from_points()
            
    def _calculate_from_points(self):
        """从两个标定点计算标定参数"""
        if len(self.calibration_points) < 2:
            return
            
        p1, p2 = self.calibration_points[0], self.calibration_points[1]
        pixel_distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
        
        # 假设两点间距离为5米（可以修改）
        self.reference_pixels = pixel_distance
        self.reference_distance = 5.0
        self.pixels_per_meter = pixel_distance / 5.0
        
    def clear_calibration_points(self):
        """清除标定点"""
        self.calibration_points.clear()
        
    def get_calibration_info(self) -> dict:
        """
        获取标定信息
        
        Returns:
            标定信息字典
        """
        return {
            'pixels_per_meter': self.pixels_per_meter,
            'reference_distance': self.reference_distance,
            'reference_pixels': self.reference_pixels
        }
        
    def save_calibration(self, filepath: str = None):
        """
        保存标定参数
        
        Args:
            filepath: 文件路径（可选）
        """
        if filepath is None:
            filepath = self.config_file
            
        if filepath is None:
            return
            
        data = {
            'pixels_per_meter': self.pixels_per_meter,
            'reference_distance': self.reference_distance,
            'reference_pixels': self.reference_pixels
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def load_calibration(self, filepath: str):
        """
        加载标定参数
        
        Args:
            filepath: 文件路径
        """
        if not os.path.exists(filepath):
            return
            
        self.config_file = filepath
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.pixels_per_meter = data.get('pixels_per_meter', 90.0)
        self.reference_distance = data.get('reference_distance', 5.0)
        self.reference_pixels = data.get('reference_pixels', 50.0)
        
    def reset(self):
        """重置标定"""
        self.pixels_per_meter = 90.0
        self.reference_distance = 5.0
        self.reference_pixels = 50.0
        self.calibration_points.clear()