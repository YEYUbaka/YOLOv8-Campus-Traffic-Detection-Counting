"""
速度检测模块
通过目标跟踪计算实时速度
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .class_profile import DEFAULT_TRAFFIC_CLASS_PROFILE, TrafficClassProfile


@dataclass
class SpeedInfo:
    """速度信息"""
    track_id: int
    class_name: str
    speed: float  # km/h
    position: Tuple[float, float]


class SpeedEstimator:
    """
    速度估计器
    通过跟踪目标的位移和帧率计算速度
    """
    
    def __init__(
        self,
        pixels_per_meter: float = 10.0,
        fps: float = 30.0,
        smoothing_factor: float = 0.3
    ):
        """
        初始化速度估计器
        
        Args:
            pixels_per_meter: 每米对应的像素数（标定参数）
            fps: 视频帧率
            smoothing_factor: 速度平滑因子（0-1，越大越平滑）
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.smoothing_factor = smoothing_factor
        
        # 速度历史（用于平滑）
        self.speed_history: Dict[int, List[float]] = {}
        
        self.display_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)
        self.vehicle_class_ids = set(DEFAULT_TRAFFIC_CLASS_PROFILE.vehicle_class_ids)

    def set_class_profile(self, profile: TrafficClassProfile):
        """同步当前模型的类别展示名。"""
        self.display_names = dict(profile.display_names)
        self.vehicle_class_ids = set(profile.vehicle_class_ids)
        
    def set_calibration(self, pixels_per_meter: float):
        """
        设置标定参数
        
        Args:
            pixels_per_meter: 每米对应的像素数
        """
        self.pixels_per_meter = pixels_per_meter
        
    def set_fps(self, fps: float):
        """
        设置帧率
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        
    def estimate_speed(
        self, 
        track_id: int, 
        class_id: int,
        history: List[Tuple[float, float]]
    ) -> float:
        """
        估计单个目标的速度
        
        Args:
            track_id: 跟踪ID
            class_id: 类别ID
            history: 位置历史 [(x, y), ...]
            
        Returns:
            速度（km/h），保留一位小数
        """
        if len(history) < 2:
            return 0.0
            
        # 计算最近几帧的平均位移
        window_size = min(5, len(history) - 1)
        total_distance = 0.0
        
        for i in range(-window_size, 0):
            p1 = history[i]
            p2 = history[i + 1]
            
            # 像素距离
            pixel_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            total_distance += pixel_distance
            
        # 平均每帧像素位移
        avg_pixel_displacement = total_distance / window_size
        
        # 转换为米/帧
        meters_per_frame = avg_pixel_displacement / self.pixels_per_meter
        
        # 转换为米/秒
        meters_per_second = meters_per_frame * self.fps
        
        # 转换为千米/小时
        km_per_hour = meters_per_second * 3.6
        
        # 平滑处理
        if track_id in self.speed_history:
            # 指数移动平均
            prev_speed = self.speed_history[track_id][-1] if self.speed_history[track_id] else km_per_hour
            smoothed_speed = self.smoothing_factor * prev_speed + (1 - self.smoothing_factor) * km_per_hour
        else:
            smoothed_speed = km_per_hour
            
        # 更新历史
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []
        self.speed_history[track_id].append(smoothed_speed)
        
        # 只保留最近10帧
        if len(self.speed_history[track_id]) > 10:
            self.speed_history[track_id].pop(0)
            
        # 保留一位小数
        return round(smoothed_speed, 1)
        
    def estimate_all(self, tracks: List) -> List[SpeedInfo]:
        """
        估计所有目标的速度
        
        Args:
            tracks: 跟踪目标列表
            
        Returns:
            速度信息列表
        """
        results = []
        
        for track in tracks:
            if not hasattr(track, 'track_id'):
                continue
                
            track_id = track.track_id
            class_id = track.class_id if hasattr(track, 'class_id') else 0
            if class_id not in self.vehicle_class_ids:
                continue
            class_name = self.display_names.get(class_id, '未知')
            history = track.history if hasattr(track, 'history') else []
            center = track.center if hasattr(track, 'center') else (0, 0)
            
            speed = self.estimate_speed(track_id, class_id, history)
            
            results.append(SpeedInfo(
                track_id=track_id,
                class_name=class_name,
                speed=speed,
                position=center
            ))
            
        return results
        
    def clear_track(self, track_id: int):
        """清除指定跟踪的速度历史"""
        if track_id in self.speed_history:
            del self.speed_history[track_id]
            
    def reset(self):
        """重置所有速度历史"""
        self.speed_history.clear()
        
    def get_speed_range(self, tracks: List) -> Tuple[float, float]:
        """
        获取速度范围
        
        Args:
            tracks: 跟踪目标列表
            
        Returns:
            (最小速度, 最大速度)
        """
        speeds = []
        for track in tracks:
            if hasattr(track, 'track_id'):
                track_id = track.track_id
                if track_id in self.speed_history and self.speed_history[track_id]:
                    speeds.append(self.speed_history[track_id][-1])
                    
        if speeds:
            return round(min(speeds), 1), round(max(speeds), 1)
        return 0.0, 0.0
