"""
碰撞预警模块
检测车辆之间的距离，发出碰撞预警
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .class_profile import DEFAULT_TRAFFIC_CLASS_PROFILE, TrafficClassProfile


@dataclass
class CollisionWarning:
    """碰撞预警信息"""
    track_id_1: int
    track_id_2: int
    class_name_1: str
    class_name_2: str
    distance: float  # 米
    position_1: Tuple[float, float]
    position_2: Tuple[float, float]
    severity: str  # 'low', 'medium', 'high'


class CollisionWarner:
    """
    碰撞预警器
    检测目标之间的距离，当距离过近时发出预警
    """
    
    def __init__(
        self,
        pixels_per_meter: float = 10.0,
        safe_distance: float = 5.0,
        warning_distance: float = 10.0
    ):
        """
        初始化碰撞预警器
        
        Args:
            pixels_per_meter: 每米对应的像素数
            safe_distance: 安全距离（米），小于此距离发出高危预警
            warning_distance: 预警距离（米），小于此距离发出预警
        """
        self.pixels_per_meter = pixels_per_meter
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        
        # 已预警的配对（避免重复预警）
        self.warned_pairs: Set[Tuple[int, int]] = set()
        
        self.display_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)
        self.vehicle_classes = set(DEFAULT_TRAFFIC_CLASS_PROFILE.vehicle_class_ids)
        self.warning_distance_pixels = self.warning_distance * self.pixels_per_meter

    def set_class_profile(self, profile: TrafficClassProfile):
        """同步当前模型的车辆类集合和展示名。"""
        self.display_names = dict(profile.display_names)
        self.vehicle_classes = set(profile.vehicle_class_ids)
        
    def set_calibration(self, pixels_per_meter: float):
        """设置标定参数"""
        self.pixels_per_meter = pixels_per_meter
        self.warning_distance_pixels = self.warning_distance * self.pixels_per_meter
        
    def set_safe_distance(self, distance: float):
        """设置安全距离"""
        self.safe_distance = distance
        self.warning_distance = distance * 2  # 预警距离为安全距离的2倍
        self.warning_distance_pixels = self.warning_distance * self.pixels_per_meter
        
    def _pixel_to_meter(self, pixel_distance: float) -> float:
        """像素距离转换为米"""
        return pixel_distance / self.pixels_per_meter
        
    def _compute_center_distance(
        self, 
        center1: Tuple[float, float], 
        center2: Tuple[float, float]
    ) -> float:
        """计算两个中心点的像素距离"""
        return math.hypot(center1[0] - center2[0], center1[1] - center2[1])
        
    def _compute_bbox_distance(
        self, 
        bbox1: List[float], 
        bbox2: List[float]
    ) -> float:
        """
        计算两个边界框之间的最小距离
        返回像素距离
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算矩形之间的最小距离
        # 如果矩形相交，距离为0
        
        # X方向距离
        if x2_1 < x1_2:  # bbox1 在 bbox2 左边
            dx = x1_2 - x2_1
        elif x2_2 < x1_1:  # bbox1 在 bbox2 右边
            dx = x1_1 - x2_2
        else:  # X方向重叠
            dx = 0
            
        # Y方向距离
        if y2_1 < y1_2:  # bbox1 在 bbox2 上边
            dy = y1_2 - y2_1
        elif y2_2 < y1_1:  # bbox1 在 bbox2 下边
            dy = y1_1 - y2_2
        else:  # Y方向重叠
            dy = 0
            
        return math.hypot(dx, dy)

    def _bbox_x_gap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """返回两个边界框在 X 轴上的最小间隔，重叠时为 0。"""
        if bbox1[2] < bbox2[0]:
            return bbox2[0] - bbox1[2]
        if bbox2[2] < bbox1[0]:
            return bbox1[0] - bbox2[2]
        return 0.0
        
    def check_collisions(self, tracks: List) -> List[CollisionWarning]:
        """
        检查碰撞风险
        
        Args:
            tracks: 跟踪目标列表
            
        Returns:
            碰撞预警列表
        """
        warnings = []
        
        # 过滤出车辆类型的目标
        vehicle_tracks = [
            t for t in tracks 
            if hasattr(t, 'class_id') and t.class_id in self.vehicle_classes
        ]
        
        if len(vehicle_tracks) < 2:
            return warnings

        sorted_tracks = sorted(
            vehicle_tracks,
            key=lambda track: track.bbox[0] if hasattr(track, 'bbox') and track.bbox else 0.0,
        )
        pixel_gate = max(1.0, self.warning_distance_pixels)

        # 检查每对目标，先用 X 轴间隔做粗筛
        for i, track1 in enumerate(sorted_tracks):
            bbox1 = track1.bbox if hasattr(track1, 'bbox') else None
            if bbox1 is None:
                continue

            for track2 in sorted_tracks[i+1:]:
                if not hasattr(track1, 'track_id') or not hasattr(track2, 'track_id'):
                    continue
                    
                id1, id2 = track1.track_id, track2.track_id
                
                # 确保ID有序，避免重复检查
                pair = (min(id1, id2), max(id1, id2))
                
                # 获取边界框和中心点
                bbox2 = track2.bbox if hasattr(track2, 'bbox') else None
                center1 = track1.center if hasattr(track1, 'center') else None
                center2 = track2.center if hasattr(track2, 'center') else None
                
                if bbox1 is None or bbox2 is None:
                    continue

                if bbox2[0] - bbox1[2] > pixel_gate:
                    break

                if self._bbox_x_gap(bbox1, bbox2) > pixel_gate:
                    continue
                    
                # 计算边界框距离
                pixel_distance = self._compute_bbox_distance(bbox1, bbox2)
                
                # 如果边界框重叠，使用中心点距离
                if pixel_distance == 0 and center1 and center2:
                    pixel_distance = self._compute_center_distance(center1, center2)
                    
                # 转换为米
                meter_distance = self._pixel_to_meter(pixel_distance)
                
                # 判断是否需要预警
                if meter_distance < self.safe_distance:
                    severity = 'high'
                elif meter_distance < self.warning_distance:
                    severity = 'medium'
                else:
                    # 如果之前预警过，现在距离安全了，从预警集合移除
                    self.warned_pairs.discard(pair)
                    continue
                    
                # 创建预警
                class_name1 = self.display_names.get(
                    track1.class_id if hasattr(track1, 'class_id') else 0, '未知'
                )
                class_name2 = self.display_names.get(
                    track2.class_id if hasattr(track2, 'class_id') else 0, '未知'
                )
                
                warning = CollisionWarning(
                    track_id_1=id1,
                    track_id_2=id2,
                    class_name_1=class_name1,
                    class_name_2=class_name2,
                    distance=round(meter_distance, 1),
                    position_1=center1 if center1 else (0, 0),
                    position_2=center2 if center2 else (0, 0),
                    severity=severity
                )
                
                warnings.append(warning)
                self.warned_pairs.add(pair)
                
        return warnings
        
    def get_warning_count(self, warnings: List[CollisionWarning]) -> Dict[str, int]:
        """
        统计预警数量
        
        Args:
            warnings: 预警列表
            
        Returns:
            {'high': 数量, 'medium': 数量, 'total': 总数}
        """
        high_count = sum(1 for w in warnings if w.severity == 'high')
        medium_count = sum(1 for w in warnings if w.severity == 'medium')
        
        return {
            'high': high_count,
            'medium': medium_count,
            'total': len(warnings)
        }
        
    def reset(self):
        """重置预警器"""
        self.warned_pairs.clear()
