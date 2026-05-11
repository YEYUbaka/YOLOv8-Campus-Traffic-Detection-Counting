"""
违规区域检测模块
支持多个多边形区域的违规检测
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np
import cv2

from .class_profile import DEFAULT_TRAFFIC_CLASS_PROFILE, TrafficClassProfile


@dataclass
class Zone:
    """违规区域"""
    zone_id: int
    name: str
    points: List[Tuple[int, int]]  # 多边形顶点
    zone_type: str  # 'no_parking', 'no_entry', 'restricted', 'custom'
    enabled: bool = True
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR颜色
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'zone_id': self.zone_id,
            'name': self.name,
            'points': self.points,
            'zone_type': self.zone_type,
            'enabled': self.enabled,
            'color': self.color
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Zone':
        """从字典创建"""
        return cls(
            zone_id=data['zone_id'],
            name=data['name'],
            points=[tuple(p) for p in data['points']],
            zone_type=data['zone_type'],
            enabled=data.get('enabled', True),
            color=tuple(data.get('color', (0, 0, 255)))
        )


@dataclass
class ZoneViolation:
    """违规事件"""
    track_id: int
    class_name: str
    zone_id: int
    zone_name: str
    zone_type: str
    position: Tuple[float, float]


class ZoneDetector:
    """
    违规区域检测器
    支持多个多边形区域，检测目标是否进入违规区域
    """
    
    def __init__(self, pixels_per_meter: float = 90.0):
        """
        初始化违规区域检测器
        
        Args:
            pixels_per_meter: 每米对应的像素数
        """
        self.pixels_per_meter = pixels_per_meter
        
        # 区域列表
        self.zones: Dict[int, Zone] = {}
        self.next_zone_id = 1
        
        # 已记录的违规（避免重复报警）
        self.recorded_violations: Set[Tuple[int, int]] = set()  # (track_id, zone_id)
        
        self.display_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)
        
        # 区域类型名称
        self.zone_type_names = {
            'no_parking': '禁停区',
            'no_entry': '禁止驶入',
            'restricted': '专用车道',
            'custom': '自定义区域'
        }
        
        # 区域默认颜色（BGR）
        self.zone_colors = {
            'no_parking': (0, 0, 255),      # 红色
            'no_entry': (0, 100, 255),      # 橙色
            'restricted': (255, 100, 0),    # 蓝色
            'custom': (255, 0, 255)         # 紫色
        }

    def set_class_profile(self, profile: TrafficClassProfile):
        """同步当前模型的类别展示名。"""
        self.display_names = dict(profile.display_names)
        
    def add_zone(
        self, 
        points: List[Tuple[int, int]], 
        name: str = None,
        zone_type: str = 'no_parking'
    ) -> Zone:
        """
        添加违规区域
        
        Args:
            points: 多边形顶点列表
            name: 区域名称
            zone_type: 区域类型
            
        Returns:
            创建的区域对象
        """
        if name is None:
            name = f"{self.zone_type_names.get(zone_type, '区域')} {self.next_zone_id}"
            
        color = self.zone_colors.get(zone_type, (0, 0, 255))
        
        zone = Zone(
            zone_id=self.next_zone_id,
            name=name,
            points=points,
            zone_type=zone_type,
            color=color
        )
        
        self.zones[self.next_zone_id] = zone
        self.next_zone_id += 1
        
        return zone
        
    def remove_zone(self, zone_id: int) -> bool:
        """
        移除区域
        
        Args:
            zone_id: 区域ID
            
        Returns:
            是否成功移除
        """
        if zone_id in self.zones:
            del self.zones[zone_id]
            return True
        return False
        
    def get_zone(self, zone_id: int) -> Optional[Zone]:
        """获取区域"""
        return self.zones.get(zone_id)
        
    def get_all_zones(self) -> List[Zone]:
        """获取所有区域"""
        return list(self.zones.values())
        
    def clear_zones(self):
        """清除所有区域"""
        self.zones.clear()
        self.next_zone_id = 1
        
    def enable_zone(self, zone_id: int, enabled: bool = True):
        """启用/禁用区域"""
        if zone_id in self.zones:
            self.zones[zone_id].enabled = enabled
            
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """
        判断点是否在多边形内
        
        Args:
            point: 点坐标
            polygon: 多边形顶点列表
            
        Returns:
            是否在多边形内
        """
        # 使用OpenCV的pointPolygonTest
        pts = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pts, point, False)
        return result >= 0
        
    def check_violations(self, tracks: List) -> List[ZoneViolation]:
        """
        检查违规
        
        Args:
            tracks: 跟踪目标列表
            
        Returns:
            违规事件列表
        """
        violations = []
        
        for track in tracks:
            if not hasattr(track, 'track_id') or not hasattr(track, 'center'):
                continue
                
            track_id = track.track_id
            center = track.center
            class_id = track.class_id if hasattr(track, 'class_id') else 0
            class_name = self.display_names.get(class_id, '未知')
            
            # 检查每个区域
            for zone in self.zones.values():
                if not zone.enabled:
                    continue
                    
                # 检查是否在区域内
                if self._point_in_polygon(center, zone.points):
                    violation_key = (track_id, zone.zone_id)
                    
                    # 创建违规事件
                    violation = ZoneViolation(
                        track_id=track_id,
                        class_name=class_name,
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        zone_type=zone.zone_type,
                        position=center
                    )
                    
                    violations.append(violation)
                    self.recorded_violations.add(violation_key)
                    
        # 清理已消失的跟踪记录
        current_ids = {t.track_id for t in tracks if hasattr(t, 'track_id')}
        to_remove = {(tid, zid) for tid, zid in self.recorded_violations if tid not in current_ids}
        self.recorded_violations -= to_remove
        
        return violations
        
    def get_zone_containing_point(self, point: Tuple[int, int]) -> Optional[Zone]:
        """
        获取包含指定点的区域（用于点击选择区域）
        
        Args:
            point: 点坐标
            
        Returns:
            包含该点的区域，如果没有则返回None
        """
        for zone in self.zones.values():
            if self._point_in_polygon(point, zone.points):
                return zone
        return None
        
    def draw_zones(
        self,
        frame: np.ndarray,
        active: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        在帧上绘制区域
        
        Args:
            frame: 视频帧
            active: 是否绘制活动状态（违规时高亮）
            
        Returns:
            绘制后的帧
        """
        result = frame
        overlay = result.copy()
        has_fill = False

        for zone in self.zones.values():
            if not zone.enabled:
                continue
                
            pts = np.array(zone.points, dtype=np.int32)
            
            cv2.fillPoly(overlay, [pts], zone.color)
            has_fill = True
            
            # 绘制边框
            cv2.polylines(result, [pts], True, zone.color, 2)
            
            # 绘制区域名称
            if show_labels and zone.points:
                # 计算中心点
                center_x = sum(p[0] for p in zone.points) // len(zone.points)
                center_y = sum(p[1] for p in zone.points) // len(zone.points)
                
                cv2.putText(
                    result, zone.name,
                    (center_x - 50, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

        if has_fill:
            cv2.addWeighted(overlay, 0.14, result, 0.86, 0, result)

        return result
        
    def save_zones(self, filepath: str):
        """
        保存区域配置到文件
        
        Args:
            filepath: 文件路径
        """
        import json
        
        data = {
            'zones': [zone.to_dict() for zone in self.zones.values()],
            'next_zone_id': self.next_zone_id
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_zones(self, filepath: str):
        """
        从文件加载区域配置
        
        Args:
            filepath: 文件路径
        """
        import json
        import os
        
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.zones.clear()
        
        for zone_data in data.get('zones', []):
            zone = Zone.from_dict(zone_data)
            self.zones[zone.zone_id] = zone
            
        self.next_zone_id = data.get('next_zone_id', len(self.zones) + 1)
        
    def reset(self):
        """重置检测器"""
        self.recorded_violations.clear()
