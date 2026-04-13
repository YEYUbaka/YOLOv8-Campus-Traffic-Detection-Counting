"""
核心功能模块
包含目标跟踪、计数、速度检测、碰撞预警、违规区域检测
"""

from .class_profile import TrafficClassProfile, build_traffic_class_profile
from .tracker import ObjectTracker
from .counter import ClassCounter, LineCounter, VehicleSessionCounter
from .speed_estimator import SpeedEstimator
from .collision_warner import CollisionWarner
from .zone_detector import ZoneDetector

__all__ = [
    'TrafficClassProfile',
    'build_traffic_class_profile',
    'ObjectTracker',
    'ClassCounter',
    'LineCounter',
    'VehicleSessionCounter',
    'SpeedEstimator',
    'CollisionWarner',
    'ZoneDetector'
]
