"""
目标跟踪器
使用基于 IoU + 中心点距离 + 框尺寸相似度的轻量级跟踪算法。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .class_profile import DEFAULT_TRAFFIC_CLASS_PROFILE, TrafficClassProfile


@dataclass
class Track:
    """跟踪目标"""
    track_id: int
    class_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    age: int = 0
    hits: int = 1
    history: List[Tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        self.update_center()
        
    def update_center(self):
        """更新中心点"""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
    def update(self, bbox: List[float], confidence: float):
        """更新跟踪状态"""
        self.bbox = bbox
        self.confidence = confidence
        self.update_center()
        self.history.append(self.center)
        # 只保留最近30帧的历史
        if len(self.history) > 30:
            self.history.pop(0)
        self.hits += 1
        self.age = 0
        
    def predict(self):
        """预测下一帧位置（简单预测）"""
        self.age += 1
        return self.bbox


class ObjectTracker:
    """
    基于IoU的简单目标跟踪器
    实现类似SORT的跟踪逻辑
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        size_similarity_threshold: float = 0.45,
        reconnect_window: int = 12,
    ):
        """
        初始化跟踪器
        
        Args:
            max_age: 目标消失后保留的最大帧数
            min_hits: 确认跟踪需要的最小命中次数
            iou_threshold: IoU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.size_similarity_threshold = size_similarity_threshold
        self.reconnect_window = reconnect_window
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        
        self.class_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)
        
    def set_class_names(self, class_names: Dict[int, str]):
        """设置类别名称映射"""
        self.class_names = class_names

    def set_class_profile(self, profile: TrafficClassProfile):
        """按当前模型配置类别名称。"""
        self.class_names = dict(profile.display_names)
        
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # 计算并集
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
        
    def _compute_distance(self, center1: Tuple[float, float], 
                          center2: Tuple[float, float]) -> float:
        """计算两个中心点的欧氏距离"""
        return math.hypot(center1[0] - center2[0], center1[1] - center2[1])

    def _compute_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """计算边界框中心点。"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _compute_bbox_size_similarity(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的尺寸相似度，范围为 0-1。"""
        w1 = max(1.0, box1[2] - box1[0])
        h1 = max(1.0, box1[3] - box1[1])
        w2 = max(1.0, box2[2] - box2[0])
        h2 = max(1.0, box2[3] - box2[1])

        area_ratio = min(w1 * h1, w2 * h2) / max(w1 * h1, w2 * h2)
        aspect1 = w1 / h1
        aspect2 = w2 / h2
        aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2)

        return 0.7 * area_ratio + 0.3 * aspect_ratio

    def _compute_detection_score(self, track: Track, det_bbox: List[float]) -> float:
        """
        计算检测与已有轨迹的匹配分数。
        分数越高表示越可能属于同一目标。
        """
        iou = self._compute_iou(det_bbox, track.bbox)
        det_center = self._compute_bbox_center(det_bbox)
        center_distance = self._compute_distance(track.center, det_center)
        size_similarity = self._compute_bbox_size_similarity(det_bbox, track.bbox)

        track_width = max(1.0, track.bbox[2] - track.bbox[0])
        track_height = max(1.0, track.bbox[3] - track.bbox[1])
        diag = math.hypot(track_width, track_height)
        max_distance = max(50.0, diag * (2.0 if track.age > 0 else 1.5))
        distance_score = max(0.0, 1.0 - center_distance / max_distance)

        return 0.55 * iou + 0.30 * distance_score + 0.15 * size_similarity

    def _is_reasonable_match(self, track: Track, det_bbox: List[float]) -> bool:
        """联合门控判断一个检测是否可以匹配到当前轨迹。"""
        iou = self._compute_iou(det_bbox, track.bbox)
        det_center = self._compute_bbox_center(det_bbox)
        center_distance = self._compute_distance(track.center, det_center)
        size_similarity = self._compute_bbox_size_similarity(det_bbox, track.bbox)

        track_width = max(1.0, track.bbox[2] - track.bbox[0])
        track_height = max(1.0, track.bbox[3] - track.bbox[1])
        diag = math.hypot(track_width, track_height)
        max_distance = max(50.0, diag * (2.0 if track.age > 0 else 1.5))

        if size_similarity < self.size_similarity_threshold:
            return False

        if iou >= self.iou_threshold:
            return True

        if track.age <= self.reconnect_window and center_distance <= max_distance:
            return True

        return False
        
    def update(self, detections: List[Tuple[List[float], int, float]]) -> List[Track]:
        """
        更新跟踪器
        
        Args:
            detections: 检测结果列表，每个元素为 (bbox, class_id, confidence)
                       bbox格式为 [x1, y1, x2, y2]
        
        Returns:
            当前帧确认的跟踪目标列表
        """
        self.frame_count += 1
        
        # 如果没有检测结果
        if not detections:
            # 更新所有现有跟踪的age
            tracks_to_remove = []
            for track_id, track in self.tracks.items():
                track.predict()
                if track.age > self.max_age:
                    tracks_to_remove.append(track_id)
                    
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                
            return self._get_confirmed_tracks()
        
        track_ids = list(self.tracks.keys())
        candidates = []

        for d_idx, (det_bbox, det_cls, det_conf) in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if track.class_id != det_cls:
                    continue

                if not self._is_reasonable_match(track, det_bbox):
                    continue

                score = self._compute_detection_score(track, det_bbox)
                candidates.append((score, d_idx, t_idx))

        matched_detections = set()
        matched_tracks = set()

        for _score, d_idx, t_idx in sorted(candidates, key=lambda item: item[0], reverse=True):
            if d_idx in matched_detections or t_idx in matched_tracks:
                continue

            track_id = track_ids[t_idx]
            det_bbox, det_cls, det_conf = detections[d_idx]
            self.tracks[track_id].update(det_bbox, det_conf)
            matched_detections.add(d_idx)
            matched_tracks.add(t_idx)

        # 处理未匹配的检测 - 创建新跟踪
        for d_idx in range(len(detections)):
            if d_idx not in matched_detections:
                det_bbox, det_cls, det_conf = detections[d_idx]
                class_name = self.class_names.get(det_cls, f'class_{det_cls}')
                
                new_track = Track(
                    track_id=self.next_id,
                    class_id=det_cls,
                    class_name=class_name,
                    bbox=det_bbox,
                    confidence=det_conf
                )
                new_track.history.append(new_track.center)
                self.tracks[self.next_id] = new_track
                self.next_id += 1
                
        # 处理未匹配的跟踪 - 增加age
        tracks_to_remove = []
        for t_idx, track_id in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self.tracks[track_id].predict()
                if self.tracks[track_id].age > self.max_age:
                    tracks_to_remove.append(track_id)
                    
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        return self._get_confirmed_tracks()
        
    def _get_confirmed_tracks(self) -> List[Track]:
        """获取已确认的跟踪目标"""
        confirmed = []
        for track in self.tracks.values():
            if track.hits >= self.min_hits and track.age == 0:
                confirmed.append(track)
        return confirmed
        
    def get_all_tracks(self) -> List[Track]:
        """获取所有活跃的跟踪目标"""
        return [t for t in self.tracks.values() if t.age < self.max_age]
        
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """根据ID获取跟踪目标"""
        return self.tracks.get(track_id)
        
    def reset(self):
        """重置跟踪器"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
