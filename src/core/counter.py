"""
计数模块
包含类别统计、会话级车辆去重计数和上下行计数功能。
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .class_profile import DEFAULT_TRAFFIC_CLASS_PROFILE, TrafficClassProfile


@dataclass
class CountResult:
    """计数结果"""
    class_counts: Dict[str, int] = field(default_factory=dict)
    up_counts: Dict[str, int] = field(default_factory=dict)
    down_counts: Dict[str, int] = field(default_factory=dict)
    total_up: int = 0
    total_down: int = 0


@dataclass
class UniqueVehicleState:
    """会话级唯一车辆状态。"""
    unique_id: int
    class_id: int
    class_name: str
    last_center: Tuple[float, float]
    last_bbox: List[float]
    last_seen_frame: int
    active_track_id: Optional[int] = None


class ClassCounter:
    """类别计数器 - 统计各类型目标数量"""
    
    def __init__(self):
        """初始化类别计数器"""
        self.display_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)

    def set_class_profile(self, profile: TrafficClassProfile):
        """同步当前模型的类别展示名。"""
        self.display_names = dict(profile.display_names)
        
    def count(self, detections: List[Tuple]) -> Dict[str, int]:
        """
        统计各类别数量
        
        Args:
            detections: 检测结果列表，每个元素包含类别信息
            
        Returns:
            类别计数字典 {类别名: 数量}
        """
        counts = defaultdict(int)
        
        for det in detections:
            if hasattr(det, 'class_id'):
                class_id = det.class_id
            elif isinstance(det, (list, tuple)) and len(det) >= 2:
                class_id = det[1]
            else:
                continue
                
            class_name = self.display_names.get(class_id, f'未知({class_id})')
            counts[class_name] += 1
            
        return dict(counts)
        
    def count_with_ids(self, tracks: List) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
        """
        统计各类别数量和对应的跟踪ID
        
        Args:
            tracks: 跟踪目标列表
            
        Returns:
            (类别计数, 类别对应的ID列表)
        """
        counts = defaultdict(int)
        ids = defaultdict(list)
        
        for track in tracks:
            if hasattr(track, 'class_id') and hasattr(track, 'track_id'):
                class_id = track.class_id
                track_id = track.track_id
            else:
                continue
                
            class_name = self.display_names.get(class_id, f'未知({class_id})')
            counts[class_name] += 1
            ids[class_name].append(track_id)
            
        return dict(counts), dict(ids)


class VehicleSessionCounter:
    """
    会话级唯一车辆计数器。

    通过轨迹 ID 优先关联，并对短时断开的新轨迹做一次轻量级重关联，
    降低同一车辆被重复计入总数的概率。
    """

    def __init__(self):
        self.vehicle_class_ids = set(DEFAULT_TRAFFIC_CLASS_PROFILE.vehicle_class_ids)
        self.display_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)

        self.frame_index = 0
        self.next_unique_id = 1
        self.track_to_unique: Dict[int, int] = {}
        self.unique_vehicles: Dict[int, UniqueVehicleState] = {}
        self.reconnect_candidates: Dict[int, UniqueVehicleState] = {}
        self.reconnect_window = 45
        self.size_similarity_threshold = 0.5

    def set_class_profile(self, profile: TrafficClassProfile):
        """同步当前模型的车辆类集合和展示名。"""
        self.vehicle_class_ids = set(profile.vehicle_class_ids)
        self.display_names = dict(profile.display_names)

    def _compute_distance(
        self,
        center1: Tuple[float, float],
        center2: Tuple[float, float],
    ) -> float:
        return math.hypot(center1[0] - center2[0], center1[1] - center2[1])

    def _compute_bbox_size_similarity(
        self,
        bbox1: List[float],
        bbox2: List[float],
    ) -> float:
        w1 = max(1.0, bbox1[2] - bbox1[0])
        h1 = max(1.0, bbox1[3] - bbox1[1])
        w2 = max(1.0, bbox2[2] - bbox2[0])
        h2 = max(1.0, bbox2[3] - bbox2[1])

        area_ratio = min(w1 * h1, w2 * h2) / max(w1 * h1, w2 * h2)
        aspect_ratio = min(w1 / h1, w2 / h2) / max(w1 / h1, w2 / h2)
        return 0.7 * area_ratio + 0.3 * aspect_ratio

    def _find_best_match(self, track) -> Optional[int]:
        """
        为新出现的 track_id 寻找最近的唯一车辆。
        只处理短时断开后的重连，避免把真正的新车误当成旧车。
        """
        best_match = None
        best_score = 0.0

        bbox = track.bbox if hasattr(track, 'bbox') else None
        center = track.center if hasattr(track, 'center') else None
        if bbox is None or center is None:
            return None

        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        diagonal = math.hypot(width, height)
        max_distance = max(80.0, diagonal * 1.8)

        self._prune_reconnect_candidates()

        for unique_id, state in self.reconnect_candidates.items():
            if state.class_id != track.class_id:
                continue

            frame_gap = self.frame_index - state.last_seen_frame
            if frame_gap <= 0 or frame_gap > self.reconnect_window:
                continue

            if state.active_track_id is not None:
                continue

            center_distance = self._compute_distance(center, state.last_center)
            if center_distance > max_distance:
                continue

            size_similarity = self._compute_bbox_size_similarity(bbox, state.last_bbox)
            if size_similarity < self.size_similarity_threshold:
                continue

            distance_score = max(0.0, 1.0 - center_distance / max_distance)
            score = 0.65 * distance_score + 0.35 * size_similarity
            if score > best_score:
                best_score = score
                best_match = unique_id

        return best_match

    def _prune_reconnect_candidates(self):
        """只保留 reconnect_window 内仍可重连的唯一车辆候选。"""
        expired_ids = [
            unique_id
            for unique_id, state in self.reconnect_candidates.items()
            if (self.frame_index - state.last_seen_frame) > self.reconnect_window
        ]
        for unique_id in expired_ids:
            self.reconnect_candidates.pop(unique_id, None)

    def update(self, tracks: List) -> Dict[str, int]:
        """更新会话级唯一车辆总数与当前画面车辆数。"""
        self.frame_index += 1

        vehicle_tracks = [
            track for track in tracks
            if hasattr(track, 'class_id') and track.class_id in self.vehicle_class_ids
        ]

        current_track_ids = {track.track_id for track in vehicle_tracks if hasattr(track, 'track_id')}

        for track_id in list(self.track_to_unique.keys()):
            if track_id not in current_track_ids:
                unique_id = self.track_to_unique.pop(track_id)
                state = self.unique_vehicles.get(unique_id)
                if state and state.active_track_id == track_id:
                    state.active_track_id = None
                    self.reconnect_candidates[unique_id] = state

        self._prune_reconnect_candidates()

        for track in vehicle_tracks:
            track_id = getattr(track, 'track_id', None)
            bbox = getattr(track, 'bbox', None)
            center = getattr(track, 'center', None)
            class_id = getattr(track, 'class_id', None)
            if track_id is None or bbox is None or center is None or class_id is None:
                continue

            class_name = self.display_names.get(class_id, '未知车辆')

            if track_id in self.track_to_unique:
                unique_id = self.track_to_unique[track_id]
            else:
                unique_id = self._find_best_match(track)
                if unique_id is None:
                    unique_id = self.next_unique_id
                    self.next_unique_id += 1
                    self.unique_vehicles[unique_id] = UniqueVehicleState(
                        unique_id=unique_id,
                        class_id=class_id,
                        class_name=class_name,
                        last_center=center,
                        last_bbox=list(bbox),
                        last_seen_frame=self.frame_index,
                        active_track_id=track_id,
                    )
                    self.reconnect_candidates.pop(unique_id, None)
                self.track_to_unique[track_id] = unique_id

            state = self.unique_vehicles[unique_id]
            state.class_id = class_id
            state.class_name = class_name
            state.last_center = center
            state.last_bbox = list(bbox)
            state.last_seen_frame = self.frame_index
            state.active_track_id = track_id
            self.reconnect_candidates.pop(unique_id, None)

        return {
            'unique_vehicle_total': len(self.unique_vehicles),
            'current_vehicle_count': len(vehicle_tracks),
        }

    def reset(self):
        """重置会话级唯一车辆计数。"""
        self.frame_index = 0
        self.next_unique_id = 1
        self.track_to_unique.clear()
        self.unique_vehicles.clear()
        self.reconnect_candidates.clear()


class LineCounter:
    """
    检测线计数器 - 统计上下行数量
    通过检测目标穿越检测线的方向来判断上行/下行
    """
    
    def __init__(self):
        """初始化检测线计数器"""
        self.line_start: Optional[Tuple[int, int]] = None
        self.line_end: Optional[Tuple[int, int]] = None
        self.line_direction: str = 'vertical'  # 'vertical' 或 'horizontal'
        
        # 上行方向定义（检测线的法向量方向）
        # 'up' = 法向量指向上的方向
        # 'down' = 法向量指向下的方向
        self.up_direction: Tuple[float, float] = (0, -1)  # 默认向上
        
        # 计数统计
        self.up_counts: Dict[str, int] = defaultdict(int)  # 上行计数
        self.down_counts: Dict[str, int] = defaultdict(int)  # 下行计数
        
        # 已计数的跟踪ID（避免重复计数）
        self.counted_ids: set = set()
        
        # 跟踪历史记录（用于判断方向）
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        self.display_names = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)

    def set_class_profile(self, profile: TrafficClassProfile):
        """同步当前模型的类别展示名。"""
        self.display_names = dict(profile.display_names)
        
    def set_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        """
        设置检测线
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
        """
        self.line_start = start
        self.line_end = end
        
        # 自动判断检测线方向
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > dy:
            self.line_direction = 'horizontal'
        else:
            self.line_direction = 'vertical'
            
        # 计算法向量（用于判断上下行方向）
        # 法向量垂直于检测线
        line_vec = (end[0] - start[0], end[1] - start[1])
        length = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
        if length > 0:
            # 法向量（逆时针旋转90度）
            self.up_direction = (-line_vec[1] / length, line_vec[0] / length)
            
    def set_up_direction(self, direction: str = 'up'):
        """
        设置上行方向
        
        Args:
            direction: 'up' 或 'down'，默认 'up'
        """
        if direction == 'down':
            self.up_direction = (-self.up_direction[0], -self.up_direction[1])
            
    def _point_to_line_distance(self, point: Tuple[float, float]) -> float:
        """
        计算点到检测线的距离（带符号）
        正值表示在法向量方向一侧，负值表示在另一侧
        """
        if self.line_start is None or self.line_end is None:
            return 0
            
        # 使用向量叉积判断方向
        px, py = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        
        # 向量叉积 (P1P x P1P2)
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        
        # 归一化
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > 0:
            cross /= length
            
        return cross
        
    def _is_crossing_line(self, prev_pos: Tuple[float, float], 
                          curr_pos: Tuple[float, float]) -> Optional[str]:
        """
        判断目标是否穿越检测线
        
        Args:
            prev_pos: 上一帧位置
            curr_pos: 当前帧位置
            
        Returns:
            'up' 表示上行，'down' 表示下行，None 表示未穿越
        """
        if self.line_start is None or self.line_end is None:
            return None
            
        prev_dist = self._point_to_line_distance(prev_pos)
        curr_dist = self._point_to_line_distance(curr_pos)
        
        # 判断是否穿越（符号变化）
        if prev_dist * curr_dist < 0:
            # 穿越了检测线
            # 判断方向：当前距离与法向量方向的关系
            if curr_dist > 0:
                return 'up'
            else:
                return 'down'
                
        return None
        
    def update(self, tracks: List) -> CountResult:
        """
        更新计数
        
        Args:
            tracks: 跟踪目标列表，每个目标需要有 track_id, class_id, center 属性
            
        Returns:
            CountResult 计数结果
        """
        for track in tracks:
            if not hasattr(track, 'track_id') or not hasattr(track, 'center'):
                continue
                
            track_id = track.track_id
            class_id = track.class_id if hasattr(track, 'class_id') else 0
            class_name = self.display_names.get(class_id, '未知')
            
            # 记录历史位置
            self.track_history[track_id].append(track.center)
            
            # 只保留最近10帧
            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id].pop(0)
                
            # 检查是否穿越检测线
            if track_id not in self.counted_ids and len(self.track_history[track_id]) >= 2:
                prev_pos = self.track_history[track_id][-2]
                curr_pos = self.track_history[track_id][-1]
                
                direction = self._is_crossing_line(prev_pos, curr_pos)
                
                if direction == 'up':
                    self.up_counts[class_name] += 1
                    self.counted_ids.add(track_id)
                elif direction == 'down':
                    self.down_counts[class_name] += 1
                    self.counted_ids.add(track_id)
                    
        # 清理已消失的目标历史
        current_ids = {t.track_id for t in tracks if hasattr(t, 'track_id')}
        ids_to_remove = [tid for tid in self.track_history if tid not in current_ids]
        for tid in ids_to_remove:
            del self.track_history[tid]
            self.counted_ids.discard(tid)
            
        # 构建结果
        result = CountResult()
        result.up_counts = dict(self.up_counts)
        result.down_counts = dict(self.down_counts)
        result.total_up = sum(self.up_counts.values())
        result.total_down = sum(self.down_counts.values())
        
        return result
        
    def reset(self):
        """重置计数器"""
        self.up_counts.clear()
        self.down_counts.clear()
        self.counted_ids.clear()
        self.track_history.clear()
        
    def is_line_set(self) -> bool:
        """检查检测线是否已设置"""
        return self.line_start is not None and self.line_end is not None
        
    def get_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """获取检测线坐标"""
        if self.is_line_set():
            return (self.line_start, self.line_end)
        return None
