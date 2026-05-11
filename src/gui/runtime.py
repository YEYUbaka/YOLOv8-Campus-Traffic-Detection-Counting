
"""GUI runtime threads and inference helpers."""

import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from core.class_profile import (
    DEFAULT_TRAFFIC_CLASS_PROFILE,
    TrafficClassProfile,
)
from core.collision_warner import CollisionWarner
from core.counter import ClassCounter, LineCounter, VehicleSessionCounter
from core.speed_estimator import SpeedEstimator
from core.tracker import ObjectTracker
from core.zone_detector import ZoneDetector
from utils.drawing import DrawingUtils


def resolve_runtime_device(device_preference):
    """解析推理设备偏好。"""
    preference = (device_preference or "auto").lower()
    has_cuda = torch.cuda.is_available()

    if preference == "cpu":
        return "cpu", False, "CPU"

    if preference == "gpu":
        if has_cuda:
            return 0, True, "GPU"
        return "cpu", False, "CPU"

    if has_cuda:
        return 0, True, "GPU"
    return "cpu", False, "CPU"

def compute_video_processing_size(width, height, resolution_profile):
    """按视频预处理档位计算工作分辨率，保持比例且不放大。"""
    safe_width = max(1, int(width or 1))
    safe_height = max(1, int(height or 1))
    profile = (resolution_profile or "720p").lower()

    if profile == "1080p":
        limit_width, limit_height = 1920, 1080
    else:
        limit_width, limit_height = 1280, 720

    scale = min(
        1.0,
        limit_width / safe_width,
        limit_height / safe_height,
    )
    target_width = max(1, int(round(safe_width * scale)))
    target_height = max(1, int(round(safe_height * scale)))
    return target_width, target_height

def open_video_capture(source, source_mode):
    preferred_backends = []
    if os.name == "nt":
        if source_mode == "camera" and hasattr(cv2, "CAP_DSHOW"):
            preferred_backends.append(cv2.CAP_DSHOW)
        elif source_mode == "video" and hasattr(cv2, "CAP_FFMPEG"):
            preferred_backends.append(cv2.CAP_FFMPEG)

    for backend in preferred_backends:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(source)

class FrameReaderThread(QThread):
    """独立的视频读取线程，仅保留最近一帧原始图像。"""

    def __init__(self, source, source_mode, source_fps=0.0):
        super().__init__()
        self.source = source
        self.source_mode = source_mode
        self.source_fps = float(source_fps or 0.0)
        self.running = True
        self.paused = False
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._latest_frame = None
        self._latest_sequence = 0
        self._latest_decode_ms = 0.0
        self._open_error = None
        self._video_ended = False

    def run(self):
        cap = open_video_capture(self.source, self.source_mode)
        if self.source_mode == "camera":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self._open_error = f"无法打开视频源：{self.source}"
            self._ready_event.set()
            return

        self._ready_event.set()

        # 视频模式帧率控制
        frame_duration = 1.0 / self.source_fps if self.source_mode == "video" and self.source_fps > 0 else 0.0
        last_frame_time = 0.0

        while self.running:
            if self.paused:
                self.msleep(10)
                continue

            # 视频模式：按原始帧率控制读取速度
            if self.source_mode == "video" and frame_duration > 0:
                now = time.perf_counter()
                if last_frame_time > 0 and (now - last_frame_time) < frame_duration:
                    self.msleep(1)
                    continue
                last_frame_time = now

            read_start = time.perf_counter()
            ret, frame = cap.read()
            decode_ms = (time.perf_counter() - read_start) * 1000.0

            if not ret:
                if self.source_mode == "video":
                    self._video_ended = True
                    self.msleep(100)
                    continue
                self.msleep(5)
                continue

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_sequence += 1
                self._latest_decode_ms = decode_ms

            if self.source_mode == "camera":
                time.sleep(0.001)

        cap.release()

    def wait_until_ready(self, timeout=2.0):
        return self._ready_event.wait(timeout)

    def get_open_error(self):
        return self._open_error

    def get_latest_frame(self, after_sequence=0):
        with self._frame_lock:
            if self._latest_frame is None or self._latest_sequence <= after_sequence:
                return None
            return (
                self._latest_sequence,
                self._latest_frame.copy(),
                float(self._latest_decode_ms),
            )

    def set_paused(self, paused):
        self.paused = bool(paused)

    def stop(self):
        self.running = False
        self.wait()

class DetectionThread(QThread):
    """视频/摄像头检测线程，支持所有增强功能"""
    frame_ready = pyqtSignal(np.ndarray)
    preview_ready = pyqtSignal(int)
    status_update = pyqtSignal(str)
    stats_update = pyqtSignal(dict)  # 统计数据更新
    warning_update = pyqtSignal(dict)  # 预警信息更新
    overspeed_alert = pyqtSignal(dict)  # 超速报警信号

    def __init__(
        self,
        model,
        source,
        conf_threshold=0.5,
        iou_threshold=0.45,
        enable_tracking=True,
        enable_speed=True,
        enable_collision=True,
        enable_zone=True,
        enable_overspeed=False,
        overspeed_threshold=60,
        pixels_per_meter=90.0,
        safe_distance=5.0,
        class_profile: TrafficClassProfile = None,
        source_mode: str = "video",
        source_info: dict = None,
        device_preference: str = "auto",
        video_resolution_profile: str = "720p",
    ):
        super().__init__()
        self.model = model
        self.source = source
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_profile = class_profile or DEFAULT_TRAFFIC_CLASS_PROFILE
        self.source_mode = source_mode
        self.source_info = source_info or {}
        self.raw_source_width = int(self.source_info.get("width", 1280) or 1280)
        self.raw_source_height = int(self.source_info.get("height", 720) or 720)
        self.source_fps = self._sanitize_fps(self.source_info.get("fps", 30.0))
        self.device_preference = (device_preference or "auto").lower()
        self.video_resolution_profile = video_resolution_profile if self.source_mode == "video" else None
        self.source_width = self.raw_source_width
        self.source_height = self.raw_source_height
        if self.source_mode == "video":
            self.source_width, self.source_height = compute_video_processing_size(
                self.raw_source_width,
                self.raw_source_height,
                self.video_resolution_profile,
            )

        # 功能开关
        self.enable_tracking = enable_tracking
        self.enable_speed = enable_speed
        self.enable_collision = enable_collision
        self.enable_zone = enable_zone
        self.enable_overspeed = enable_overspeed
        self.overspeed_threshold = float(overspeed_threshold)
        self._overspeed_alerted_ids = set()

        self.device, self.use_half, self.device_display_name = resolve_runtime_device(
            self.device_preference
        )
        self.cpu_optimized = self.device == "cpu"
        if self.cpu_optimized:
            try:
                torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        self.inference_imgsz = self._resolve_inference_imgsz()
        self.predict_classes = self.class_profile.classes_filter or None
        self.max_det = self._resolve_max_det()
        self.base_max_det = self.max_det
        self.target_frame_interval = 1.0 / self.source_fps if self.source_mode == "video" and self.source_fps > 0 else 0.0
        self.sequential_video_playback = self.source_mode == "video"
        self._video_frame_interval = 1.0 / max(1.0, self.source_fps)
        self._last_video_frame_time = 0.0

        # 视频进度追踪
        self._video_total_frames = 0
        self._video_current_frame = 0
        self._video_progress_sec = 0.0
        self._video_total_sec = 0.0
        cpu_live_mode = self.cpu_optimized and self.source_mode in {"video", "camera"}
        self.frame_stride = 1
        self.max_frame_stride = 1
        self._stride_recovery_threshold = 0
        self.base_warning_display_limit = 4 if cpu_live_mode else 8
        self.base_max_tracking_line_tracks = 2 if cpu_live_mode else 6
        self.base_max_warning_markers = 2 if cpu_live_mode else 5
        self.base_max_labels_per_frame = 2 if cpu_live_mode else 6
        self.base_min_label_box_edge = 56 if cpu_live_mode else (40 if self.source_mode in {"video", "camera"} else 28)
        self.warning_display_limit = self.base_warning_display_limit
        self.max_tracking_line_tracks = self.base_max_tracking_line_tracks
        self.max_warning_markers = self.base_max_warning_markers
        self.max_labels_per_frame = self.base_max_labels_per_frame
        self.min_label_box_edge = self.base_min_label_box_edge
        self._stride_recovery_hits = 0
        self.display_fps = 0.0
        self.processed_fps = 0.0
        self.inference_fps = 0.0
        self.avg_frame_step = 1.0
        self.decode_ms = 0.0
        self.model_ms = 0.0
        self.analysis_ms = 0.0
        self.draw_ms = 0.0
        self.preview_ms = 0.0
        self.fast_live_overlay = self.source_mode in {"video", "camera"}
        self.base_speed_analysis_interval = 3 if cpu_live_mode else 2
        self.base_collision_analysis_interval = 4 if cpu_live_mode else 3
        self.base_zone_analysis_interval = 4 if cpu_live_mode else 3
        self.speed_analysis_interval = self.base_speed_analysis_interval
        self.collision_analysis_interval = self.base_collision_analysis_interval
        self.zone_analysis_interval = self.base_zone_analysis_interval
        self.stats_emit_interval = 0.33 if cpu_live_mode else 0.25
        self.warning_emit_interval = 0.60 if cpu_live_mode else 0.40
        self._inference_cycle = 0
        self._last_stats_emit_at = 0.0
        self._last_warning_emit_at = 0.0
        self._cached_speed_infos = []
        self._cached_collision_warnings = []
        self._cached_zone_violations = []
        self.fast_predictor = None
        self.frame_reader = None
        self._preview_lock = threading.Lock()
        self._latest_preview_image = None
        self._latest_preview_sequence = 0
        self._latest_preview_source_size = (self.source_width, self.source_height)
        self._preview_target_size = (0, 0)
        self._warning_timeline = deque(maxlen=50)
        self._warning_state_lookup = {}
        self._warning_state_order = []

        # 初始化核心模块
        self.tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        self.class_counter = ClassCounter()
        self.vehicle_session_counter = VehicleSessionCounter()
        self.line_counter = LineCounter()
        self.speed_estimator = SpeedEstimator(pixels_per_meter=pixels_per_meter, fps=30.0)
        self.collision_warner = CollisionWarner(
            pixels_per_meter=pixels_per_meter,
            safe_distance=safe_distance
        )
        self.zone_detector = ZoneDetector(pixels_per_meter=pixels_per_meter)
        self._apply_class_profile()
        DrawingUtils.configure_class_profile(self.class_profile)

        # 绘制状态
        self.drawing_mode = None  # 'line', 'zone'
        self.drawing_points = []
        self.current_mouse_pos = None

        self.last_render_payload = self._build_empty_render_payload()
        self.last_stats_payload = self._build_empty_stats_payload()
        self.last_warning_payload = self._build_empty_warning_payload()

        # 运行状态
        self.running = True
        self.paused = False

    def _apply_class_profile(self):
        self.tracker.set_class_profile(self.class_profile)
        self.class_counter.set_class_profile(self.class_profile)
        self.vehicle_session_counter.set_class_profile(self.class_profile)
        self.line_counter.set_class_profile(self.class_profile)
        self.speed_estimator.set_class_profile(self.class_profile)
        self.collision_warner.set_class_profile(self.class_profile)
        self.zone_detector.set_class_profile(self.class_profile)

    def _build_empty_render_payload(self):
        return {
            "detections": [],
            "tracks": [],
            "speed_infos": [],
            "collision_warnings": [],
            "zone_violations": [],
            "overspeed_track_ids": set(),
            "class_counts": {},
            "up_counts": {},
            "down_counts": {},
            "unique_vehicle_total": 0,
            "current_vehicle_count": 0,
        }

    def _build_empty_stats_payload(self):
        return {
            "unique_vehicle_total": 0,
            "current_vehicle_count": 0,
            "warning_count": 0,
            "class_counts": {},
            "cumulative_class_counts": {},
            "up_counts": {},
            "down_counts": {},
            "track_count": 0,
            "speed_range": (0.0, 0.0),
            "display_fps": 0.0,
            "processed_fps": 0.0,
            "inference_fps": 0.0,
            "avg_frame_step": 1.0,
            "decode_ms": 0.0,
            "model_ms": 0.0,
            "analysis_ms": 0.0,
            "draw_ms": 0.0,
            "preview_ms": 0.0,
            "fps": 0.0,
        }

    def _build_empty_warning_payload(self):
        return {
            "active_items": [],
            "timeline_items": [],
        }

    def _append_timeline_item(self, text):
        normalized = str(text or "").strip()
        if not normalized:
            return
        self._warning_timeline.appendleft(
            {
                "timestamp": time.strftime("%H:%M:%S"),
                "text": normalized,
            }
        )

    def _format_distance_text(self, distance):
        try:
            return f"{float(distance):.1f}m"
        except (TypeError, ValueError):
            return "未知"

    def _build_warning_payload(self, collision_warnings, zone_violations):
        active_items = []
        current_lookup = {}
        current_order = []

        for warning in collision_warnings:
            track_ids = sorted((int(warning.track_id_1), int(warning.track_id_2)))
            key = ("collision", track_ids[0], track_ids[1])
            if key in current_lookup:
                continue

            severity = str(getattr(warning, "severity", "high") or "high").lower()
            severity_label = {
                "high": "高",
                "medium": "中",
                "low": "低",
            }.get(severity, severity)
            distance_text = self._format_distance_text(getattr(warning, "distance", None))
            subject = (
                f"{warning.class_name_1}#{warning.track_id_1} / "
                f"{warning.class_name_2}#{warning.track_id_2}"
            )
            detail = f"距离 {distance_text} | 等级：{severity_label}"

            current_order.append(key)
            current_lookup[key] = {
                "item": {
                    "kind": "碰撞风险",
                    "subject": subject,
                    "detail": detail,
                    "severity": severity,
                },
                "start_text": f"碰撞风险：{subject}，距离 {distance_text}",
                "end_text": f"碰撞风险解除：{subject}",
            }
            active_items.append(current_lookup[key]["item"])

        for violation in zone_violations:
            zone_name = str(getattr(violation, "zone_name", "区域") or "区域")
            key = ("zone", int(violation.track_id), zone_name)
            if key in current_lookup:
                continue

            subject = f"{violation.class_name}#{violation.track_id}"
            detail = f"进入 {zone_name} | 等级：高"

            current_order.append(key)
            current_lookup[key] = {
                "item": {
                    "kind": "违规",
                    "subject": subject,
                    "detail": detail,
                    "severity": "high",
                },
                "start_text": f"违规：{subject} 进入 {zone_name}",
                "end_text": f"违规解除：{subject} 离开 {zone_name}",
            }
            active_items.append(current_lookup[key]["item"])

        previous_keys = set(self._warning_state_lookup.keys())
        current_keys = set(current_lookup.keys())
        state_changed = previous_keys != current_keys

        if state_changed:
            for key in current_order:
                if key not in previous_keys:
                    self._append_timeline_item(current_lookup[key]["start_text"])
            for key in self._warning_state_order:
                if key not in current_keys and key in self._warning_state_lookup:
                    self._append_timeline_item(self._warning_state_lookup[key]["end_text"])

        self._warning_state_lookup = current_lookup
        self._warning_state_order = list(current_order)

        return {
            "active_items": active_items,
            "timeline_items": list(self._warning_timeline),
        }, state_changed

    def _sanitize_fps(self, fps_value):
        try:
            fps = float(fps_value)
        except (TypeError, ValueError):
            return 30.0
        if fps <= 1.0 or fps > 240.0:
            return 30.0
        return fps

    def _resolve_inference_imgsz(self):
        longest_side = max(self.source_width, self.source_height)
        if self.source_mode in {"video", "camera"}:
            if self.cpu_optimized:
                if longest_side >= 1920:
                    return 192
                if longest_side >= 1280:
                    return 224
                if longest_side >= 960:
                    return 256
                return 288
            if longest_side >= 1600:
                return 320
            if longest_side >= 1280:
                return 352
            return 384
        return 640

    def _resolve_max_det(self):
        if self.source_mode not in {"video", "camera"}:
            return 64
        return 24 if self.cpu_optimized else 40

    def _using_cuda_runtime(self):
        return self.device != "cpu" and torch.cuda.is_available()

    def _sync_inference_device(self):
        if not self._using_cuda_runtime():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    def set_preview_target_size(self, width, height):
        width = max(0, int(width or 0))
        height = max(0, int(height or 0))
        with self._preview_lock:
            self._preview_target_size = (width, height)

    def get_latest_preview(self, after_sequence=0):
        with self._preview_lock:
            if self._latest_preview_image is None or self._latest_preview_sequence <= after_sequence:
                return None
            return (
                self._latest_preview_sequence,
                self._latest_preview_image,
                self._latest_preview_source_size,
            )

    def _build_preview_image(self, frame):
        with self._preview_lock:
            target_width, target_height = self._preview_target_size

        preview = frame
        if target_width > 0 and target_height > 0:
            frame_height, frame_width = frame.shape[:2]
            scale = min(target_width / max(1, frame_width), target_height / max(1, frame_height))
            scaled_width = max(1, int(frame_width * scale))
            scaled_height = max(1, int(frame_height * scale))
            if scaled_width != frame_width or scaled_height != frame_height:
                interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
                preview = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)

        if not preview.flags["C_CONTIGUOUS"]:
            preview = np.ascontiguousarray(preview)

        image_format = getattr(QImage, "Format_BGR888", None)
        if image_format is None:
            preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            image_format = QImage.Format_RGB888

        bytes_per_line = preview.shape[1] * preview.shape[2]
        return QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            bytes_per_line,
            image_format,
        ).copy()

    def _store_latest_preview(self, frame, sequence):
        preview_start = time.perf_counter()
        preview_image = self._build_preview_image(frame)
        preview_ms = (time.perf_counter() - preview_start) * 1000.0
        with self._preview_lock:
            self._latest_preview_image = preview_image
            self._latest_preview_sequence = int(sequence)
            self._latest_preview_source_size = (frame.shape[1], frame.shape[0])
        return preview_ms

    def _preprocess_inference_frame(self, frame):
        """仅对视频文件模式应用实时缩放预处理。"""
        if self.source_mode != "video" or not self.video_resolution_profile:
            return frame

        frame_height, frame_width = frame.shape[:2]
        target_width, target_height = compute_video_processing_size(
            frame_width,
            frame_height,
            self.video_resolution_profile,
        )
        if target_width == frame_width and target_height == frame_height:
            return frame

        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _refresh_cpu_runtime_profile(self, track_count):
        """根据当前拥挤度动态降低 CPU 实时检测的附加开销。"""
        if not (self.cpu_optimized and self.source_mode in {"video", "camera"}):
            return

        load_hint = max(int(track_count), self.frame_stride * 2)
        if load_hint >= 18:
            self.max_det = min(self.base_max_det, 16)
            self.speed_analysis_interval = 6
            self.collision_analysis_interval = 8
            self.zone_analysis_interval = 8
            self.max_tracking_line_tracks = 0
            self.max_warning_markers = 1
            self.max_labels_per_frame = 1
            self.warning_display_limit = 3
            self.min_label_box_edge = max(self.base_min_label_box_edge, 88)
            return

        if load_hint >= 10:
            self.max_det = min(self.base_max_det, 20)
            self.speed_analysis_interval = 5
            self.collision_analysis_interval = 6
            self.zone_analysis_interval = 6
            self.max_tracking_line_tracks = 1
            self.max_warning_markers = 2
            self.max_labels_per_frame = 1
            self.warning_display_limit = 3
            self.min_label_box_edge = max(self.base_min_label_box_edge, 72)
            return

        self.max_det = self.base_max_det
        self.speed_analysis_interval = self.base_speed_analysis_interval
        self.collision_analysis_interval = self.base_collision_analysis_interval
        self.zone_analysis_interval = self.base_zone_analysis_interval
        self.max_tracking_line_tracks = self.base_max_tracking_line_tracks
        self.max_warning_markers = self.base_max_warning_markers
        self.max_labels_per_frame = self.base_max_labels_per_frame
        self.warning_display_limit = self.base_warning_display_limit
        self.min_label_box_edge = self.base_min_label_box_edge

    def _tune_inference_imgsz(self, inference_elapsed):
        if self.source_mode not in {"video", "camera"}:
            return

        if self.cpu_optimized:
            if inference_elapsed > 0.45 and self.inference_imgsz > 192:
                self.inference_imgsz = max(192, self.inference_imgsz - 32)
            elif (
                inference_elapsed > 0.28
                and self.inference_imgsz > 224
                and self.frame_stride >= 4
            ):
                self.inference_imgsz = max(224, self.inference_imgsz - 32)
            elif inference_elapsed < 0.14 and self.inference_imgsz < 288 and self.frame_stride <= 3:
                self.inference_imgsz = min(288, self.inference_imgsz + 32)
            return

        if inference_elapsed > 0.080 and self.inference_imgsz > 320:
            self.inference_imgsz = max(320, self.inference_imgsz - 32)
        elif inference_elapsed < 0.040 and self.inference_imgsz < 384 and self.frame_stride <= 2:
            self.inference_imgsz = min(384, self.inference_imgsz + 32)

    def _open_capture(self, source):
        preferred_backends = []
        if os.name == "nt":
            if self.source_mode == "camera" and hasattr(cv2, "CAP_DSHOW"):
                preferred_backends.append(cv2.CAP_DSHOW)
            elif self.source_mode == "video" and hasattr(cv2, "CAP_FFMPEG"):
                preferred_backends.append(cv2.CAP_FFMPEG)

        for backend in preferred_backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
            cap.release()

        return cv2.VideoCapture(source)

    def _ensure_fast_predictor(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return False

        predictor.args.conf = self.conf_threshold
        predictor.args.iou = self.iou_threshold
        predictor.args.classes = self.predict_classes
        predictor.args.max_det = self.max_det
        predictor.args.half = self.use_half
        predictor.args.verbose = False
        predictor.args.imgsz = self.inference_imgsz
        predictor.imgsz = (self.inference_imgsz, self.inference_imgsz)
        self.fast_predictor = predictor
        return True

    def _predict_frame(self, frame):
        if self.fast_predictor is not None:
            try:
                predictor = self.fast_predictor
                predictor.args.conf = self.conf_threshold
                predictor.args.iou = self.iou_threshold
                predictor.args.classes = self.predict_classes
                predictor.args.max_det = self.max_det
                predictor.args.half = self.use_half
                predictor.args.imgsz = self.inference_imgsz
                predictor.imgsz = (self.inference_imgsz, self.inference_imgsz)
                predictor.batch = (["frame"], [frame], [""])
                im0s = [frame]
                im = predictor.preprocess(im0s)
                preds = predictor.inference(im)
                return predictor.postprocess(preds, im, im0s)
            except Exception:
                self.fast_predictor = None

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
            half=self.use_half,
            imgsz=self.inference_imgsz,
            classes=self.predict_classes,
            max_det=self.max_det,
        )
        self._ensure_fast_predictor()
        return results

    def _warmup_model(self):
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                try:
                    self.model.to("cuda:0")
                except Exception:
                    pass
            dummy = np.zeros((self.inference_imgsz, self.inference_imgsz, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.use_half,
                imgsz=self.inference_imgsz,
                classes=self.predict_classes,
                max_det=self.max_det,
            )
            self._ensure_fast_predictor()
        except Exception:
            pass

    def _extract_detections(self, results):
        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        try:
            xyxy_list = boxes.xyxy.detach().cpu().tolist()
            cls_list = boxes.cls.detach().to(dtype=torch.int32).cpu().tolist()
            conf_list = boxes.conf.detach().cpu().tolist()
        except Exception:
            xyxy_list = [box.xyxy[0].tolist() for box in boxes]
            cls_list = [int(box.cls[0]) for box in boxes]
            conf_list = [float(box.conf[0]) for box in boxes]

        for bbox, cls, conf in zip(xyxy_list, cls_list, conf_list):
            cls = int(cls)
            if cls not in self.class_profile.traffic_class_ids:
                continue
            detections.append((bbox, cls, float(conf)))
        return detections

    def _build_stats_payload(
        self,
        class_counts,
        cumulative_class_counts,
        up_counts,
        down_counts,
        unique_vehicle_total,
        current_vehicle_count,
        warning_count,
        track_count,
        speed_range,
    ):
        return {
            "unique_vehicle_total": unique_vehicle_total,
            "current_vehicle_count": current_vehicle_count,
            "warning_count": warning_count,
            "class_counts": class_counts,
            "cumulative_class_counts": cumulative_class_counts,
            "up_counts": up_counts,
            "down_counts": down_counts,
            "track_count": track_count,
            "speed_range": speed_range,
            "display_fps": round(self.display_fps, 1),
            "processed_fps": round(self.processed_fps, 1),
            "inference_fps": round(self.inference_fps, 1),
            "avg_frame_step": round(self.avg_frame_step, 2),
            "decode_ms": round(self.decode_ms, 1),
            "model_ms": round(self.model_ms, 1),
            "analysis_ms": round(self.analysis_ms, 1),
            "draw_ms": round(self.draw_ms, 1),
            "preview_ms": round(self.preview_ms, 1),
            "fps": round(self.inference_fps, 1),
        }

    def run(self):
        # 初始化视频总帧数
        if self.source_mode == "video":
            temp_cap = cv2.VideoCapture(self.source)
            if temp_cap.isOpened():
                self._video_total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._video_total_sec = self._video_total_frames / max(1.0, self.source_fps)
            temp_cap.release()

        self.frame_reader = FrameReaderThread(self.source, self.source_mode, self.source_fps)
        self.frame_reader.start()
        if not self.frame_reader.wait_until_ready():
            self.status_update.emit(f"无法打开视频源：{self.source}")
            self.running = False
            return

        open_error = self.frame_reader.get_open_error()
        if open_error:
            self.status_update.emit(open_error)
            self.running = False
            return

        self.speed_estimator.set_fps(self.source_fps)
        self._warmup_model()

        fps_window_start = time.perf_counter()
        processed_frame_count = 0
        model_frame_count = 0
        frame_step_total = 0.0
        decode_time_total = 0.0
        model_time_total = 0.0
        analysis_time_total = 0.0
        draw_time_total = 0.0
        preview_time_total = 0.0
        last_processed_sequence = 0

        try:
            while self.running:
                if self.paused:
                    self.frame_reader.set_paused(True)
                    self.status_update.emit("已暂停")
                    self.msleep(50)
                    continue

                self.frame_reader.set_paused(False)
                frame_packet = self.frame_reader.get_latest_frame(last_processed_sequence)

                # 视频播放完毕→自动暂停
                if frame_packet is None and self.source_mode == "video":
                    if self.frame_reader and self.frame_reader._video_ended:
                        self.set_paused(True)
                        self.status_update.emit("视频播放完毕")
                        self.msleep(100)
                        continue

                if frame_packet is None:
                    self.msleep(1)
                    continue

                frame_sequence, frame, decode_ms = frame_packet
                frame_step = max(1, frame_sequence - last_processed_sequence) if last_processed_sequence > 0 else 1
                last_processed_sequence = frame_sequence

                if self.source_mode == "camera":
                    frame_height, frame_width = frame.shape[:2]
                    if frame_width != self.source_width or frame_height != self.source_height:
                        self.raw_source_width = max(1, frame_width)
                        self.raw_source_height = max(1, frame_height)
                        self.source_width = self.raw_source_width
                        self.source_height = self.raw_source_height
                        self.inference_imgsz = self._resolve_inference_imgsz()

                # 视频模式：按原始帧率节流处理速度
                if self.source_mode == "video" and self._video_frame_interval > 0:
                    now = time.perf_counter()
                    if self._last_video_frame_time > 0:
                        sleep_needed = self._video_frame_interval - (now - self._last_video_frame_time)
                        if sleep_needed > 0.002:
                            time.sleep(sleep_needed)
                    self._last_video_frame_time = time.perf_counter()

                # 更新视频帧进度
                if self.source_mode == "video":
                    frame_index = self.frame_reader._latest_sequence if self.frame_reader else 0
                    self._video_current_frame = frame_index
                    self._video_progress_sec = self._video_current_frame / max(1.0, self.source_fps)

                try:
                    working_frame = self._preprocess_inference_frame(frame)
                    self._sync_inference_device()
                    model_start = time.perf_counter()
                    results = self._predict_frame(working_frame)
                    self._sync_inference_device()
                    model_elapsed = time.perf_counter() - model_start

                    analysis_start = time.perf_counter()
                    detections = self._extract_detections(results)

                    tracks = self.tracker.update(detections)
                    class_counts = self.class_counter.count(detections)
                    session_counts = self.vehicle_session_counter.update(tracks)
                    unique_vehicle_total = session_counts["unique_vehicle_total"]
                    current_vehicle_count = session_counts["current_vehicle_count"]
                    cumulative_class_counts = session_counts.get("unique_class_counts", {})

                    if self.line_counter.is_line_set():
                        count_result = self.line_counter.update(tracks)
                        up_counts = count_result.up_counts
                        down_counts = count_result.down_counts
                    else:
                        up_counts = {}
                        down_counts = {}

                    track_count = len(tracks)
                    self._refresh_cpu_runtime_profile(track_count)
                    self._inference_cycle += 1
                    run_speed_analysis = (
                        self._inference_cycle == 1
                        or self._inference_cycle % self.speed_analysis_interval == 0
                    )
                    run_collision_analysis = (
                        self._inference_cycle == 1
                        or self._inference_cycle % self.collision_analysis_interval == 0
                    )
                    run_zone_analysis = (
                        self._inference_cycle == 1
                        or self._inference_cycle % self.zone_analysis_interval == 0
                    )

                    speed_infos = list(self._cached_speed_infos)
                    if (
                        self.enable_speed
                        and self.enable_tracking
                        and track_count > 0
                        and run_speed_analysis
                    ):
                        effective_tracking_fps = self.source_fps / max(1, frame_step)
                        self.speed_estimator.set_fps(effective_tracking_fps)
                        speed_infos = self.speed_estimator.estimate_all(tracks)
                        self._cached_speed_infos = list(speed_infos)

                    collision_warnings = list(self._cached_collision_warnings)
                    if (
                        self.enable_collision
                        and self.enable_tracking
                        and track_count > 1
                        and run_collision_analysis
                    ):
                        collision_warnings = self.collision_warner.check_collisions(tracks)
                        self._cached_collision_warnings = list(collision_warnings)

                    zone_violations = list(self._cached_zone_violations)
                    if self.enable_zone and self.zone_detector.zones and run_zone_analysis:
                        zone_violations = self.zone_detector.check_violations(tracks)
                        self._cached_zone_violations = list(zone_violations)

                    # 超速检测
                    if self.enable_overspeed and self.enable_speed and speed_infos:
                        current_overspeed_ids = set()
                        for sinfo in speed_infos:
                            if sinfo.speed > self.overspeed_threshold:
                                current_overspeed_ids.add(sinfo.track_id)
                        # 检测到新的超速车辆时触发报警
                        new_overspeed = current_overspeed_ids - self._overspeed_alerted_ids
                        if new_overspeed:
                            overspeed_details = [
                                {
                                    "track_id": sinfo.track_id,
                                    "class_name": sinfo.class_name,
                                    "speed": sinfo.speed,
                                    "position": sinfo.position,
                                }
                                for sinfo in speed_infos
                                if sinfo.track_id in new_overspeed
                            ]
                            self.overspeed_alert.emit({
                                "type": "overspeed",
                                "vehicles": overspeed_details,
                                "threshold": self.overspeed_threshold,
                            })
                            # 记录日志
                            for od in overspeed_details:
                                self._append_timeline_item(
                                    f"超速：{od['class_name']}#"
                                    f"{od['track_id']} {od['speed']:.0f}km/h"
                                )
                        self._overspeed_alerted_ids = current_overspeed_ids

                    speed_range = self.speed_estimator.get_speed_range(tracks) if self.enable_speed else (0.0, 0.0)
                    warning_count = len(collision_warnings) + len(zone_violations)

                    overspeed_track_ids = set()
                    if self.enable_overspeed and speed_infos:
                        overspeed_track_ids = {
                            sinfo.track_id for sinfo in speed_infos
                            if sinfo.speed > self.overspeed_threshold
                        }

                    self.last_render_payload = {
                        "detections": detections,
                        "tracks": tracks,
                        "speed_infos": speed_infos,
                        "collision_warnings": collision_warnings,
                        "zone_violations": zone_violations,
                        "overspeed_track_ids": overspeed_track_ids,
                        "class_counts": class_counts,
                        "up_counts": up_counts,
                        "down_counts": down_counts,
                        "unique_vehicle_total": unique_vehicle_total,
                        "current_vehicle_count": current_vehicle_count,
                    }
                    self.last_stats_payload = self._build_stats_payload(
                        class_counts,
                        cumulative_class_counts,
                        up_counts,
                        down_counts,
                        unique_vehicle_total,
                        current_vehicle_count,
                        warning_count,
                        track_count,
                        speed_range,
                    )

                    now = time.perf_counter()
                    warning_payload, warning_changed = self._build_warning_payload(
                        collision_warnings,
                        zone_violations,
                    )
                    warning_refresh_due = (
                        (run_collision_analysis and track_count > 1)
                        or (run_zone_analysis and bool(self.zone_detector.zones))
                        or self._inference_cycle == 1
                    )
                    if warning_changed or (
                        warning_refresh_due
                        and (now - self._last_warning_emit_at >= self.warning_emit_interval)
                    ):
                        self.last_warning_payload = warning_payload
                        self.warning_update.emit(warning_payload)
                        self._last_warning_emit_at = now

                    analysis_elapsed = time.perf_counter() - analysis_start
                    self._tune_inference_imgsz(model_elapsed)

                    draw_start = time.perf_counter()
                    annotated_frame = self._draw_results(working_frame, **self.last_render_payload)
                    draw_elapsed = time.perf_counter() - draw_start
                    preview_ms = self._store_latest_preview(annotated_frame, frame_sequence)
                    self.preview_ready.emit(frame_sequence)

                    processed_frame_count += 1
                    model_frame_count += 1
                    frame_step_total += frame_step
                    decode_time_total += decode_ms / 1000.0
                    model_time_total += model_elapsed
                    analysis_time_total += analysis_elapsed
                    draw_time_total += draw_elapsed
                    preview_time_total += preview_ms / 1000.0

                    now = time.perf_counter()
                    elapsed = now - fps_window_start
                    if elapsed >= 1.0:
                        self.processed_fps = (
                            processed_frame_count / elapsed
                            if processed_frame_count > 0
                            else 0.0
                        )
                        self.display_fps = self.processed_fps
                        self.inference_fps = (
                            model_frame_count / model_time_total
                            if model_time_total > 0
                            else 0.0
                        )
                        self.avg_frame_step = (
                            frame_step_total / processed_frame_count
                            if processed_frame_count > 0
                            else 1.0
                        )
                        self.decode_ms = (
                            decode_time_total * 1000.0 / processed_frame_count
                            if processed_frame_count > 0
                            else 0.0
                        )
                        self.model_ms = (
                            model_time_total * 1000.0 / model_frame_count
                            if model_frame_count > 0
                            else 0.0
                        )
                        self.analysis_ms = (
                            analysis_time_total * 1000.0 / model_frame_count
                            if model_frame_count > 0
                            else 0.0
                        )
                        self.draw_ms = (
                            draw_time_total * 1000.0 / processed_frame_count
                            if processed_frame_count > 0
                            else 0.0
                        )
                        self.preview_ms = (
                            preview_time_total * 1000.0 / processed_frame_count
                            if processed_frame_count > 0
                            else 0.0
                        )
                        processed_frame_count = 0
                        model_frame_count = 0
                        frame_step_total = 0.0
                        decode_time_total = 0.0
                        model_time_total = 0.0
                        analysis_time_total = 0.0
                        draw_time_total = 0.0
                        preview_time_total = 0.0
                        fps_window_start = now

                    current_stats = dict(self.last_stats_payload)
                    current_stats["display_fps"] = round(self.display_fps, 1)
                    current_stats["processed_fps"] = round(self.processed_fps, 1)
                    current_stats["inference_fps"] = round(self.inference_fps, 1)
                    current_stats["avg_frame_step"] = round(self.avg_frame_step, 2)
                    current_stats["decode_ms"] = round(self.decode_ms, 1)
                    current_stats["model_ms"] = round(self.model_ms, 1)
                    current_stats["analysis_ms"] = round(self.analysis_ms, 1)
                    current_stats["draw_ms"] = round(self.draw_ms, 1)
                    current_stats["preview_ms"] = round(self.preview_ms, 1)
                    current_stats["fps"] = round(self.inference_fps, 1)

                    if now - self._last_stats_emit_at >= self.stats_emit_interval:
                        self.stats_update.emit(current_stats)
                        self._last_stats_emit_at = now

                except Exception as e:
                    self.status_update.emit(f"检测错误：{str(e)}")
                    self.msleep(5)
        finally:
            if self.frame_reader:
                self.frame_reader.stop()
                self.frame_reader = None
            self.status_update.emit("检测已停止")

    def _draw_results(
        self,
        frame,
        detections,
        tracks,
        speed_infos,
        collision_warnings,
        zone_violations,
        overspeed_track_ids=None,
        class_counts=None,
        up_counts=None,
        down_counts=None,
        unique_vehicle_total=None,
        current_vehicle_count=None,
    ):
        """绘制所有检测结果"""
        output = frame
        fast_live_overlay = self.fast_live_overlay
        text_items = None if fast_live_overlay else []
        show_tracking_lines = self.enable_tracking and len(tracks) <= self.max_tracking_line_tracks
        collision_markers = collision_warnings[:self.max_warning_markers]
        zone_markers = zone_violations[:self.max_warning_markers]
        overspeed_ids = set(overspeed_track_ids or [])

        def _bbox_area(bbox):
            x1, y1, x2, y2 = bbox
            return max(1.0, (x2 - x1) * (y2 - y1))

        def _bbox_min_edge(bbox):
            x1, y1, x2, y2 = bbox
            return min(max(1.0, x2 - x1), max(1.0, y2 - y1))

        labeled_track_ids = set()
        labeled_detection_indices = set()
        if tracks:
            ranked_tracks = sorted(tracks, key=lambda track: _bbox_area(track.bbox), reverse=True)
            for track in ranked_tracks[:self.max_labels_per_frame]:
                if _bbox_min_edge(track.bbox) >= self.min_label_box_edge:
                    labeled_track_ids.add(track.track_id)
        else:
            ranked_detections = sorted(
                enumerate(detections),
                key=lambda item: _bbox_area(item[1][0]),
                reverse=True,
            )
            for det_index, (bbox, _, _) in ranked_detections[:self.max_labels_per_frame]:
                if _bbox_min_edge(bbox) >= self.min_label_box_edge:
                    labeled_detection_indices.add(det_index)

        # 绘制违规区域
        if self.enable_zone and self.zone_detector.zones:
            output = self.zone_detector.draw_zones(output, show_labels=not fast_live_overlay)

        # 绘制检测线
        if self.line_counter.is_line_set():
            line = self.line_counter.get_line()
            output = DrawingUtils.draw_counting_line(
                output,
                line[0],
                line[1],
                text_items=text_items,
                show_text=not fast_live_overlay,
                ascii_label=fast_live_overlay,
            )

        # 绘制绘图预览
        if self.drawing_mode and self.drawing_points:
            output = DrawingUtils.draw_drawing_preview(
                output, self.drawing_points,
                self.current_mouse_pos, self.drawing_mode
            )

        # 构建速度字典
        speed_dict = {s.track_id: s.speed for s in speed_infos}

        # 绘制检测结果
        if tracks:
            for track in tracks:
                speed = speed_dict.get(track.track_id, None)
                output = DrawingUtils.draw_detection_box(
                    output,
                    track.bbox,
                    track.class_id,
                    track.track_id if self.enable_tracking else None,
                    track.confidence,
                    speed if self.enable_speed else None,
                    show_label=track.track_id in labeled_track_ids,
                    text_items=text_items,
                    ascii_label=fast_live_overlay,
                )

                if show_tracking_lines and len(track.history) > 1:
                    output = DrawingUtils.draw_tracking_line(output, track.history)

                # 超速高亮标记
                if self.enable_overspeed and track.track_id in overspeed_ids:
                    x1, y1, x2, y2 = [int(v) for v in track.bbox]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    output = DrawingUtils.draw_warning(output, center, "超速!", "high")
                    # 绘制红色边框
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            for det_index, (bbox, class_id, confidence) in enumerate(detections):
                output = DrawingUtils.draw_detection_box(
                    output,
                    bbox,
                    class_id,
                    None,
                    confidence,
                    None,
                    show_label=det_index in labeled_detection_indices,
                    text_items=text_items,
                    ascii_label=fast_live_overlay,
                )

        # 绘制碰撞预警
        for warning in collision_markers:
            # 高亮显示碰撞风险车辆
            pos = warning.position_1
            output = DrawingUtils.draw_warning(output, pos, "碰撞风险", warning.severity)
            pos2 = warning.position_2
            output = DrawingUtils.draw_warning(output, pos2, "碰撞风险", warning.severity)

        # 绘制违规标记
        for violation in zone_markers:
            output = DrawingUtils.draw_warning(
                output, violation.position, "违规", "high"
            )

        if text_items is None:
            return output
        return DrawingUtils.render_text_items(output, text_items)

    def update_params(self, conf_threshold, iou_threshold):
        """实时更新检测参数"""
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def update_overspeed_threshold(self, threshold):
        """更新超速阈值"""
        self.overspeed_threshold = float(threshold)

    def update_calibration(self, pixels_per_meter):
        """更新标定参数"""
        self.speed_estimator.set_calibration(pixels_per_meter)
        self.collision_warner.set_calibration(pixels_per_meter)
        self.zone_detector.pixels_per_meter = pixels_per_meter

    def update_safe_distance(self, distance):
        """更新安全距离"""
        self.collision_warner.set_safe_distance(distance)

    def set_counting_line(self, start, end):
        """设置检测线"""
        self.line_counter.set_line(start, end)

    def add_zone(self, points, name=None, zone_type='no_parking'):
        """添加违规区域"""
        self.zone_detector.add_zone(points, name, zone_type)

    def clear_zones(self):
        """清除所有区域"""
        self.zone_detector.clear_zones()

    def set_drawing_mode(self, mode, points=None):
        """设置绘制模式"""
        self.drawing_mode = mode
        self.drawing_points = points or []

    def add_drawing_point(self, point):
        """添加绘制点"""
        self.drawing_points.append(point)
        return self.drawing_points

    def set_mouse_position(self, pos):
        """设置鼠标位置"""
        self.current_mouse_pos = pos

    def finish_drawing(self):
        """完成绘制"""
        points = self.drawing_points.copy()

        if self.drawing_mode == 'line' and len(points) >= 2:
            self.set_counting_line(points[0], points[1])
        elif self.drawing_mode == 'zone' and len(points) >= 3:
            self.add_zone(points)

        self.drawing_mode = None
        self.drawing_points = []
        return points

    def get_video_progress(self):
        """获取视频播放进度 (当前秒, 总秒数)"""
        if self.source_mode != "video":
            return None
        return (self._video_progress_sec, self._video_total_sec)

    def stop(self):
        self.running = False
        if self.frame_reader:
            self.frame_reader.stop()
            self.frame_reader = None
        self.wait()

    def set_paused(self, paused):
        self.paused = bool(paused)
        if self.frame_reader:
            self.frame_reader.set_paused(self.paused)

    def reset_counters(self):
        """重置所有计数器"""
        self.tracker.reset()
        self.vehicle_session_counter.reset()
        self.line_counter.reset()
        self.speed_estimator.reset()
        self.collision_warner.reset()
        self.zone_detector.reset()
        self.last_render_payload = self._build_empty_render_payload()
        self.last_stats_payload = self._build_empty_stats_payload()
        self.last_warning_payload = self._build_empty_warning_payload()
        self._cached_speed_infos = []
        self._cached_collision_warnings = []
        self._cached_zone_violations = []
        self._warning_timeline.clear()
        self._warning_state_lookup = {}
        self._warning_state_order = []
        self._inference_cycle = 0
        self._last_stats_emit_at = 0.0
        self._last_warning_emit_at = 0.0
        self.display_fps = 0.0
        self.processed_fps = 0.0
        self.inference_fps = 0.0
        self.avg_frame_step = 1.0
        self.decode_ms = 0.0
        self.model_ms = 0.0
        self.analysis_ms = 0.0
        self.draw_ms = 0.0
        self.preview_ms = 0.0
        with self._preview_lock:
            self._latest_preview_image = None
            self._latest_preview_sequence = 0
