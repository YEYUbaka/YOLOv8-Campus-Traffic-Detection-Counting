"""
YOLOv8 校园交通目标检测系统 - 增强版 GUI
支持目标跟踪、类别统计、上下行计数、速度检测、碰撞预警、违规区域检测
"""

from ultralytics import YOLO
import sys
import os
import threading
import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QPlainTextEdit, QSlider, QGroupBox, QRadioButton, QMessageBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSpinBox, QDoubleSpinBox, QComboBox,
    QGridLayout, QSizePolicy, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from datetime import datetime
import numpy as np

import PyQt5
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time

# 导入核心模块
from core.tracker import ObjectTracker
from core.counter import ClassCounter, LineCounter, VehicleSessionCounter
from core.class_profile import (
    DEFAULT_TRAFFIC_CLASS_PROFILE,
    TrafficClassProfile,
    build_traffic_class_profile,
)
from core.speed_estimator import SpeedEstimator
from core.collision_warner import CollisionWarner
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

    def run(self):
        cap = open_video_capture(self.source, self.source_mode)
        if self.source_mode == "camera":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self._open_error = f"无法打开视频源：{self.source}"
            self._ready_event.set()
            return

        self._ready_event.set()
        target_interval = 1.0 / self.source_fps if self.source_mode == "video" and self.source_fps > 0 else 0.0
        next_deadline = time.perf_counter()

        while self.running:
            if self.paused:
                self.msleep(10)
                continue

            read_start = time.perf_counter()
            ret, frame = cap.read()
            decode_ms = (time.perf_counter() - read_start) * 1000.0

            if not ret:
                if self.source_mode == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    next_deadline = time.perf_counter()
                    continue
                self.msleep(5)
                continue

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_sequence += 1
                self._latest_decode_ms = decode_ms

            if target_interval > 0:
                next_deadline += target_interval
                now = time.perf_counter()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                else:
                    next_deadline = now
            elif self.source_mode == "camera":
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


class VideoDisplayLabel(QLabel):
    """支持鼠标事件的视频显示标签"""
    
    mouse_clicked = pyqtSignal(int, int, object)  # x, y, event
    mouse_moved = pyqtSignal(int, int)  # x, y
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouse_clicked.emit(event.x(), event.y(), event)
        elif event.button() == Qt.RightButton:
            self.mouse_clicked.emit(event.x(), event.y(), event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.mouse_moved.emit(event.x(), event.y())
        super().mouseMoveEvent(event)


class StableVideoViewport(QWidget):
    display_size_changed = pyqtSignal(int, int)
    """稳定的视频视口，避免 pixmap 尺寸反向驱动整体布局。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.aspect_ratio = 16 / 9
        self._last_display_size = (0, 0)
        self.video_label = VideoDisplayLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setStyleSheet("background: transparent; border: none;")
        self.video_label.setText("选择检测模式后点击开始")

    def set_source_size(self, width: int, height: int):
        if width > 0 and height > 0:
            self.aspect_ratio = width / height
            self._update_video_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_video_geometry()

    def _update_video_geometry(self):
        container_w = max(1, self.width())
        container_h = max(1, self.height())
        if container_w / container_h > self.aspect_ratio:
            target_h = container_h
            target_w = int(target_h * self.aspect_ratio)
        else:
            target_w = container_w
            target_h = int(target_w / self.aspect_ratio)

        x = max(0, (container_w - target_w) // 2)
        y = max(0, (container_h - target_h) // 2)
        self.video_label.setGeometry(x, y, max(1, target_w), max(1, target_h))
        display_size = (max(1, target_w), max(1, target_h))
        if display_size != self._last_display_size:
            self._last_display_size = display_size
            self.display_size_changed.emit(*display_size)


class DetectionThread(QThread):
    """视频/摄像头检测线程，支持所有增强功能"""
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    stats_update = pyqtSignal(dict)  # 统计数据更新
    warning_update = pyqtSignal(list)  # 预警信息更新

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
        pixels_per_meter=10.0,
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
        self.sequential_video_playback = self.cpu_optimized and self.source_mode == "video"
        cpu_live_mode = self.cpu_optimized and self.source_mode in {"video", "camera"}
        self.frame_stride = 4 if cpu_live_mode else 1
        self.max_frame_stride = 32 if cpu_live_mode else 10
        self._stride_recovery_threshold = 8 if cpu_live_mode else 12
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
        self.inference_fps = 0.0
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
            "up_counts": {},
            "down_counts": {},
            "track_count": 0,
            "speed_range": (0.0, 0.0),
            "display_fps": 0.0,
            "inference_fps": 0.0,
            "decode_ms": 0.0,
            "model_ms": 0.0,
            "analysis_ms": 0.0,
            "draw_ms": 0.0,
            "preview_ms": 0.0,
            "fps": 0.0,
        }

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
            "up_counts": up_counts,
            "down_counts": down_counts,
            "track_count": track_count,
            "speed_range": speed_range,
            "display_fps": round(self.display_fps, 1),
            "inference_fps": round(self.inference_fps, 1),
            "decode_ms": round(self.decode_ms, 1),
            "model_ms": round(self.model_ms, 1),
            "analysis_ms": round(self.analysis_ms, 1),
            "draw_ms": round(self.draw_ms, 1),
            "preview_ms": round(self.preview_ms, 1),
            "fps": round(self.inference_fps, 1),
        }

    def _format_status(self, stats_payload, detection_count):
        speed_range = stats_payload.get("speed_range", (0.0, 0.0))
        current_time = datetime.now().strftime("%H:%M:%S")
        model_ms = stats_payload.get("model_ms", 0.0)
        analysis_ms = stats_payload.get("analysis_ms", 0.0)
        draw_ms = stats_payload.get("draw_ms", 0.0)
        runtime_parts = [f"设备：{self.device_display_name}"]
        if self.source_mode == "video" and self.video_resolution_profile:
            runtime_parts.append(f"工作视频：{self.video_resolution_profile}")
        return (
            f"时间：{current_time} | 显示：{stats_payload.get('display_fps', 0.0):.1f} FPS | "
            f"模型：{stats_payload.get('inference_fps', 0.0):.1f} FPS | "
            f"检测：{detection_count} | 当前车辆：{stats_payload.get('current_vehicle_count', 0)} | "
            f"去重总车辆：{stats_payload.get('unique_vehicle_total', 0)} | "
            f"跟踪：{stats_payload.get('track_count', 0)} | "
            f"速度：{speed_range[0]:.1f}-{speed_range[1]:.1f} km/h | "
            f"模型耗时：{model_ms:.1f} ms | 分析：{analysis_ms:.1f} ms | 绘制：{draw_ms:.1f} ms | "
            + " | ".join(runtime_parts)
        )

    def _adapt_frame_stride(self, inference_elapsed):
        budget = self.target_frame_interval if self.target_frame_interval > 0 else (1.0 / 30.0)
        required_stride = max(1, int(np.ceil(inference_elapsed / max(budget, 1e-3))))
        required_stride = min(required_stride, self.max_frame_stride)

        if required_stride > self.frame_stride:
            self.frame_stride = required_stride
            self._stride_recovery_hits = 0
            return

        if required_stride < self.frame_stride:
            self._stride_recovery_hits += 1
            if self._stride_recovery_hits >= self._stride_recovery_threshold:
                self.frame_stride = max(1, self.frame_stride - 1)
                self._stride_recovery_hits = 0
        else:
            self._stride_recovery_hits = 0

    def run(self):
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
                    current_time = datetime.now().strftime("%H:%M:%S")
                    self.status_update.emit(f"时间：{current_time} | 已暂停")
                    self.msleep(50)
                    continue

                self.frame_reader.set_paused(False)
                frame_packet = self.frame_reader.get_latest_frame(last_processed_sequence)
                if frame_packet is None:
                    self.msleep(1)
                    continue

                frame_sequence, frame, decode_ms = frame_packet
                frame_step = max(1, frame_sequence - last_processed_sequence)
                if last_processed_sequence > 0 and frame_step < self.frame_stride:
                    self.msleep(1)
                    continue
                last_processed_sequence = frame_sequence

                if self.source_mode == "camera":
                    frame_height, frame_width = frame.shape[:2]
                    if frame_width != self.source_width or frame_height != self.source_height:
                        self.raw_source_width = max(1, frame_width)
                        self.raw_source_height = max(1, frame_height)
                        self.source_width = self.raw_source_width
                        self.source_height = self.raw_source_height
                        self.inference_imgsz = self._resolve_inference_imgsz()

                try:
                    processing_start = time.perf_counter()
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

                    speed_range = self.speed_estimator.get_speed_range(tracks) if self.enable_speed else (0.0, 0.0)
                    warning_count = len(collision_warnings) + len(zone_violations)

                    self.last_render_payload = {
                        "detections": detections,
                        "tracks": tracks,
                        "speed_infos": speed_infos,
                        "collision_warnings": collision_warnings,
                        "zone_violations": zone_violations,
                        "class_counts": class_counts,
                        "up_counts": up_counts,
                        "down_counts": down_counts,
                        "unique_vehicle_total": unique_vehicle_total,
                        "current_vehicle_count": current_vehicle_count,
                    }
                    self.last_stats_payload = self._build_stats_payload(
                        class_counts,
                        up_counts,
                        down_counts,
                        unique_vehicle_total,
                        current_vehicle_count,
                        warning_count,
                        track_count,
                        speed_range,
                    )

                    now = time.perf_counter()
                    warning_refresh_due = (
                        (run_collision_analysis and track_count > 1)
                        or (run_zone_analysis and bool(self.zone_detector.zones))
                        or self._inference_cycle == 1
                    )
                    if warning_refresh_due and (now - self._last_warning_emit_at >= self.warning_emit_interval):
                        all_warnings = []
                        for warning in collision_warnings:
                            all_warnings.append(
                                f"鈿?纰版挒椋庨櫓: {warning.class_name_1}#{warning.track_id_1} - "
                                f"{warning.class_name_2}#{warning.track_id_2} ({warning.distance}m)"
                            )
                        for violation in zone_violations:
                            all_warnings.append(
                                f"馃毇 杩濊: {violation.class_name}#{violation.track_id} 杩涘叆 {violation.zone_name}"
                            )
                        if len(all_warnings) > self.warning_display_limit:
                            hidden_count = len(all_warnings) - self.warning_display_limit
                            all_warnings = all_warnings[:self.warning_display_limit] + [
                                f"... 鍏?{warning_count} 鏉￠璀︼紝鍏朵綑 {hidden_count} 鏉℃湭灞曠ず"
                            ]
                        self.warning_update.emit(all_warnings)
                        self._last_warning_emit_at = now

                    analysis_elapsed = time.perf_counter() - analysis_start
                    self._tune_inference_imgsz(model_elapsed)

                    draw_start = time.perf_counter()
                    annotated_frame = self._draw_results(working_frame, **self.last_render_payload)
                    draw_elapsed = time.perf_counter() - draw_start
                    preview_ms = self._store_latest_preview(annotated_frame, frame_sequence)
                    total_pipeline_elapsed = (time.perf_counter() - processing_start) + (decode_ms / 1000.0)
                    self._adapt_frame_stride(total_pipeline_elapsed)

                    processed_frame_count += 1
                    model_frame_count += 1
                    decode_time_total += decode_ms / 1000.0
                    model_time_total += model_elapsed
                    analysis_time_total += analysis_elapsed
                    draw_time_total += draw_elapsed
                    preview_time_total += preview_ms / 1000.0

                    now = time.perf_counter()
                    elapsed = now - fps_window_start
                    if elapsed >= 1.0:
                        self.display_fps = processed_frame_count / elapsed
                        self.inference_fps = (
                            model_frame_count / model_time_total
                            if model_time_total > 0
                            else 0.0
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
                        decode_time_total = 0.0
                        model_time_total = 0.0
                        analysis_time_total = 0.0
                        draw_time_total = 0.0
                        preview_time_total = 0.0
                        fps_window_start = now

                    current_stats = dict(self.last_stats_payload)
                    current_stats["display_fps"] = round(self.display_fps, 1)
                    current_stats["inference_fps"] = round(self.inference_fps, 1)
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
                    self.status_update.emit(f"妫€娴嬮敊璇細{str(e)}")
                    self.msleep(5)
        finally:
            if self.frame_reader:
                self.frame_reader.stop()
                self.frame_reader = None
            self.status_update.emit("妫€娴嬪凡鍋滄")
        return

        cap = self._open_capture(self.source)
        if self.source_mode == "camera":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.status_update.emit(f"无法打开视频源：{self.source}")
            return

        actual_fps = self._sanitize_fps(cap.get(cv2.CAP_PROP_FPS) or self.source_fps)
        if self.source_mode == "camera":
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.source_width)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.source_height)
            self.raw_source_width = max(1, width)
            self.raw_source_height = max(1, height)
            self.source_width = self.raw_source_width
            self.source_height = self.raw_source_height
            self.inference_imgsz = self._resolve_inference_imgsz()
        self.source_fps = actual_fps
        self.target_frame_interval = 1.0 / self.source_fps if self.source_mode == "video" and self.source_fps > 0 else 0.0
        self.speed_estimator.set_fps(self.source_fps)
        self._warmup_model()

        fps_window_start = time.perf_counter()
        display_frame_count = 0
        model_frame_count = 0
        model_time_total = 0.0
        analysis_time_total = 0.0
        draw_time_total = 0.0
        frame_index = 0
        presentation_deadline = time.perf_counter()

        while self.running:
            if self.paused:
                current_time = datetime.now().strftime("%H:%M:%S")
                self.status_update.emit(f"时间：{current_time} | 已暂停")
                self.msleep(50)
                continue

            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if self.source_mode == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    presentation_deadline = time.perf_counter()
                    continue
                self.msleep(5)
                continue

            frame_index += 1
            frames_advanced = 1

            try:
                processing_start = time.perf_counter()
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
                    effective_tracking_fps = self.source_fps / max(1, self.frame_stride)
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

                speed_range = self.speed_estimator.get_speed_range(tracks) if self.enable_speed else (0.0, 0.0)
                warning_count = len(collision_warnings) + len(zone_violations)

                self.last_render_payload = {
                    "detections": detections,
                    "tracks": tracks,
                    "speed_infos": speed_infos,
                    "collision_warnings": collision_warnings,
                    "zone_violations": zone_violations,
                    "class_counts": class_counts,
                    "up_counts": up_counts,
                    "down_counts": down_counts,
                    "unique_vehicle_total": unique_vehicle_total,
                    "current_vehicle_count": current_vehicle_count,
                }
                self.last_stats_payload = self._build_stats_payload(
                    class_counts,
                    up_counts,
                    down_counts,
                    unique_vehicle_total,
                    current_vehicle_count,
                    warning_count,
                    track_count,
                    speed_range,
                )

                now = time.perf_counter()
                warning_refresh_due = (
                    (run_collision_analysis and track_count > 1)
                    or (run_zone_analysis and bool(self.zone_detector.zones))
                    or self._inference_cycle == 1
                )
                if warning_refresh_due and (now - self._last_warning_emit_at >= self.warning_emit_interval):
                    all_warnings = []
                    for warning in collision_warnings:
                        all_warnings.append(
                            f"⚠ 碰撞风险: {warning.class_name_1}#{warning.track_id_1} - "
                            f"{warning.class_name_2}#{warning.track_id_2} ({warning.distance}m)"
                        )
                    for violation in zone_violations:
                        all_warnings.append(
                            f"🚫 违规: {violation.class_name}#{violation.track_id} 进入 {violation.zone_name}"
                        )
                    if len(all_warnings) > self.warning_display_limit:
                        hidden_count = len(all_warnings) - self.warning_display_limit
                        all_warnings = all_warnings[:self.warning_display_limit] + [
                            f"... 共 {warning_count} 条预警，其余 {hidden_count} 条未展示"
                        ]
                    self.warning_update.emit(all_warnings)
                    self._last_warning_emit_at = now

                analysis_elapsed = time.perf_counter() - analysis_start
                self._tune_inference_imgsz(model_elapsed)

                draw_start = time.perf_counter()
                annotated_frame = self._draw_results(working_frame, **self.last_render_payload)
                draw_elapsed = time.perf_counter() - draw_start
                total_pipeline_elapsed = time.perf_counter() - processing_start
                self._adapt_frame_stride(total_pipeline_elapsed)

                display_frame_count += 1
                model_frame_count += 1
                model_time_total += model_elapsed
                analysis_time_total += analysis_elapsed
                draw_time_total += draw_elapsed

                now = time.perf_counter()
                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    self.display_fps = display_frame_count / elapsed
                    self.inference_fps = (
                        model_frame_count / model_time_total
                        if model_time_total > 0
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
                        draw_time_total * 1000.0 / display_frame_count
                        if display_frame_count > 0
                        else 0.0
                    )
                    display_frame_count = 0
                    model_frame_count = 0
                    model_time_total = 0.0
                    analysis_time_total = 0.0
                    draw_time_total = 0.0
                    fps_window_start = now

                current_stats = dict(self.last_stats_payload)
                current_stats["display_fps"] = round(self.display_fps, 1)
                current_stats["inference_fps"] = round(self.inference_fps, 1)
                current_stats["model_ms"] = round(self.model_ms, 1)
                current_stats["analysis_ms"] = round(self.analysis_ms, 1)
                current_stats["draw_ms"] = round(self.draw_ms, 1)
                current_stats["fps"] = round(self.inference_fps, 1)
                detection_count = len(self.last_render_payload["detections"])

                if now - self._last_stats_emit_at >= self.stats_emit_interval:
                    self.stats_update.emit(current_stats)
                    self.status_update.emit(self._format_status(current_stats, detection_count))
                    self._last_stats_emit_at = now
                self.frame_ready.emit(annotated_frame)

            except Exception as e:
                self.status_update.emit(f"检测错误：{str(e)}")

            if self.source_mode in {"video", "camera"} and self.frame_stride > 1:
                skipped_by_stride = 0
                for _ in range(self.frame_stride - 1):
                    if not cap.grab():
                        break
                    skipped_by_stride += 1
                frame_index += skipped_by_stride
                frames_advanced += skipped_by_stride

            if self.target_frame_interval > 0:
                presentation_deadline += self.target_frame_interval * frames_advanced
                now = time.perf_counter()
                if now < presentation_deadline:
                    time.sleep(presentation_deadline - now)
                elif self.sequential_video_playback:
                    presentation_deadline = now
                else:
                    lag = now - presentation_deadline
                    frames_to_drop = min(
                        self.max_frame_stride,
                        int(np.ceil(lag / self.target_frame_interval))
                    )
                    if frames_to_drop > 0:
                        dropped = 0
                        for _ in range(frames_to_drop):
                            if not cap.grab():
                                break
                            dropped += 1
                        frame_index += dropped
                        presentation_deadline += dropped * self.target_frame_interval
            elif self.source_mode == "camera":
                loop_elapsed = time.perf_counter() - loop_start
                if loop_elapsed < 0.001:
                    time.sleep(0.001)

        cap.release()
        self.status_update.emit("检测已停止")

    def _draw_results(
        self,
        frame,
        detections,
        tracks,
        speed_infos,
        collision_warnings,
        zone_violations,
        class_counts,
        up_counts,
        down_counts,
        unique_vehicle_total,
        current_vehicle_count,
    ):
        """绘制所有检测结果"""
        output = frame
        fast_live_overlay = self.fast_live_overlay
        text_items = None if fast_live_overlay else []
        show_tracking_lines = self.enable_tracking and len(tracks) <= self.max_tracking_line_tracks
        collision_markers = collision_warnings[:self.max_warning_markers]
        zone_markers = zone_violations[:self.max_warning_markers]

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
        self._cached_speed_infos = []
        self._cached_collision_warnings = []
        self._cached_zone_violations = []
        self._inference_cycle = 0
        self._last_stats_emit_at = 0.0
        self._last_warning_emit_at = 0.0
        self.decode_ms = 0.0
        self.model_ms = 0.0
        self.analysis_ms = 0.0
        self.draw_ms = 0.0
        self.preview_ms = 0.0
        with self._preview_lock:
            self._latest_preview_image = None
            self._latest_preview_sequence = 0


class YOLOv8GUI(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()

        # 初始化模型
        self.model = None
        self.class_profile = DEFAULT_TRAFFIC_CLASS_PROFILE
        self.current_mode = None
        self.detection_thread = None

        # 绘制状态
        self.drawing_mode = None
        self.drawing_points = []

        # 图像缩放信息
        self.original_frame_size = None
        self.displayed_pixmap_rect = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        if hasattr(Qt, "PreciseTimer"):
            self.preview_timer.setTimerType(Qt.PreciseTimer)
        self.preview_timer.timeout.connect(self.refresh_preview_frame)
        self.preview_present_fps = 0.0
        self.preview_present_ms = 0.0
        self.preview_frame_count = 0
        self.preview_present_total = 0.0
        self.preview_window_start = 0.0
        self.last_preview_sequence = 0
        self.last_live_stats = None
        
        # 配置文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(current_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)

        self.init_model()
        self.init_ui()

    def init_model(self):
        """加载 YOLO 模型"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        model_paths = [
            os.path.join(project_root, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(project_root, "runs", "detect", "train", "weights", "last.pt"),
            os.path.join(project_root, "models", "yolov8n.pt"),
            os.path.join(project_root, "models", "yolo26n.pt"),
            "yolov8n.pt"
        ]

        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model = YOLO(path)
                    self._refresh_class_profile()
                    print(f"成功加载模型：{path}")
                    break
                except Exception as e:
                    print(f"加载失败 {path}: {e}")

        if self.model is None:
            QMessageBox.critical(
                None, "错误",
                "未找到模型！\n请将模型文件放在以下位置：\n"
                "runs/detect/train/weights/best.pt\n"
                "models/yolov8n.pt"
            )

    def _refresh_class_profile(self):
        model_names = getattr(self.model, "names", {}) if self.model is not None else {}
        profile = build_traffic_class_profile(model_names)
        if not profile.table_class_ids:
            profile = DEFAULT_TRAFFIC_CLASS_PROFILE
        self.class_profile = profile
        DrawingUtils.configure_class_profile(profile)

    def _format_model_label_text(self, model_name: str) -> str:
        category_text = "、".join(self.class_profile.table_display_names) or "未识别交通类"
        return f"当前模型: {model_name}\n交通类别: {category_text}"

    def _selected_device_preference(self):
        if hasattr(self, 'device_combo'):
            return self.device_combo.currentData() or "auto"
        return "auto"

    def _selected_video_resolution_profile(self):
        if hasattr(self, 'video_resolution_combo'):
            return self.video_resolution_combo.currentData() or "720p"
        return "720p"

    def _resolved_device_display_name(self, device_preference=None):
        _, _, device_name = resolve_runtime_device(device_preference or self._selected_device_preference())
        return device_name

    def _build_runtime_note_text(self, inference_fps=0.0):
        parts = [f"模型 {inference_fps:.1f} FPS", self._resolved_device_display_name()]
        if self.current_mode == "video":
            parts.append(self._selected_video_resolution_profile())
        return " | ".join(parts)

    def _sync_runtime_option_states(self):
        running = bool(self.detection_thread and self.detection_thread.isRunning())
        is_video_mode = self.current_mode == "video"
        gpu_available = torch.cuda.is_available()

        if hasattr(self, 'device_combo'):
            self.device_combo.setEnabled(not running)
            gpu_text = "GPU" if gpu_available else "GPU（不可用）"
            self.device_combo.setItemText(2, gpu_text)
            gpu_item = self.device_combo.model().item(2)
            if gpu_item is not None:
                gpu_item.setEnabled(gpu_available)
            if not gpu_available and self.device_combo.currentData() == "gpu":
                self.device_combo.setCurrentIndex(0)

        if hasattr(self, 'video_resolution_combo'):
            self.video_resolution_combo.setEnabled(is_video_mode and not running)
            self.video_resolution_label.setEnabled(is_video_mode)
            tooltip = "仅视频文件模式支持实时缩放预处理"
            self.video_resolution_combo.setToolTip(tooltip)
            self.video_resolution_label.setToolTip(tooltip)

    def _probe_source_info(self, source, source_mode):
        if source_mode == "image":
            image = cv2.imread(source)
            if image is None:
                return {"width": 1280, "height": 720, "fps": 0.0}
            height, width = image.shape[:2]
            return {"width": width, "height": height, "fps": 0.0}

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {"width": 1280, "height": 720, "fps": 30.0}

        if source_mode == "camera":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if width <= 0 or height <= 0:
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]

        cap.release()
        return {
            "width": width if width > 0 else 1280,
            "height": height if height > 0 else 720,
            "fps": fps,
        }

    def _prepare_video_viewport(self, source_info):
        width = int(source_info.get("width", 1280) or 1280)
        height = int(source_info.get("height", 720) or 720)
        self.video_shell.set_source_size(width, height)
        self.original_frame_size = (width, height)
        self.on_video_display_resized(self.image_label.width(), self.image_label.height())

    def on_video_display_resized(self, width, height):
        if self.detection_thread:
            self.detection_thread.set_preview_target_size(width, height)

    def start_preview_refresh(self):
        """启动最新预览帧轮询，只展示最新一帧。"""
        self.preview_present_fps = 0.0
        self.preview_present_ms = 0.0
        self.preview_frame_count = 0
        self.preview_present_total = 0.0
        self.preview_window_start = time.perf_counter()
        self.last_preview_sequence = 0
        self.last_live_stats = None
        self.preview_timer.start()

    def stop_preview_refresh(self):
        """停止预览轮询并清空轮询统计。"""
        self.preview_timer.stop()
        self.preview_present_fps = 0.0
        self.preview_present_ms = 0.0
        self.preview_frame_count = 0
        self.preview_present_total = 0.0
        self.preview_window_start = 0.0
        self.last_preview_sequence = 0
        self.last_live_stats = None

    def _build_runtime_metrics_note(self, stats):
        inference_fps = stats.get('inference_fps', stats.get('fps', 0.0))
        decode_ms = stats.get('decode_ms', 0.0)
        model_ms = stats.get('model_ms', 0.0)
        analysis_ms = stats.get('analysis_ms', 0.0)
        draw_ms = stats.get('draw_ms', 0.0)
        preview_ms = stats.get('preview_ms', 0.0)
        present_ms = self.preview_present_ms if self.detection_thread else 0.0

        runtime_parts = [
            f"模型 {inference_fps:.1f} FPS",
            (
                f"解{decode_ms:.1f}/模{model_ms:.1f}/析{analysis_ms:.1f}/"
                f"绘{draw_ms:.1f}/预{preview_ms:.1f}/显{present_ms:.1f} ms"
            ),
        ]
        if self.detection_thread:
            runtime_parts.append(self.detection_thread.device_display_name)
            if self.detection_thread.source_mode == "video" and self.detection_thread.video_resolution_profile:
                runtime_parts.append(self.detection_thread.video_resolution_profile)
        return " | ".join(runtime_parts)

    def _build_live_status_text(self, stats):
        current_time = datetime.now().strftime("%H:%M:%S")
        display_fps = self.preview_present_fps if self.detection_thread else stats.get('display_fps', 0.0)
        inference_fps = stats.get('inference_fps', stats.get('fps', 0.0))
        decode_ms = stats.get('decode_ms', 0.0)
        model_ms = stats.get('model_ms', 0.0)
        analysis_ms = stats.get('analysis_ms', 0.0)
        draw_ms = stats.get('draw_ms', 0.0)
        preview_ms = stats.get('preview_ms', 0.0)
        present_ms = self.preview_present_ms if self.detection_thread else 0.0
        speed_range = stats.get('speed_range', (0.0, 0.0))

        status_parts = [
            f"时间：{current_time}",
            f"显示：{display_fps:.1f} FPS",
            f"模型：{inference_fps:.1f} FPS",
            (
                f"decode/model/analysis/draw/preview/present "
                f"{decode_ms:.1f}/{model_ms:.1f}/{analysis_ms:.1f}/"
                f"{draw_ms:.1f}/{preview_ms:.1f}/{present_ms:.1f} ms"
            ),
            f"当前车辆：{stats.get('current_vehicle_count', 0)}",
            f"去重总车辆：{stats.get('unique_vehicle_total', 0)}",
            f"跟踪：{stats.get('track_count', 0)}",
            f"速度：{speed_range[0]:.1f}-{speed_range[1]:.1f} km/h",
        ]
        if self.detection_thread:
            status_parts.append(f"设备：{self.detection_thread.device_display_name}")
            if self.detection_thread.source_mode == "video" and self.detection_thread.video_resolution_profile:
                status_parts.append(f"视频档位：{self.detection_thread.video_resolution_profile}")
        return " | ".join(status_parts)

    def _apply_live_runtime_indicators(self, stats):
        display_fps = self.preview_present_fps if self.detection_thread else stats.get('display_fps', 0.0)
        self.runtime_value_label.setText(f"{display_fps:.1f}")
        if self.detection_thread:
            self.runtime_note_label.setText(self._build_runtime_metrics_note(stats))
            self.status_label.setText(self._build_live_status_text(stats))
        else:
            self.runtime_note_label.setText("单图结果")

    def refresh_preview_frame(self):
        """主线程主动拉取最新预览帧，丢弃所有过时帧。"""
        if not self.detection_thread:
            return
        if not self.detection_thread.isRunning():
            self.stop_preview_refresh()
            self.set_running_state(False)
            self.pause_btn.setText("暂停")
            return

        preview_packet = self.detection_thread.get_latest_preview(self.last_preview_sequence)
        now = time.perf_counter()
        if preview_packet is None:
            if self.preview_window_start > 0 and now - self.preview_window_start >= 0.5 and self.preview_frame_count == 0:
                self.preview_present_fps = 0.0
                self.preview_present_ms = 0.0
                self.preview_window_start = now
                if self.last_live_stats:
                    self._apply_live_runtime_indicators(self.last_live_stats)
            return

        sequence, q_image, source_size = preview_packet
        if q_image is None or q_image.isNull():
            return

        present_start = time.perf_counter()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        present_elapsed = time.perf_counter() - present_start

        self.last_preview_sequence = sequence
        self.original_frame_size = source_size

        target_w = q_image.width()
        target_h = q_image.height()
        offset_x = max(0, (self.image_label.width() - target_w) // 2)
        offset_y = max(0, (self.image_label.height() - target_h) // 2)
        self.displayed_pixmap_rect = (offset_x, offset_y, target_w, target_h)

        if self.preview_window_start <= 0.0:
            self.preview_window_start = now
        self.preview_frame_count += 1
        self.preview_present_total += present_elapsed

        elapsed = max(1e-6, now - self.preview_window_start)
        if self.preview_frame_count > 0 and elapsed >= 0.25:
            self.preview_present_fps = self.preview_frame_count / elapsed
            self.preview_present_ms = self.preview_present_total * 1000.0 / self.preview_frame_count
            if elapsed >= 1.0:
                self.preview_frame_count = 0
                self.preview_present_total = 0.0
                self.preview_window_start = now

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("YOLOv8 校园交通检测系统")
        self.setMinimumSize(1366, 768)

        # 应用全局样式
        self.setStyleSheet(self._load_stylesheet())

        # 主窗口布局
        main_widget = QWidget()
        main_widget.setObjectName("mainCentral")
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 左侧：视频显示区域 + KPI + 统计信息
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        top_visual_widget = QWidget()
        top_visual_layout = QHBoxLayout(top_visual_widget)
        top_visual_layout.setContentsMargins(0, 0, 0, 0)
        top_visual_layout.setSpacing(14)

        left_kpi_column = QWidget()
        left_kpi_column.setFixedWidth(148)
        left_kpi_column.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_kpi_layout = QVBoxLayout(left_kpi_column)
        left_kpi_layout.setContentsMargins(0, 0, 0, 0)
        left_kpi_layout.setSpacing(8)

        center_visual_column = QWidget()
        center_visual_layout = QVBoxLayout(center_visual_column)
        center_visual_layout.setContentsMargins(0, 0, 0, 0)
        center_visual_layout.setSpacing(8)

        self.video_shell = StableVideoViewport()
        self.video_shell.setObjectName("videoDisplayShell")
        self.video_shell.setMinimumSize(680, 420)
        self.video_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label = self.video_shell.video_label
        self.image_label.mouse_clicked.connect(self.on_video_clicked)
        self.image_label.mouse_moved.connect(self.on_video_mouse_moved)
        self.video_shell.display_size_changed.connect(self.on_video_display_resized)
        center_visual_layout.addWidget(self.video_shell, 1)

        self.draw_hint_label = QLabel("绘制提示：未启用绘制工具")
        self.draw_hint_label.setObjectName("drawHint")
        self.draw_hint_label.setWordWrap(True)
        self.draw_hint_label.setMaximumHeight(32)
        center_visual_layout.addWidget(self.draw_hint_label)

        kpi_card_1, self.unique_vehicle_value_label, self.unique_vehicle_note_label = self._create_kpi_card(
            "去重总车辆", "metricCardPrimary", "0", "会话累计"
        )
        kpi_card_2, self.current_vehicle_value_label, self.current_vehicle_note_label = self._create_kpi_card(
            "当前画面车辆", "metricCardNeutral", "0", "稳定轨迹"
        )
        kpi_card_3, self.warning_count_value_label, self.warning_count_note_label = self._create_kpi_card(
            "预警数", "metricCardWarning", "0", "碰撞 / 违规"
        )
        kpi_card_4, self.runtime_value_label, self.runtime_note_label = self._create_kpi_card(
            "显示 FPS", "metricCardStatus", "0.0", "等待开始检测"
        )

        for card in (kpi_card_1, kpi_card_2, kpi_card_3, kpi_card_4):
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        left_kpi_layout.addStretch(1)
        left_kpi_layout.addWidget(kpi_card_1)
        left_kpi_layout.addWidget(kpi_card_2)
        left_kpi_layout.addWidget(kpi_card_3)
        left_kpi_layout.addWidget(kpi_card_4)
        left_kpi_layout.addStretch(1)

        top_visual_layout.addWidget(left_kpi_column, 0)
        top_visual_layout.addWidget(center_visual_column, 1)
        left_layout.addWidget(top_visual_widget, 1)

        bottom_info_layout = QHBoxLayout()
        bottom_info_layout.setSpacing(8)

        stats_group = QGroupBox("分类统计")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(10, 22, 10, 10)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["类型", "数量", "上行", "下行"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setMaximumHeight(112)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        bottom_info_layout.addWidget(stats_group, 5)

        warning_group = QGroupBox("预警与事件")
        warning_layout = QVBoxLayout()
        warning_layout.setContentsMargins(10, 22, 10, 10)

        self.warning_text = QPlainTextEdit()
        self.warning_text.setReadOnly(True)
        self.warning_text.setMaximumHeight(112)
        self.warning_text.setObjectName("warningText")
        self.warning_text.setPlaceholderText("检测开始后，这里会显示碰撞风险和违规事件。")
        warning_layout.addWidget(self.warning_text)

        warning_group.setLayout(warning_layout)
        bottom_info_layout.addWidget(warning_group, 4)

        left_layout.addLayout(bottom_info_layout)

        self.status_label = QLabel("就绪 | 等待开始检测...")
        self.status_label.setObjectName("statusBar")
        self.status_label.setMinimumHeight(24)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        main_layout.addWidget(left_widget, 1)

        # 右侧：控制面板
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(0, 0, 0, 0)

        input_group = QGroupBox("模型与输入")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)
        input_layout.setContentsMargins(10, 22, 10, 10)

        model_section = QLabel("当前模型")
        model_section.setObjectName("sectionLabel")
        input_layout.addWidget(model_section)

        self.model_path_label = QLabel("当前模型：未加载")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setObjectName("successText")
        if self.model:
            model_name = os.path.basename(str(self.model.ckpt_path)) if hasattr(self.model, 'ckpt_path') else "未知"
            self.model_path_label.setText(self._format_model_label_text(model_name))
        input_layout.addWidget(self.model_path_label)

        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.setObjectName("warningBtn")
        self.select_model_btn.setFixedHeight(34)
        self.select_model_btn.clicked.connect(self.select_model_file)
        input_layout.addWidget(self.select_model_btn)

        mode_section = QLabel("检测模式")
        mode_section.setObjectName("sectionLabel")
        input_layout.addWidget(mode_section)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_radio = QRadioButton("摄像头")
        self.video_radio = QRadioButton("视频文件")
        self.image_radio = QRadioButton("图片")

        self.camera_radio.toggled.connect(lambda c: self.switch_mode('camera') if c else None)
        self.video_radio.toggled.connect(lambda c: self.switch_mode('video') if c else None)
        self.image_radio.toggled.connect(lambda c: self.switch_mode('image') if c else None)

        mode_layout.addWidget(self.camera_radio)
        mode_layout.addWidget(self.video_radio)
        mode_layout.addWidget(self.image_radio)
        input_layout.addLayout(mode_layout)

        runtime_section = QLabel("推理设置")
        runtime_section.setObjectName("sectionLabel")
        input_layout.addWidget(runtime_section)

        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.addWidget(QLabel("推理设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU", "gpu")
        self.device_combo.setCurrentIndex(0)
        self.device_combo.currentIndexChanged.connect(lambda _: self._sync_runtime_option_states())
        device_row.addWidget(self.device_combo, 1)
        input_layout.addLayout(device_row)

        video_resolution_row = QHBoxLayout()
        video_resolution_row.setContentsMargins(0, 0, 0, 0)
        self.video_resolution_label = QLabel("视频预处理:")
        video_resolution_row.addWidget(self.video_resolution_label)
        self.video_resolution_combo = QComboBox()
        self.video_resolution_combo.addItem("720p", "720p")
        self.video_resolution_combo.addItem("1080p", "1080p")
        self.video_resolution_combo.setCurrentIndex(0)
        self.video_resolution_combo.currentIndexChanged.connect(lambda _: self._sync_runtime_option_states())
        video_resolution_row.addWidget(self.video_resolution_combo, 1)
        input_layout.addLayout(video_resolution_row)

        feature_section = QLabel("功能开关")
        feature_section.setObjectName("sectionLabel")
        input_layout.addWidget(feature_section)

        feature_layout = QGridLayout()
        feature_layout.setHorizontalSpacing(8)
        feature_layout.setVerticalSpacing(6)
        feature_layout.setContentsMargins(0, 0, 0, 0)

        self.tracking_cb = QCheckBox("目标跟踪")
        self.tracking_cb.setChecked(True)
        self.speed_cb = QCheckBox("速度检测")
        self.speed_cb.setChecked(True)
        self.collision_cb = QCheckBox("碰撞预警")
        self.collision_cb.setChecked(True)
        self.zone_cb = QCheckBox("违规检测")
        self.zone_cb.setChecked(True)

        feature_layout.addWidget(self.tracking_cb, 0, 0)
        feature_layout.addWidget(self.speed_cb, 0, 1)
        feature_layout.addWidget(self.collision_cb, 1, 0)
        feature_layout.addWidget(self.zone_cb, 1, 1)
        input_layout.addLayout(feature_layout)

        input_group.setLayout(input_layout)
        right_layout.addWidget(input_group)

        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(10, 22, 10, 10)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_row.addWidget(self.conf_slider)
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setMinimumWidth(40)
        conf_row.addWidget(self.conf_value_label)
        param_layout.addLayout(conf_row)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IOU:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(10)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.on_iou_changed)
        iou_row.addWidget(self.iou_slider)
        self.iou_value_label = QLabel("0.45")
        self.iou_value_label.setMinimumWidth(40)
        iou_row.addWidget(self.iou_value_label)
        param_layout.addLayout(iou_row)

        distance_row = QHBoxLayout()
        distance_row.addWidget(QLabel("安全距离(m):"))
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1.0, 50.0)
        self.distance_spin.setValue(5.0)
        self.distance_spin.setSingleStep(0.5)
        self.distance_spin.valueChanged.connect(self.on_distance_changed)
        distance_row.addWidget(self.distance_spin)
        distance_row.addStretch()
        param_layout.addLayout(distance_row)

        calib_row = QHBoxLayout()
        calib_row.addWidget(QLabel("像素/米:"))
        self.calib_spin = QSpinBox()
        self.calib_spin.setRange(1, 500)
        self.calib_spin.setValue(50)
        self.calib_spin.valueChanged.connect(self.on_calibration_changed)
        calib_row.addWidget(self.calib_spin)
        calib_row.addStretch()
        param_layout.addLayout(calib_row)

        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)

        draw_group = QGroupBox("绘制工具")
        draw_layout = QVBoxLayout()
        draw_layout.setSpacing(8)
        draw_layout.setContentsMargins(10, 22, 10, 10)

        self.draw_state_label = QLabel("状态：未启用绘制工具")
        self.draw_state_label.setObjectName("hintText")
        self.draw_state_label.setWordWrap(True)
        draw_layout.addWidget(self.draw_state_label)

        draw_button_layout = QGridLayout()
        draw_button_layout.setHorizontalSpacing(6)
        draw_button_layout.setVerticalSpacing(6)
        draw_button_layout.setContentsMargins(0, 0, 0, 0)

        self.draw_line_btn = QPushButton("绘制检测线")
        self.draw_line_btn.setObjectName("infoBtn")
        self.draw_line_btn.setFixedHeight(34)
        self.draw_line_btn.clicked.connect(self.start_draw_line)
        draw_button_layout.addWidget(self.draw_line_btn, 0, 0)

        self.draw_zone_btn = QPushButton("绘制违规区域")
        self.draw_zone_btn.setObjectName("infoBtn")
        self.draw_zone_btn.setFixedHeight(34)
        self.draw_zone_btn.clicked.connect(self.start_draw_zone)
        draw_button_layout.addWidget(self.draw_zone_btn, 0, 1)

        self.clear_zones_btn = QPushButton("清空线/区域")
        self.clear_zones_btn.setObjectName("dangerBtn")
        self.clear_zones_btn.setFixedHeight(34)
        self.clear_zones_btn.clicked.connect(self.clear_zones)
        draw_button_layout.addWidget(self.clear_zones_btn, 1, 0, 1, 2)
        draw_layout.addLayout(draw_button_layout)

        draw_group.setLayout(draw_layout)
        right_layout.addWidget(draw_group)

        control_group = QGroupBox("运行控制")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(10, 22, 10, 10)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setFixedHeight(38)
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setObjectName("infoBtn")
        self.pause_btn.setFixedHeight(38)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setFixedHeight(38)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        right_layout.addStretch(1)
        main_layout.addWidget(right_panel, 0)

        self._init_stats_tables()
        self.reset_metrics_display()
        self._sync_runtime_option_states()

    def _create_kpi_card(self, title, object_name, value="0", note=""):
        """创建 KPI 指标卡。"""
        card = QFrame()
        card.setObjectName(object_name)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("metricCaption")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        note_label = QLabel(note)
        note_label.setObjectName("metricNote")
        note_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(note_label)
        layout.addStretch(1)

        return card, value_label, note_label

    def set_draw_hint(self, message):
        """同步更新绘制提示。"""
        text = f"绘制提示：{message}"
        if hasattr(self, 'draw_hint_label'):
            self.draw_hint_label.setText(text)
        if hasattr(self, 'draw_state_label'):
            self.draw_state_label.setText(f"状态：{message}")

    def reset_metrics_display(self):
        """重置 KPI 与运行提示。"""
        if hasattr(self, 'unique_vehicle_value_label'):
            self.unique_vehicle_value_label.setText("0")
            self.unique_vehicle_note_label.setText("会话累计")
            self.current_vehicle_value_label.setText("0")
            self.current_vehicle_note_label.setText("已确认轨迹")
            self.warning_count_value_label.setText("0")
            self.warning_count_note_label.setText("碰撞 / 违规")
            self.runtime_value_label.setText("0.0")
            self.runtime_note_label.setText("等待开始检测")
        self.set_draw_hint("未启用绘制工具")

    def _log_runtime_environment(self, device_preference=None, video_resolution_profile=None):
        """记录当前 GUI 实际使用的解释器和推理设备。"""
        selected_device = device_preference or self._selected_device_preference()
        resolved_device = self._resolved_device_display_name(selected_device)
        runtime_parts = [
            f"Python: {sys.executable}",
            f"Torch: {torch.__version__}",
            f"DevicePref: {selected_device}",
            f"RuntimeDevice: {resolved_device}",
        ]
        if self.model and hasattr(self.model, 'ckpt_path'):
            runtime_parts.append(f"Model: {os.path.basename(str(self.model.ckpt_path))}")
        if torch.cuda.is_available():
            runtime_parts.append(f"CUDA: {torch.version.cuda}")
            runtime_parts.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            runtime_parts.append("CUDA: False")
        if self.current_mode == "video":
            runtime_parts.append(f"VideoProfile: {video_resolution_profile or self._selected_video_resolution_profile()}")
        self.log_message(" | ".join(runtime_parts))

    def _map_label_to_frame_coords(self, x, y):
        """将显示区域坐标映射回原始视频帧坐标。"""
        if not self.original_frame_size:
            return None

        if not self.displayed_pixmap_rect:
            return None

        pixmap_x, pixmap_y, display_w, display_h = self.displayed_pixmap_rect
        if display_w <= 0 or display_h <= 0:
            return None

        orig_w, orig_h = self.original_frame_size
        local_x = x - pixmap_x
        local_y = y - pixmap_y
        if local_x < 0 or local_y < 0:
            return None
        if local_x > display_w or local_y > display_h:
            return None

        scale_x = orig_w / display_w
        scale_y = orig_h / display_h

        real_x = int(local_x * scale_x)
        real_y = int(local_y * scale_y)
        real_x = max(0, min(orig_w - 1, real_x))
        real_y = max(0, min(orig_h - 1, real_y))
        return real_x, real_y

    def _init_stats_tables(self):
        """初始化统计表"""
        self.stats_table_class_names = list(self.class_profile.table_display_names)
        self.stats_table.setRowCount(len(self.stats_table_class_names))
        for i, name in enumerate(self.stats_table_class_names):
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem("0"))
            self.stats_table.setItem(i, 2, QTableWidgetItem("0"))
            self.stats_table.setItem(i, 3, QTableWidgetItem("0"))

    def _rebuild_stats_table(self):
        """按当前模型类别重建统计表。"""
        self.stats_table.clearContents()
        self._init_stats_tables()

    def _load_stylesheet(self):
        """从外部QSS文件加载样式表"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        qss_path = os.path.join(current_dir, "styles", "campus_ops_light.qss")
        
        try:
            with open(qss_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"警告: 样式文件未找到: {qss_path}")
            return ""
        except Exception as e:
            print(f"加载样式文件失败: {e}")
            return ""

    def switch_mode(self, mode):
        """切换检测模式"""
        self.stop_detection()
        self.current_mode = mode
        self.image_label.clear()
        self.image_label.setText("选择检测模式后点击开始")
        self.reset_metrics_display()
        self._sync_runtime_option_states()
        self.log_message(f"切换到 {mode} 模式")

    def select_model_file(self):
        """选择模型文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型文件", project_root,
            "模型文件 (*.pt);;所有文件 (*.*)"
        )

        if file_path:
            try:
                self.stop_detection()
                self.model = YOLO(file_path)
                self._refresh_class_profile()
                self._rebuild_stats_table()
                self.reset_metrics_display()
                model_name = os.path.basename(file_path)
                self.model_path_label.setText(self._format_model_label_text(model_name))
                self.log_message(f"已加载模型: {model_name}")
                QMessageBox.information(self, "成功", f"模型加载成功：{model_name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")

    def start_draw_line(self):
        """开始绘制检测线"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.drawing_mode = 'line'
            self.drawing_points = []
            self.detection_thread.set_drawing_mode('line')
            self.set_draw_hint("正在绘制检测线：左键选择两个点，右键可提前结束")
            QMessageBox.information(self, "提示", "请在视频画面上点击两个点绘制检测线\n右键点击完成绘制")

    def start_draw_zone(self):
        """开始绘制违规区域"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.drawing_mode = 'zone'
            self.drawing_points = []
            self.detection_thread.set_drawing_mode('zone')
            self.set_draw_hint("正在绘制违规区域：左键添加点，右键闭合区域")
            QMessageBox.information(self, "提示", "请在视频画面上点击绘制多边形区域\n至少需要3个点\n右键点击完成绘制")

    def clear_zones(self):
        """清除所有区域"""
        if self.detection_thread:
            self.detection_thread.clear_zones()
            self.line_counter_reset()
            self.detection_thread.set_drawing_mode(None)
            self.detection_thread.set_mouse_position(None)
        self.drawing_mode = None
        self.drawing_points = []
        self.set_draw_hint("已清空检测线和违规区域")
        self.log_message("已清空检测线和违规区域")

    def line_counter_reset(self):
        """重置检测线"""
        if self.detection_thread:
            self.detection_thread.line_counter.reset()

    def on_video_clicked(self, x, y, event):
        """视频画面点击事件"""
        if not self.drawing_mode:
            return

        mapped_point = self._map_label_to_frame_coords(x, y)
        if mapped_point is None:
            return

        if event.button() == Qt.LeftButton:
            self.drawing_points.append(mapped_point)
            if self.detection_thread:
                self.detection_thread.add_drawing_point(mapped_point)

            if self.drawing_mode == 'line' and len(self.drawing_points) >= 2:
                self.finish_drawing()

        elif event.button() == Qt.RightButton:
            self.finish_drawing()

    def on_video_mouse_moved(self, x, y):
        """视频画面鼠标移动事件，用于绘制预览。"""
        if not self.drawing_mode or not self.detection_thread:
            return

        mapped_point = self._map_label_to_frame_coords(x, y)
        self.detection_thread.set_mouse_position(mapped_point)

    def finish_drawing(self):
        """完成绘制"""
        if self.detection_thread and self.drawing_points:
            points = self.detection_thread.finish_drawing()
            self.log_message(f"绘制完成: {len(points)} 个点")
            
        self.drawing_mode = None
        self.drawing_points = []
        if self.detection_thread:
            self.detection_thread.set_mouse_position(None)
        self.set_draw_hint("未启用绘制工具")

    def start_detection(self):
        """开始检测"""
        if not self.model:
            QMessageBox.warning(self, "警告", "模型未加载！")
            return

        if not self.current_mode:
            QMessageBox.warning(self, "警告", "请先选择检测模式！")
            return

        if self._selected_device_preference() == "gpu" and not torch.cuda.is_available():
            QMessageBox.warning(self, "警告", "当前环境未检测到可用 GPU，请改用“自动”或“CPU”。")
            self._sync_runtime_option_states()
            return

        if self.current_mode == 'camera':
            self.start_camera()
        elif self.current_mode == 'video':
            self.start_video()
        elif self.current_mode == 'image':
            self.detect_image()

    def start_camera(self):
        """启动摄像头检测"""
        device_preference = self._selected_device_preference()
        self.log_message("启动摄像头...")
        self._log_runtime_environment(device_preference=device_preference)
        self.reset_metrics_display()
        self.warning_text.clear()
        source_info = self._probe_source_info(0, "camera")
        self._prepare_video_viewport(source_info)

        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0

        self.detection_thread = DetectionThread(
            self.model, 0, conf, iou,
            enable_tracking=self.tracking_cb.isChecked(),
            enable_speed=self.speed_cb.isChecked(),
            enable_collision=self.collision_cb.isChecked(),
            enable_zone=self.zone_cb.isChecked(),
            pixels_per_meter=self.calib_spin.value(),
            safe_distance=self.distance_spin.value(),
            class_profile=self.class_profile,
            source_mode="camera",
            source_info=source_info,
            device_preference=device_preference,
        )
        self.detection_thread.set_preview_target_size(self.image_label.width(), self.image_label.height())
        self.detection_thread.status_update.connect(self.status_label.setText)
        self.detection_thread.stats_update.connect(self.update_stats)
        self.detection_thread.warning_update.connect(self.update_warnings)
        self.detection_thread.start()
        self.start_preview_refresh()

        self.set_running_state(True)
        self.runtime_note_label.setText(self._build_runtime_note_text(0.0))
        self.set_draw_hint("未启用绘制工具")
        self.log_message("摄像头检测已开始")

    def start_video(self):
        """启动视频检测"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)"
        )

        if file_path:
            device_preference = self._selected_device_preference()
            video_resolution_profile = self._selected_video_resolution_profile()
            self.log_message(f"加载视频: {os.path.basename(file_path)}")
            self._log_runtime_environment(
                device_preference=device_preference,
                video_resolution_profile=video_resolution_profile,
            )
            self.reset_metrics_display()
            self.warning_text.clear()
            source_info = self._probe_source_info(file_path, "video")
            self._prepare_video_viewport(source_info)

            conf = self.conf_slider.value() / 100.0
            iou = self.iou_slider.value() / 100.0

            self.detection_thread = DetectionThread(
                self.model, file_path, conf, iou,
                enable_tracking=self.tracking_cb.isChecked(),
                enable_speed=self.speed_cb.isChecked(),
                enable_collision=self.collision_cb.isChecked(),
                enable_zone=self.zone_cb.isChecked(),
                pixels_per_meter=self.calib_spin.value(),
                safe_distance=self.distance_spin.value(),
                class_profile=self.class_profile,
                source_mode="video",
                source_info=source_info,
                device_preference=device_preference,
                video_resolution_profile=video_resolution_profile,
            )
            self.detection_thread.set_preview_target_size(self.image_label.width(), self.image_label.height())
            self.detection_thread.status_update.connect(self.status_label.setText)
            self.detection_thread.stats_update.connect(self.update_stats)
            self.detection_thread.warning_update.connect(self.update_warnings)
            self.detection_thread.start()
            self.start_preview_refresh()

            self.set_running_state(True)
            self.runtime_note_label.setText(self._build_runtime_note_text(0.0))
            self.set_draw_hint("未启用绘制工具")
            self.log_message("视频检测已开始")

    def detect_image(self):
        """检测图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要检测的图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )

        if not file_path:
            return

        self.stop_preview_refresh()
        self.log_message(f"检测图片: {os.path.basename(file_path)}")
        self.reset_metrics_display()
        self.warning_text.clear()

        try:
            self._log_runtime_environment(device_preference=self._selected_device_preference())
            image_device, image_half, image_device_name = resolve_runtime_device(
                self._selected_device_preference()
            )
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("图片读取失败")
            height, width = image.shape[:2]
            self._prepare_video_viewport({"width": width, "height": height, "fps": 0.0})
            conf = self.conf_slider.value() / 100.0
            iou = self.iou_slider.value() / 100.0

            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                verbose=False,
                classes=self.class_profile.classes_filter or None,
                device=image_device,
                half=image_half,
            )

            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    if cls not in self.class_profile.traffic_class_ids:
                        continue
                    conf_score = float(box.conf[0])
                    detections.append(([x1, y1, x2, y2], cls, conf_score))

            class_counter = ClassCounter()
            class_counter.set_class_profile(self.class_profile)
            class_counts = class_counter.count(detections)
            current_vehicle_count = sum(
                1 for _, cls, _ in detections
                if cls in self.class_profile.vehicle_class_ids
            )

            text_items = []
            annotated = image.copy()
            for bbox, class_id, confidence in detections:
                annotated = DrawingUtils.draw_detection_box(
                    annotated,
                    bbox,
                    class_id,
                    None,
                    confidence,
                    None,
                    text_items=text_items,
                )
            annotated = DrawingUtils.render_text_items(annotated, text_items)

            self.display_frame(annotated)
            self.update_stats({
                'unique_vehicle_total': current_vehicle_count,
                'current_vehicle_count': current_vehicle_count,
                'warning_count': 0,
                'class_counts': class_counts,
                'up_counts': {},
                'down_counts': {},
                'track_count': 0,
                'speed_range': (0.0, 0.0),
                'display_fps': 0.0,
                'inference_fps': 0.0,
                'fps': 0.0,
            })
            self.runtime_note_label.setText(f"单图 | {image_device_name}")
            self.update_warnings([])
            self.status_label.setText(
                f"图片检测完成 | 设备：{image_device_name} | 当前车辆：{current_vehicle_count} | 去重总车辆：{current_vehicle_count}"
            )
            self.log_message(f"检测到 {len(detections)} 个目标")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败：{str(e)}")

    def update_stats(self, stats):
        """更新统计信息"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.last_live_stats = dict(stats)
        else:
            self.last_live_stats = None

        unique_vehicle_total = stats.get('unique_vehicle_total', 0)
        current_vehicle_count = stats.get('current_vehicle_count', 0)
        warning_count = stats.get('warning_count', 0)
        display_fps = self.preview_present_fps if self.detection_thread else stats.get('display_fps', 0.0)
        inference_fps = stats.get('inference_fps', stats.get('fps', 0.0))
        speed_range = stats.get('speed_range', (0.0, 0.0))
        class_counts = stats.get('class_counts', {})
        up_counts = stats.get('up_counts', {})
        down_counts = stats.get('down_counts', {})

        self.unique_vehicle_value_label.setText(str(unique_vehicle_total))
        self.unique_vehicle_note_label.setText("本次会话累计")
        self.current_vehicle_value_label.setText(str(current_vehicle_count))
        self.current_vehicle_note_label.setText("已确认轨迹")
        self.warning_count_value_label.setText(str(warning_count))
        self.warning_count_note_label.setText("当前帧事件")
        self.runtime_value_label.setText(f"{display_fps:.1f}")
        self._apply_live_runtime_indicators(stats)

        for i, name in enumerate(self.stats_table_class_names):
            count = class_counts.get(name, 0)
            up = up_counts.get(name, 0)
            down = down_counts.get(name, 0)

            self.stats_table.item(i, 1).setText(str(count))
            self.stats_table.item(i, 2).setText(str(up))
            self.stats_table.item(i, 3).setText(str(down))

    def update_warnings(self, warnings):
        """更新预警信息"""
        if not warnings:
            self.warning_text.setPlainText("当前没有预警事件。")
            return
        self.warning_text.setPlainText("\n".join(warnings))

    def display_frame(self, frame):
        """显示帧"""
        try:
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame

                # 保存原始帧尺寸
                h, w, ch = rgb_frame.shape
                self.original_frame_size = (w, h)

                label_w = max(1, self.image_label.width())
                label_h = max(1, self.image_label.height())
                scale = min(label_w / max(1, w), label_h / max(1, h))
                target_w = max(1, int(w * scale))
                target_h = max(1, int(h * scale))

                if target_w != w or target_h != h:
                    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
                    display_rgb = cv2.resize(rgb_frame, (target_w, target_h), interpolation=interpolation)
                else:
                    display_rgb = rgb_frame

                bytes_per_line = display_rgb.shape[2] * display_rgb.shape[1]
                q_image = QImage(
                    display_rgb.data,
                    display_rgb.shape[1],
                    display_rgb.shape[0],
                    bytes_per_line,
                    QImage.Format_RGB888
                )

                pixmap = QPixmap.fromImage(q_image)
                self.image_label.setPixmap(pixmap)
                offset_x = max(0, (self.image_label.width() - target_w) // 2)
                offset_y = max(0, (self.image_label.height() - target_h) // 2)
                self.displayed_pixmap_rect = (offset_x, offset_y, target_w, target_h)
        except Exception:
            pass

    def stop_detection(self):
        """停止检测"""
        self.stop_preview_refresh()
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None

        self.set_running_state(False)
        self.pause_btn.setText("暂停")
        self.image_label.clear()
        self.image_label.setText("选择检测模式后点击开始")
        self.status_label.setText("就绪 | 等待开始检测...")
        self.displayed_pixmap_rect = None
        self.original_frame_size = None
        self.drawing_mode = None
        self.drawing_points = []
        self.update_warnings([])
        self.reset_metrics_display()

    def toggle_pause(self):
        """暂停/继续"""
        if self.detection_thread:
            paused = not self.detection_thread.paused
            self.detection_thread.set_paused(paused)
            state = "暂停" if paused else "继续"
            self.log_message(f"检测已{state}")
            self.pause_btn.setText("继续" if paused else "暂停")
            self.runtime_note_label.setText("已暂停" if paused else "等待推理刷新")

    def on_conf_changed(self, value):
        """置信度变化"""
        self.conf_value_label.setText(f"{value / 100:.2f}")
        self.update_thread_params()

    def on_iou_changed(self, value):
        """IOU 变化"""
        self.iou_value_label.setText(f"{value / 100:.2f}")
        self.update_thread_params()

    def on_distance_changed(self, value):
        """安全距离变化"""
        if self.detection_thread:
            self.detection_thread.update_safe_distance(value)

    def on_calibration_changed(self, value):
        """标定参数变化"""
        if self.detection_thread:
            self.detection_thread.update_calibration(value)

    def update_thread_params(self):
        """实时更新检测线程参数"""
        if self.detection_thread:
            conf = self.conf_slider.value() / 100.0
            iou = self.iou_slider.value() / 100.0
            self.detection_thread.update_params(conf, iou)

    def set_running_state(self, running):
        """设置运行状态"""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        self._sync_runtime_option_states()

    def log_message(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_detection()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = YOLOv8GUI()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
