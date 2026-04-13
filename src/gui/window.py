
"""Main GUI window composition and event wiring."""

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from core.class_profile import (
    DEFAULT_TRAFFIC_CLASS_PROFILE,
    build_traffic_class_profile,
)
from core.counter import ClassCounter
from gui.presentation import (
    apply_live_runtime_indicators,
    apply_stats_update,
    apply_warning_text,
    build_live_status_text,
    build_runtime_metrics_note,
    build_runtime_note_text,
)
from gui.runtime import DetectionThread, resolve_runtime_device
from gui.widgets import StableVideoViewport, VideoDisplayLabel
from utils.drawing import DrawingUtils


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
        self.preview_timer.setInterval(100)
        if hasattr(Qt, "PreciseTimer"):
            self.preview_timer.setTimerType(Qt.PreciseTimer)
        self.preview_timer.timeout.connect(self.refresh_preview_frame)
        self._preview_refresh_pending = False
        self.preview_present_fps = 0.0
        self.preview_present_ms = 0.0
        self.preview_frame_count = 0
        self.preview_present_total = 0.0
        self.preview_window_start = 0.0
        self.last_preview_sequence = 0
        self.last_live_stats = None

        # 目录路径
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.dirname(self.module_dir)
        self.project_root = os.path.dirname(self.src_dir)
        self.config_dir = os.path.join(self.src_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)

        self.init_model()
        self.init_ui()

    def init_model(self):
        """加载 YOLO 模型"""
        model_paths = [
            os.path.join(self.project_root, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(self.project_root, "runs", "detect", "train", "weights", "last.pt"),
            os.path.join(self.project_root, "models", "yolov8n.pt"),
            os.path.join(self.project_root, "models", "yolo26n.pt"),
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
        video_profile = self._selected_video_resolution_profile() if self.current_mode == "video" else None
        return build_runtime_note_text(
            self.current_mode,
            self._resolved_device_display_name(),
            video_profile,
            inference_fps,
        )

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
        """启动最新预览帧刷新，信号驱动为主，定时轮询兜底。"""
        self._preview_refresh_pending = False
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
        self._preview_refresh_pending = False
        self.preview_present_fps = 0.0
        self.preview_present_ms = 0.0
        self.preview_frame_count = 0
        self.preview_present_total = 0.0
        self.preview_window_start = 0.0
        self.last_preview_sequence = 0
        self.last_live_stats = None

    def _build_runtime_metrics_note(self, stats):
        device_display_name = None
        video_resolution_profile = None
        if self.detection_thread:
            device_display_name = self.detection_thread.device_display_name
            if self.detection_thread.source_mode == "video":
                video_resolution_profile = self.detection_thread.video_resolution_profile
        return build_runtime_metrics_note(
            stats,
            present_ms=self.preview_present_ms if self.detection_thread else 0.0,
            device_display_name=device_display_name,
            video_resolution_profile=video_resolution_profile,
        )

    def _build_live_status_text(self, stats):
        device_display_name = None
        video_resolution_profile = None
        if self.detection_thread:
            device_display_name = self.detection_thread.device_display_name
            if self.detection_thread.source_mode == "video":
                video_resolution_profile = self.detection_thread.video_resolution_profile
        return build_live_status_text(
            stats,
            display_fps=self.preview_present_fps if self.detection_thread else stats.get('display_fps', 0.0),
            present_ms=self.preview_present_ms if self.detection_thread else 0.0,
            device_display_name=device_display_name,
            video_resolution_profile=video_resolution_profile,
        )

    def _apply_live_runtime_indicators(self, stats):
        apply_live_runtime_indicators(self, stats)

    def request_preview_refresh(self, *_args):
        """由检测线程发出轻量信号，请求主线程尽快拉取最新预览。"""
        if self._preview_refresh_pending:
            return
        self._preview_refresh_pending = True
        QTimer.singleShot(0, self.refresh_preview_frame)

    def refresh_preview_frame(self):
        """主线程主动拉取最新预览帧，丢弃所有过时帧。"""
        self._preview_refresh_pending = False
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

        if self.last_live_stats:
            self._apply_live_runtime_indicators(self.last_live_stats)

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
        qss_path = os.path.join(self.src_dir, "styles", "campus_ops_light.qss")

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型文件", self.project_root,
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
        self.detection_thread.preview_ready.connect(self.request_preview_refresh)
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
            self.detection_thread.preview_ready.connect(self.request_preview_refresh)
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
        apply_stats_update(self, stats)

    def update_warnings(self, warnings):
        """更新预警信息"""
        apply_warning_text(self.warning_text, warnings)

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
