
"""GUI viewport widgets."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel, QWidget


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
