"""Presentation helpers for the GUI status and statistics panes."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem


_EMPTY_ACTIVE_WARNING_DETAIL = "当前没有活动预警。"
_EMPTY_TIMELINE_TEXT = "当前没有事件流水。"


def normalize_ui_text(text):
    return str(text or "").replace("\ufeff", "").strip()


def build_empty_warning_payload():
    return {
        "active_items": [],
        "timeline_items": [],
    }


def build_runtime_note_text(mode, device_display_name, video_resolution_profile=None, inference_fps=0.0):
    parts = [f"模型 {inference_fps:.1f} FPS", device_display_name]
    if mode == "video" and video_resolution_profile:
        parts.append(video_resolution_profile)
    return " | ".join(part for part in parts if part)


def apply_live_runtime_indicators(window, stats):
    """KPI 指标卡已移除，保留桩函数避免外部调用报错。"""
    pass


def apply_stats_update(window, stats):
    if window.detection_thread and window.detection_thread.isRunning():
        window.last_live_stats = dict(stats)
    else:
        window.last_live_stats = None

    class_counts = stats.get("class_counts", {})
    cumulative_class_counts = stats.get("cumulative_class_counts", {})
    up_counts = stats.get("up_counts", {})
    down_counts = stats.get("down_counts", {})

    apply_live_runtime_indicators(window, stats)

    for i, name in enumerate(window.stats_table_class_names):
        current_count = class_counts.get(name, 0)
        cumulative_count = cumulative_class_counts.get(name, 0)
        up = up_counts.get(name, 0)
        down = down_counts.get(name, 0)
        window.stats_table.item(i, 1).setText(str(current_count))
        window.stats_table.item(i, 2).setText(str(cumulative_count))
        window.stats_table.item(i, 3).setText(str(up))
        window.stats_table.item(i, 4).setText(str(down))


def _build_table_item(text, alignment=Qt.AlignLeft | Qt.AlignVCenter):
    item = QTableWidgetItem(normalize_ui_text(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    item.setTextAlignment(alignment)
    return item


def _apply_active_warning_table(table_widget, active_items):
    rows = []
    for item in active_items:
        rows.append(
            (
                item.get("kind", ""),
                item.get("subject", ""),
                item.get("detail", ""),
            )
        )

    if not rows:
        rows = [("暂无", "-", _EMPTY_ACTIVE_WARNING_DETAIL)]

    table_widget.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            table_widget.setItem(row_index, col_index, _build_table_item(value))


def _apply_event_timeline(text_widget, timeline_items):
    lines = []
    for item in timeline_items:
        timestamp = normalize_ui_text(item.get("timestamp", ""))
        text = normalize_ui_text(item.get("text", ""))
        if not text:
            continue
        if timestamp:
            lines.append(f"[{timestamp}] {text}")
        else:
            lines.append(text)

    content = "\n".join(lines) if lines else _EMPTY_TIMELINE_TEXT
    if text_widget.toPlainText() != content:
        text_widget.setPlainText(content)


def apply_warning_payload(window, warning_payload):
    payload = warning_payload or build_empty_warning_payload()
    active_items = payload.get("active_items") or []
    timeline_items = payload.get("timeline_items") or []
    _apply_active_warning_table(window.active_warning_table, active_items)
    _apply_event_timeline(window.event_timeline_text, timeline_items)


def apply_warning_text(warning_text_widget, warnings):
    if not warnings:
        if warning_text_widget.toPlainText() != _EMPTY_ACTIVE_WARNING_DETAIL:
            warning_text_widget.setPlainText(_EMPTY_ACTIVE_WARNING_DETAIL)
        return

    text = "\n".join(
        normalized for normalized in (normalize_ui_text(warning) for warning in warnings)
        if normalized
    )
    if not text:
        text = _EMPTY_ACTIVE_WARNING_DETAIL

    if warning_text_widget.toPlainText() != text:
        warning_text_widget.setPlainText(text)
