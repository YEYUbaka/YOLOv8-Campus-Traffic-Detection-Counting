"""Presentation helpers for the GUI status and statistics panes."""

from datetime import datetime


def build_runtime_note_text(mode, device_display_name, video_resolution_profile=None, inference_fps=0.0):
    parts = [f"模型 {inference_fps:.1f} FPS", device_display_name]
    if mode == "video" and video_resolution_profile:
        parts.append(video_resolution_profile)
    return " | ".join(parts)


def build_runtime_metrics_note(stats, present_ms=0.0, device_display_name=None, video_resolution_profile=None):
    inference_fps = stats.get("inference_fps", stats.get("fps", 0.0))
    processed_fps = stats.get("processed_fps", 0.0)
    avg_frame_step = stats.get("avg_frame_step", 1.0)
    decode_ms = stats.get("decode_ms", 0.0)
    model_ms = stats.get("model_ms", 0.0)
    analysis_ms = stats.get("analysis_ms", 0.0)
    draw_ms = stats.get("draw_ms", 0.0)
    preview_ms = stats.get("preview_ms", 0.0)

    runtime_parts = [
        f"处理 {processed_fps:.1f} FPS",
        f"模型 {inference_fps:.1f} FPS",
        f"步长 {avg_frame_step:.2f}",
        (
            f"解{decode_ms:.1f}/模{model_ms:.1f}/析{analysis_ms:.1f}/"
            f"绘{draw_ms:.1f}/预{preview_ms:.1f}/显{present_ms:.1f} ms"
        ),
    ]
    if device_display_name:
        runtime_parts.append(device_display_name)
    if video_resolution_profile:
        runtime_parts.append(video_resolution_profile)
    return " | ".join(runtime_parts)


def build_live_status_text(
    stats,
    display_fps,
    present_ms=0.0,
    device_display_name=None,
    video_resolution_profile=None,
):
    current_time = datetime.now().strftime("%H:%M:%S")
    processed_fps = stats.get("processed_fps", 0.0)
    inference_fps = stats.get("inference_fps", stats.get("fps", 0.0))
    avg_frame_step = stats.get("avg_frame_step", 1.0)
    decode_ms = stats.get("decode_ms", 0.0)
    model_ms = stats.get("model_ms", 0.0)
    analysis_ms = stats.get("analysis_ms", 0.0)
    draw_ms = stats.get("draw_ms", 0.0)
    preview_ms = stats.get("preview_ms", 0.0)
    speed_range = stats.get("speed_range", (0.0, 0.0))

    status_parts = [
        f"时间：{current_time}",
        f"显示 FPS：{display_fps:.1f}",
        f"处理 FPS：{processed_fps:.1f}",
        f"模型 FPS：{inference_fps:.1f}",
        f"平均步长：{avg_frame_step:.2f}",
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
    if device_display_name:
        status_parts.append(f"设备：{device_display_name}")
    if video_resolution_profile:
        status_parts.append(f"视频档位：{video_resolution_profile}")
    return " | ".join(status_parts)


def apply_live_runtime_indicators(window, stats):
    has_thread = bool(window.detection_thread)
    display_fps = window.preview_present_fps if has_thread else stats.get("display_fps", 0.0)
    window.runtime_value_label.setText(f"{display_fps:.1f}")

    if has_thread:
        device_display_name = window.detection_thread.device_display_name
        video_resolution_profile = (
            window.detection_thread.video_resolution_profile
            if window.detection_thread.source_mode == "video"
            else None
        )
        window.runtime_note_label.setText(
            build_runtime_metrics_note(
                stats,
                present_ms=window.preview_present_ms,
                device_display_name=device_display_name,
                video_resolution_profile=video_resolution_profile,
            )
        )
        window.status_label.setText(
            build_live_status_text(
                stats,
                display_fps=display_fps,
                present_ms=window.preview_present_ms,
                device_display_name=device_display_name,
                video_resolution_profile=video_resolution_profile,
            )
        )
    else:
        window.runtime_note_label.setText("单图结果")


def apply_stats_update(window, stats):
    if window.detection_thread and window.detection_thread.isRunning():
        window.last_live_stats = dict(stats)
    else:
        window.last_live_stats = None

    unique_vehicle_total = stats.get("unique_vehicle_total", 0)
    current_vehicle_count = stats.get("current_vehicle_count", 0)
    warning_count = stats.get("warning_count", 0)
    class_counts = stats.get("class_counts", {})
    up_counts = stats.get("up_counts", {})
    down_counts = stats.get("down_counts", {})

    window.unique_vehicle_value_label.setText(str(unique_vehicle_total))
    window.unique_vehicle_note_label.setText("本次会话累计")
    window.current_vehicle_value_label.setText(str(current_vehicle_count))
    window.current_vehicle_note_label.setText("已确认轨迹")
    window.warning_count_value_label.setText(str(warning_count))
    window.warning_count_note_label.setText("当前帧事件")
    apply_live_runtime_indicators(window, stats)

    for i, name in enumerate(window.stats_table_class_names):
        count = class_counts.get(name, 0)
        up = up_counts.get(name, 0)
        down = down_counts.get(name, 0)
        window.stats_table.item(i, 1).setText(str(count))
        window.stats_table.item(i, 2).setText(str(up))
        window.stats_table.item(i, 3).setText(str(down))


def apply_warning_text(warning_text_widget, warnings):
    if not warnings:
        warning_text_widget.setPlainText("当前没有预警事件。")
        return
    warning_text_widget.setPlainText("\n".join(warnings))
