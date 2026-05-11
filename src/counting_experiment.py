"""
计数误差实验脚本
对测试视频运行检测+跟踪+计数，输出各视频的计数结果，
配合人工标注计算误差率。

用法: cd src && python counting_experiment.py
"""

import os
import sys
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

from core.tracker import ObjectTracker
from core.counter import LineCounter, VehicleSessionCounter

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 配置 ──────────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train', 'weights', 'best.pt')
VIDEO_DIR = os.path.join(PROJECT_ROOT, 'videos')
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# 人工标注的计数结果（观看视频后手动填写）
# 格式: {视频文件名: {"person_up": N, "person_down": N, "car_up": N, ...}}
GROUND_TRUTH = {
    # 示例 - 需要你观看视频后填写
    # "1_1路面监控25FPS.mp4": {"person_up": 0, "person_down": 0, "car_up": 0, "car_down": 0},
}


def process_video(video_path, model, tracker, counter, line_counter):
    """处理单个视频，返回计数结果和帧数。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 默认计数线：画面中间水平线
    if not line_counter.is_line_set():
        line_counter.set_line(
            (0, height // 2),
            (width, height // 2),
        )

    session_counter = VehicleSessionCounter()
    frame_idx = 0

    print(f"  分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO 推理
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2, y2], cls_id, conf))

        # 跟踪
        tracks = tracker.update(detections)

        # 计数
        session_counter.update(tracks)
        line_counter.update(tracks)

        if frame_idx % 100 == 0:
            print(f"  已处理 {frame_idx}/{total_frames} 帧...")

    cap.release()

    line_result = line_counter.get_line()
    return {
        "frames": frame_idx,
        "up_counts": dict(line_counter.up_counts),
        "down_counts": dict(line_counter.down_counts),
        "total_up": sum(line_counter.up_counts.values()),
        "total_down": sum(line_counter.down_counts.values()),
    }


def run_experiment():
    """运行计数误差实验。"""
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}\n")

    # 列出所有视频
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        print(f"视频目录为空: {VIDEO_DIR}")
        return

    print(f"找到 {len(videos)} 个视频\n")
    print("=" * 70)

    all_results = {}

    for video_name in videos:
        video_path = os.path.join(VIDEO_DIR, video_name)
        print(f"\n处理视频: {video_name}")

        tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        counter = VehicleSessionCounter()
        line_counter = LineCounter()

        result = process_video(video_path, model, tracker, counter, line_counter)
        if result is None:
            continue

        all_results[video_name] = result

        print(f"  处理帧数: {result['frames']}")
        print(f"  上行计数: {result['up_counts']}")
        print(f"  下行计数: {result['down_counts']}")
        print(f"  上行总计: {result['total_up']}")
        print(f"  下行总计: {result['total_down']}")

    # 误差分析
    print("\n" + "=" * 70)
    print("计数误差分析")
    print("=" * 70)

    if not GROUND_TRUTH:
        print("\n未提供人工标注数据。请在脚本顶部 GROUND_TRUTH 字典中填写后重新运行。")
        print("\n系统计数结果汇总:")
        for name, res in all_results.items():
            print(f"  {name}: 上行={res['total_up']}, 下行={res['total_down']}")
        return

    print(f"\n{'视频':<30s} {'类别':<12s} {'方向':<6s} {'系统计数':<8s} {'人工计数':<8s} {'误差率':<8s}")
    print("-" * 70)

    total_system = 0
    total_manual = 0

    for name, res in all_results.items():
        if name not in GROUND_TRUTH:
            print(f"  {name}: 无人工标注，跳过")
            continue

        gt = GROUND_TRUTH[name]
        for direction in ['up', 'down']:
            counts = res[f'{direction}_counts']
            for cls_name, sys_count in counts.items():
                gt_key = f"{cls_name}_{direction}"
                manual_count = gt.get(gt_key, 0)
                if manual_count > 0:
                    error_rate = abs(sys_count - manual_count) / manual_count * 100
                else:
                    error_rate = 100.0 if sys_count > 0 else 0.0

                dir_label = "上行" if direction == "up" else "下行"
                print(f"  {name:<30s} {cls_name:<12s} {dir_label:<6s} {sys_count:<8d} {manual_count:<8d} {error_rate:.1f}%")

                total_system += sys_count
                total_manual += manual_count

    if total_manual > 0:
        overall_error = abs(total_system - total_manual) / total_manual * 100
        print("-" * 70)
        print(f"  总体: 系统={total_system}, 人工={total_manual}, 误差率={overall_error:.1f}%")


if __name__ == '__main__':
    run_experiment()
