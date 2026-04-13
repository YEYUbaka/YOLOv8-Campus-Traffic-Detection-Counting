"""
绘图工具模块
用于在视频帧上绘制各种元素
"""

from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

from core.class_profile import DEFAULT_TRAFFIC_CLASS_PROFILE, TrafficClassProfile


class DrawingUtils:
    """绘图工具类"""

    # 类别颜色（BGR）
    CLASS_COLORS = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.class_colors)

    # 中文类别名称
    CLASS_NAMES = dict(DEFAULT_TRAFFIC_CLASS_PROFILE.display_names)
    CLASS_ASCII_NAMES = {
        class_id: DEFAULT_TRAFFIC_CLASS_PROFILE.canonical_names.get(class_id, str(class_id))
        for class_id in DEFAULT_TRAFFIC_CLASS_PROFILE.display_names
    }

    # 字体缓存
    _font_cache = {}
    _measure_draw = ImageDraw.Draw(Image.new('RGB', (2, 2)))

    @staticmethod
    def configure_class_profile(profile: TrafficClassProfile):
        """同步当前模型的类别名称和颜色。"""
        DrawingUtils.CLASS_NAMES = dict(profile.display_names)
        DrawingUtils.CLASS_COLORS = dict(profile.class_colors)
        DrawingUtils.CLASS_ASCII_NAMES = {
            class_id: profile.canonical_names.get(class_id) or str(profile.raw_names.get(class_id, class_id))
            for class_id in profile.display_names
        }

    @staticmethod
    def _measure_ascii_text(text, font_scale=0.5, thickness=1):
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        return text_width, text_height, baseline

    @staticmethod
    def _put_ascii_text(
        img,
        text,
        position,
        font_scale=0.5,
        color=(255, 255, 255),
        thickness=1,
    ):
        x, y = int(position[0]), int(position[1])
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return img

    @staticmethod
    def _get_font(size=20):
        """获取中文字体"""
        if size not in DrawingUtils._font_cache:
            try:
                # Windows 系统字体路径
                font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                ]
                for path in font_paths:
                    if os.path.exists(path):
                        DrawingUtils._font_cache[size] = ImageFont.truetype(path, size)
                        break
                else:
                    DrawingUtils._font_cache[size] = ImageFont.load_default()
            except:
                DrawingUtils._font_cache[size] = ImageFont.load_default()
        return DrawingUtils._font_cache[size]

    @staticmethod
    def _put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = DrawingUtils._get_font(font_size)
        # PIL 使用 RGB 颜色
        color_rgb = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _measure_text(text, font_size=20):
        """测量中文文本尺寸，避免每次都转换整帧。"""
        font = DrawingUtils._get_font(font_size)
        bbox = DrawingUtils._measure_draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def render_text_items(frame: np.ndarray, text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]]) -> np.ndarray:
        """将一帧中的所有中文文本批量绘制到图像上。"""
        if not text_items:
            return frame

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for text, position, font_size, color in text_items:
            font = DrawingUtils._get_font(font_size)
            color_rgb = (color[2], color[1], color[0])
            draw.text(position, text, font=font, fill=color_rgb)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def draw_detection_box(
        frame: np.ndarray,
        bbox: List[float],
        class_id: int,
        track_id: int = None,
        confidence: float = None,
        speed: float = None,
        color: Tuple[int, int, int] = None,
        show_label: bool = True,
        text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = None,
        ascii_label: bool = False,
    ) -> np.ndarray:
        """绘制检测框"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        min_edge = min(box_w, box_h)

        if color is None:
            color = DrawingUtils.CLASS_COLORS.get(class_id, (0, 255, 0))

        box_thickness = 3 if min_edge >= 90 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # 远处小目标保留框线即可，减少标签堆叠和绘制开销
        if min_edge < 24:
            return frame

        if not show_label:
            return frame

        # 构建标签文本
        if ascii_label:
            class_name = DrawingUtils.CLASS_ASCII_NAMES.get(class_id, f'cls{class_id}')
        else:
            class_name = DrawingUtils.CLASS_NAMES.get(class_id, f'ID:{class_id}')
        label_parts = [class_name]

        if track_id is not None:
            label_parts.append(f'#{track_id}')

        if confidence is not None and track_id is None:
            label_parts.append(f'{confidence:.2f}')

        if speed is not None and speed > 0:
            label_parts.append(f'{round(speed):.0f}km/h')

        label = ' '.join(label_parts)

        if ascii_label:
            font_scale = 0.45 if min_edge < 56 else 0.55 if min_edge < 110 else 0.65
            text_thickness = 1 if min_edge < 90 else 2
            text_width, text_height, baseline = DrawingUtils._measure_ascii_text(
                label,
                font_scale,
                text_thickness,
            )
            label_top = max(0, y1 - text_height - baseline - 10)
            cv2.rectangle(
                frame,
                (x1, label_top),
                (x1 + text_width + 8, y1),
                (0, 0, 0), -1
            )
            text_y = max(text_height + 2, y1 - baseline - 4)
            return DrawingUtils._put_ascii_text(
                frame,
                label,
                (x1 + 4, text_y),
                font_scale,
                (255, 255, 255),
                text_thickness,
            )

        font_size = 16 if min_edge < 56 else 18 if min_edge < 110 else 20
        text_width, text_height = DrawingUtils._measure_text(label, font_size)
        label_top = max(0, y1 - text_height - 8)
        text_pos = (x1 + 4, max(0, y1 - text_height - 6))

        # 深色半透明背景
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 8, y1),
            (0, 0, 0), -1
        )

        if text_items is not None:
            text_items.append((label, text_pos, font_size, (255, 255, 255)))
        else:
            frame = DrawingUtils._put_chinese_text(
                frame, label, text_pos, font_size, (255, 255, 255)
            )

        return frame
        
    @staticmethod
    def draw_tracking_line(
        frame: np.ndarray,
        history: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        绘制跟踪轨迹线
        
        Args:
            frame: 视频帧
            history: 位置历史
            color: 颜色
            thickness: 线宽
            
        Returns:
            绘制后的帧
        """
        if len(history) < 2:
            return frame
        recent_history = history[-6:]

        # 绘制轨迹
        for i in range(1, len(recent_history)):
            pt1 = (int(recent_history[i-1][0]), int(recent_history[i-1][1]))
            pt2 = (int(recent_history[i][0]), int(recent_history[i][1]))
            
            cv2.line(frame, pt1, pt2, color, thickness)
            
        return frame
        
    @staticmethod
    def draw_counting_line(
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
        text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = None,
        show_text: bool = True,
        ascii_label: bool = False,
    ) -> np.ndarray:
        """绘制计数检测线"""
        # 绘制主检测线
        cv2.line(frame, start, end, color, thickness)

        # 绘制端点
        cv2.circle(frame, start, 8, color, -1)
        cv2.circle(frame, end, 8, color, -1)

        # 绘制方向箭头
        DrawingUtils._draw_direction_arrow(
            frame,
            start,
            end,
            color,
            text_items=text_items,
            show_text=show_text,
            ascii_label=ascii_label,
        )

        if not show_text:
            return frame

        # 添加标签
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2

        text_pos = (mid_x - 30, mid_y - 20)
        if ascii_label:
            return DrawingUtils._put_ascii_text(frame, 'LINE', text_pos, 0.6, color, 2)
        if text_items is not None:
            text_items.append(('检测线', text_pos, 20, color))
        else:
            frame = DrawingUtils._put_chinese_text(
                frame, '检测线', text_pos, 20, color
            )

        return frame
        
    @staticmethod
    def _draw_direction_arrow(
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = None,
        show_text: bool = True,
        ascii_label: bool = False,
    ):
        """绘制方向箭头"""
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)

        if length > 0:
            nx = -dy / length * 30
            ny = dx / length * 30

            arrow_end = (int(mid_x + nx), int(mid_y + ny))
            cv2.arrowedLine(
                frame, (mid_x, mid_y), arrow_end,
                (255, 255, 255), 2, tipLength=0.3
            )

            if not show_text:
                return frame

            text_pos = (arrow_end[0] + 5, arrow_end[1] - 10)
            if ascii_label:
                return DrawingUtils._put_ascii_text(
                    frame, 'UP', text_pos, 0.5, (255, 255, 255), 1
                )
            if text_items is not None:
                text_items.append(('上行', text_pos, 18, (255, 255, 255)))
            else:
                frame = DrawingUtils._put_chinese_text(
                    frame, '上行', text_pos, 18, (255, 255, 255)
                )

        return frame
            
    @staticmethod
    def draw_zone(
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        name: str = None,
        color: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.3,
        text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """绘制违规区域"""
        if len(points) < 3:
            return frame

        pts = np.array(points, dtype=np.int32)

        # 绘制半透明填充
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 绘制边框
        cv2.polylines(frame, [pts], True, color, 2)

        # 绘制顶点
        for pt in points:
            cv2.circle(frame, pt, 5, color, -1)

        # 绘制名称
        if name and len(points) >= 3:
            center_x = sum(p[0] for p in points) // len(points)
            center_y = sum(p[1] for p in points) // len(points)

            text_pos = (center_x - 40, center_y - 10)
            if text_items is not None:
                text_items.append((name, text_pos, 20, (255, 255, 255)))
            else:
                frame = DrawingUtils._put_chinese_text(
                    frame, name, text_pos, 20, (255, 255, 255)
                )

        return frame
        
    @staticmethod
    def draw_warning(
        frame: np.ndarray,
        position: Tuple[float, float],
        message: str,
        severity: str = 'high'
    ) -> np.ndarray:
        """
        绘制预警标记
        
        Args:
            frame: 视频帧
            position: 位置
            message: 预警消息
            severity: 严重程度 ('high', 'medium')
            
        Returns:
            绘制后的帧
        """
        x, y = int(position[0]), int(position[1])
        
        # 根据严重程度选择颜色
        color = (0, 0, 255) if severity == 'high' else (0, 165, 255)
        
        # 绘制警告圆圈
        cv2.circle(frame, (x, y), 15, color, 3)
        
        # 绘制感叹号
        cv2.putText(
            frame, '!',
            (x - 5, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )
        
        return frame
        
    @staticmethod
    def draw_speed_display(
        frame: np.ndarray,
        position: Tuple[float, float],
        speed: float,
        is_speeding: bool = False
    ) -> np.ndarray:
        """
        绘制速度显示
        
        Args:
            frame: 视频帧
            position: 位置
            speed: 速度
            is_speeding: 是否超速
            
        Returns:
            绘制后的帧
        """
        x, y = int(position[0]), int(position[1])
        
        # 根据是否超速选择颜色
        color = (0, 0, 255) if is_speeding else (0, 255, 0)
        
        # 绘制速度文本
        text = f'{speed:.1f} km/h'
        
        cv2.putText(
            frame, text,
            (x - 30, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1
        )
        
        return frame
        
    @staticmethod
    def draw_statistics_panel(
        frame: np.ndarray,
        class_counts: Dict[str, int],
        up_counts: Dict[str, int] = None,
        down_counts: Dict[str, int] = None,
        position: Tuple[int, int] = (10, 30),
        unique_vehicle_total: int = None,
        current_vehicle_count: int = None,
        warning_count: int = 0,
        text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """绘制统计面板"""
        x, y = position
        line_height = 26

        info_lines = []
        if unique_vehicle_total is not None:
            info_lines.append((f'去重总车辆: {unique_vehicle_total}', (37, 85, 54)))
        if current_vehicle_count is not None:
            info_lines.append((f'当前车辆: {current_vehicle_count}', (37, 85, 54)))
        if warning_count:
            info_lines.append((f'预警数: {warning_count}', (176, 95, 27)))

        panel_height = (len(info_lines) + len(class_counts) + 2) * line_height + 24
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 8, y - 8),
            (x + 272, y + panel_height),
            (244, 249, 241), -1
        )
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(
            frame,
            (x - 8, y - 8),
            (x + 272, y + panel_height),
            (164, 192, 168), 1
        )

        if text_items is not None:
            text_items.append(('实时统计', (x, y), 22, (37, 85, 54)))
        else:
            frame = DrawingUtils._put_chinese_text(
                frame, '实时统计', (x, y), 22, (37, 85, 54)
            )
        y += line_height

        for text, color in info_lines:
            if text_items is not None:
                text_items.append((text, (x, y), 18, color))
            else:
                frame = DrawingUtils._put_chinese_text(
                    frame, text, (x, y), 18, color
                )
            y += line_height

        # 绘制类别计数
        for class_name, count in class_counts.items():
            text = f'{class_name}: {count}'
            if text_items is not None:
                text_items.append((text, (x, y), 18, (70, 86, 72)))
            else:
                frame = DrawingUtils._put_chinese_text(
                    frame, text, (x, y), 18, (70, 86, 72)
                )
            y += line_height

        # 绘制上下行计数
        if up_counts and down_counts:
            y += 5
            total_up = sum(up_counts.values())
            total_down = sum(down_counts.values())

            text = f'上行: {total_up} | 下行: {total_down}'
            if text_items is not None:
                text_items.append((text, (x, y), 18, (40, 108, 76)))
            else:
                frame = DrawingUtils._put_chinese_text(
                    frame, text, (x, y), 18, (40, 108, 76)
                )

        return frame
        
    @staticmethod
    def draw_drawing_preview(
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        current_pos: Tuple[int, int] = None,
        mode: str = 'line'
    ) -> np.ndarray:
        """
        绘制绘图预览
        
        Args:
            frame: 视频帧
            points: 已绘制的点
            current_pos: 当前鼠标位置
            mode: 绘制模式 ('line' 或 'zone')
            
        Returns:
            绘制后的帧
        """
        if not points:
            return frame
            
        # 绘制已确定的点
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)
            
        # 绘制连线
        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (255, 255, 0), 2)
                
        # 绘制到当前鼠标位置的预览线
        if current_pos and points:
            cv2.line(frame, points[-1], current_pos, (255, 255, 0), 1, cv2.LINE_AA)
            
            if mode == 'zone' and len(points) >= 2:
                # 绘制闭合预览
                cv2.line(frame, current_pos, points[0], (255, 255, 0), 1, cv2.LINE_AA)
                
        return frame
