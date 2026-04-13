# YOLOv8 校园交通检测与计数系统 - 增强版

基于 YOLOv8 的校园交通目标检测与计数系统，支持目标跟踪、类别统计、上下行计数、速度检测、碰撞预警、违规区域检测等功能。

## 功能特性

### 🎯 核心功能

| 功能 | 描述 |
|------|------|
| **目标检测** | 基于 YOLOv8 的高精度目标检测 |
| **目标跟踪** | 基于 IoU 的多目标跟踪算法 |
| **类别统计** | 实时统计各类型目标数量 |
| **上下行计数** | 通过检测线统计上下行车辆数量 |
| **速度检测** | 实时检测车辆速度（精确到 0.1 km/h） |
| **碰撞预警** | 检测车辆间距离，发出碰撞预警 |
| **违规检测** | 检测车辆是否进入自定义违规区域 |

### 📊 支持的类别

| ID | 类别 | 英文 |
|----|------|------|
| 0 | 行人 | person |
| 1 | 自行车 | bicycle |
| 2 | 轿车 | car |
| 3 | 摩托车 | motorcycle |
| 4 | 公交车 | bus |
| 5 | 卡车 | truck |
| 6 | 三轮车 | tricycle |

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将数据集上传后，确保目录结构如下：

```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 3. 配置文件

检查 `configs/datasets.yaml` 配置是否正确。

## 使用方法

### 启动 GUI 界面

```bash
cd src
python GUI.py
```

### GUI 操作说明

1. **选择模型**：系统会自动加载训练好的模型，也可以手动选择模型文件
2. **选择检测模式**：摄像头/视频文件/图片
3. **功能开关**：
   - 目标跟踪：启用/禁用多目标跟踪
   - 速度检测：启用/禁用速度估算
   - 碰撞预警：启用/禁用碰撞检测
   - 违规检测：启用/禁用违规区域检测
4. **绘制工具**：
   - **绘制检测线**：点击两个点绘制检测线，用于上下行计数
   - **绘制违规区域**：点击多个点绘制多边形区域，右键完成
   - **清除区域**：清除所有已绘制的区域
5. **参数设置**：
   - 置信度阈值：调整检测置信度
   - IOU 阈值：调整 NMS 阈值
   - 安全距离：设置碰撞预警的安全距离（米）
   - 像素/米：设置标定参数，用于速度和距离计算

## 训练模型

```bash
cd src
python train.py
```

训练参数在 `train.py` 中可调整：
- epochs: 训练轮数（默认100）
- batch: 批次大小（默认48）
- imgsz: 图像尺寸（默认640）

## 项目结构

```
YOLOv8-Campus-Traffic-Detection-Counting/
├── configs/
│   └── datasets.yaml          # 数据集配置
├── datasets/                   # 数据集目录
├── models/                     # 预训练模型
├── runs/                       # 训练输出
├── src/
│   ├── core/                  # 核心功能模块
│   │   ├── __init__.py
│   │   ├── tracker.py         # 目标跟踪器
│   │   ├── counter.py         # 类别统计和上下行计数
│   │   ├── speed_estimator.py # 速度检测
│   │   ├── collision_warner.py# 碰撞预警
│   │   └── zone_detector.py   # 违规区域检测
│   ├── utils/                 # 工具模块
│   │   ├── __init__.py
│   │   ├── drawing.py         # 绘图工具
│   │   └── calibration.py     # 标定管理
│   ├── styles/
│   │   └── dark_theme.qss     # 深色主题样式
│   ├── GUI.py                 # 主界面
│   ├── train.py               # 训练脚本
│   └── predit.py              # 预测脚本
├── test_images/               # 测试图片
├── videos/                    # 测试视频
└── requirements.txt           # 依赖列表
```

## 核心模块说明

### 1. 目标跟踪器 (tracker.py)

基于 IoU 的简单目标跟踪算法，实现类似 SORT 的跟踪逻辑：
- 支持目标 ID 持续跟踪
- 保存位置历史用于轨迹绘制
- 自动处理目标消失和重新出现

### 2. 计数器 (counter.py)

- **ClassCounter**: 实时统计各类型目标数量
- **LineCounter**: 通过检测线统计上下行车辆数量

### 3. 速度检测 (speed_estimator.py)

- 通过目标位置历史和帧率计算速度
- 支持标定参数设置（像素/米）
- 输出精度：0.1 km/h

### 4. 碰撞预警 (collision_warner.py)

- 计算目标之间的距离
- 支持设置安全距离阈值
- 分级预警：高危/中危

### 5. 违规区域检测 (zone_detector.py)

- 支持多个多边形区域
- 区域类型：禁停区、禁止驶入、专用车道、自定义
- 支持鼠标绘制和配置文件保存

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| ultralytics | >=8.0.0 | YOLOv8 目标检测框架 |
| torch | >=2.0.0 | 深度学习框架 |
| opencv-python | >=4.8.0 | 图像处理 |
| PyQt5 | >=5.15.0 | GUI 界面 |
| numpy | >=1.24.0 | 数值计算 |

## 注意事项

### Windows 环境

1. **多进程训练**: `train.py` 必须包含 `if __name__ == '__main__':` 保护
2. **workers 参数**: Windows 下设为 `workers=0` 避免多进程错误
3. **路径分隔符**: 配置文件使用正斜杠 `/` 或双反斜杠 `\\`

### GPU 训练

```python
model.train(
    data='../configs/datasets.yaml',
    epochs=100,
    imgsz=640,
    batch=64,
    workers=8,
    device=0,      # GPU 设备 ID
    amp=True,      # 混合精度训练
)
```

## 输出文件

训练完成后:
- `runs/detect/train/weights/best.pt` - 最佳模型
- `runs/detect/train/weights/last.pt` - 最后模型
- `runs/detect/train/results.csv` - 训练指标

## License

MIT License