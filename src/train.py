from ultralytics import YOLO
import os

# 【必须添加】Windows 下多进程训练的入口保护
if __name__ == '__main__':
    # 加载模型
    # 建议检查一下模型路径是否正确，截图里显示你有 yolo26n.pt，但代码写的是 yolov8n.pt
    model_path = '../models/yolov8n.pt'

    # 如果 yolov8n.pt 不存在，而你有 yolo26n.pt，请改成下面这行：
    # model_path = '../models/yolo26n.pt'

    if not os.path.exists(model_path):
        print(f"警告：找不到模型文件 {model_path}，请检查文件名！")
        # 尝试自动修正文件名（如果你的模型确实叫 yolo26n.pt）
        if os.path.exists('../models/yolo26n.pt'):
            model_path = '../models/yolo26n.pt'
            print(f"已自动切换为：{model_path}")

    model = YOLO(model_path)

    # 开始训练
    # workers=0: 先设为 0 避免多进程报错，稳定后可改为 8 加速
    # cache=False: 强制重新扫描标签，避免旧缓存干扰
    model.train(
        data='../configs/datasets.yaml',
        epochs=100,
        imgsz=640,
        batch=48,
        workers=0,
        cache=False
    )

"""
服务器训练参数
from ultralytics import YOLO
import torch

# 检查 GPU 是否可用
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")

if __name__ == '__main__':
    # 加载模型
    model_path = 'yolo26n.pt'  # 模型文件在当前目录

    model = YOLO(model_path)

    # 开始 GPU 训练（加速配置）
    model.train(
        data='../configs/datasets.yaml',
        epochs=10,
        imgsz=640,
        batch=64,       # 进一步增大 batch size
        workers=8,      # 数据加载线程
        device=0,       # 使用 GPU 0
        amp=True,       # 混合精度训练
        cache=False,    # 内存不够，关闭缓存
        project='../runs/detect',
        name='train'
    )
"""