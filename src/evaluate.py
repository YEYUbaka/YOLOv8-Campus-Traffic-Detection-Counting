"""
模型评估脚本 - 在验证集上计算 mAP 等指标
用法: cd src && python evaluate.py
"""

from ultralytics import YOLO
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate():
    # 模型路径
    model_path = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, 'models', 'yolov8n.pt')
        print(f"未找到 best.pt，使用预训练模型: {model_path}")
    else:
        print(f"使用训练模型: {model_path}")

    model = YOLO(model_path)

    # 在验证集上评估
    results = model.val(
        data=os.path.join(PROJECT_ROOT, 'configs', 'datasets.yaml'),
        imgsz=640,
        batch=16,
        workers=0,
        conf=0.25,
        iou=0.6,
        device='cpu',
        verbose=True,
    )

    # 打印汇总结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"mAP50     (所有类别): {results.box.map50:.4f}")
    print(f"mAP50-95  (所有类别): {results.box.map:.4f}")
    print(f"Precision (所有类别): {results.box.mp:.4f}")
    print(f"Recall    (所有类别): {results.box.mr:.4f}")
    print("-" * 60)

    # 按类别打印
    names = results.names
    for i, (p, r, ap50, ap) in enumerate(
        zip(results.box.p, results.box.r, results.box.ap50, results.box.ap)
    ):
        cls_name = names.get(i, f"class_{i}")
        print(f"  {cls_name:12s}  P={p:.4f}  R={r:.4f}  mAP50={ap50:.4f}  mAP50-95={ap:.4f}")

    print("=" * 60)
    return results


if __name__ == '__main__':
    evaluate()
