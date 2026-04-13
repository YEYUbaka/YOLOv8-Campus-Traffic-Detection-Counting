import os
from ultralytics import YOLO

# 加载模型（只是为了初始化环境，其实不需要真的加载）
# model = YOLO('../models/yolov8n.pt')

# 定义路径
dataset_root = r"E:\Code_save\Python\计算机视觉\YOLOv8-Campus-Traffic-Detection-Counting\datasets"
img_dir = os.path.join(dataset_root, "images", "train")
lbl_dir = os.path.join(dataset_root, "labels", "train")  # 注意这里必须是 labels

print(f"检查图片目录: {img_dir}")
print(f"检查标签目录: {lbl_dir}")

if not os.path.exists(lbl_dir):
    print(f"❌ 错误：找不到标签目录 {lbl_dir}")
    print("💡 提示：请检查是否应该叫 'labels' 而不是 'labelsyolov'")
else:
    print("✅ 标签目录存在")

# 随机检查前 5 个文件
files = os.listdir(img_dir)[:5]
for f in files:
    if f.endswith('.jpg') or f.endswith('.png'):
        name_no_ext = os.path.splitext(f)[0]
        txt_path = os.path.join(lbl_dir, name_no_ext + '.txt')

        if os.path.exists(txt_path):
            size = os.path.getsize(txt_path)
            if size > 0:
                print(f"✅ [{f}] -> 找到标签 ({size} bytes)")
            else:
                print(f"⚠️ [{f}] -> 标签文件存在但是空的!")
        else:
            print(f"❌ [{f}] -> 找不到对应的标签文件: {txt_path}")