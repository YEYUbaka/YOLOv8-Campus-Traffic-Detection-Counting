from ultralytics import YOLO
import cv2

# 加载官方预训练的 YOLOv8n 模型 (n 代表 nano，最小最快，适合初学者测试)
model = YOLO('../models/yolov8n.pt')

# 对一张图片进行推理 (这里会自动下载一张示例图或者你可以替换成你本地的图片路径)
# 如果你本地有图片，把 'test_images/image1.jpg' 换成你的图片路径
results = model('test_images/image3.jpg')

# 获取结果图片
result_image = results[0].plot()

# 显示检测结果
cv2.imshow('YOLOv8 Detection', result_image)

# 等待按键按下，0 表示无限期等待
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
