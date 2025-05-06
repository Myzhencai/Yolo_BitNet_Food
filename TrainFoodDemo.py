from ultralytics import YOLO
# 利用下载的图片训练自己的模型
# https://universe.roboflow.com/models/object-detection
# 加载YOLOv12模型
model = YOLO('D:\\JudyYolo\\models\\yolo12n.pt')  # 或者使用已训练模型 yolov12.pt

# 训练模型
model.train(
    data='D:\\JudyYolo\\Datasets\\data.yaml',  # 数据配置文件路径
    epochs=1,
    imgsz=640,
    batch=16,
    name='yolov12_custom',
    project='runs/train',
    device="cpu"  # 设置为'cpu'或gpu编号，如0
)
