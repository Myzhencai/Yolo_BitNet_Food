from ultralytics import YOLO

# 加载模型（可以是 'yolov8n.yaml', 'yolov8s.yaml', 'yolov8m.yaml', 'yolov8l.yaml', 'yolov8x.yaml'）
model = YOLO('D:\\JudyYolo\\yolo\\ultralytics\\ultralytics\\cfg\\models\\12\\yolo12_MCAM2.yaml')

# 开始训练
model.train(
    data='D:\\JudyYolo\\Datasets\\data.yaml',
    epochs=1,
    imgsz=640,
    batch=16,
    name='yoloMCAM2',
    project='runs/train',
    device="cpu"
)
