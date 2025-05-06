from ultralytics import YOLO

# 加载模型（可以是 'yolov8n.yaml', 'yolov8s.yaml', 'yolov8m.yaml', 'yolov8l.yaml', 'yolov8x.yaml'）
model = YOLO('D:\\JudyYolo\\yolo\\ultralytics\\ultralytics\\cfg\\models\\12\\yolo12PConv.yaml')  # 你也可以替换成 yolov8m.yaml 等

# 开始训练
model.train(
    data='D:\\JudyYolo\\Datasets\\data.yaml',  # 数据集配置文件路径
    epochs=1,                     # 训练轮数
    imgsz=640,                      # 输入图像大小
    batch=16,                       # 批量大小
    name='yoloPConv',          # 实验名称
    project='runs/train',           # 保存路径
    device="cpu"                       # 使用哪个GPU，0表示第一个GPU
)
