from ultralytics import YOLO
import torch
# from multiprocessing import freeze_support

torch.cuda.empty_cache()

def traingdpIou():
    # PConv 训练代码
    model = YOLO('D:\\Yolo_BitNet_Food\\yolo\\ultralytics\\ultralytics\\cfg\\models\\12\\yolo12PConv.yaml')
    # model = YOLO('D:/Yolo_BitNet_Food/runs/train/yoloPConv2/weights/best.pt')

    # 开始训练
    model.train(
        data='D:\\Yolo_BitNet_Food\\Datasets\\data.yaml',
        epochs=2,
        imgsz=640,
        batch=40,
        name='yoloPConv',
        project='runs/train',
        device="0"
    )

if __name__ == '__main__':
    # freeze_support()  # 可选，但建议保留，兼容打包或 Windows
    traingdpIou()
