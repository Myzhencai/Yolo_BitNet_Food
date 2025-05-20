import torch
torch.cuda.empty_cache()
from ultralytics import YOLO


def trainghost():
    model = YOLO('D:\\Yolo_BitNet_Food\\yolo\\ultralytics\\ultralytics\\cfg\\models\\12\\yolo12GhostNet.yaml')

    model.train(
        data='D:\\Yolo_BitNet_Food\\Datasets\\data.yaml',
        epochs=1,
        imgsz=640,
        batch=8,
        name='yoloGhost',
        project='runs/train',
        device="0"
    )

if __name__ == '__main__':
    trainghost()
