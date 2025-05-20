from PIL import Image
from ultralytics import YOLO
from collections import Counter
import os


def load_model(model_path, device='cpu'):
    model = YOLO(model_path)
    model.to(device)
    print(f"Model loaded to {device}")
    return model

def predict_and_save2(model, img_path, conf=0.25, save_path="result.jpg"):
    # 模型预测
    results = model.predict(source=img_path, conf=conf, save=False)
    result = results[0]

    # 获取预测框相关信息
    boxes = result.boxes.xyxy.cpu().numpy()       # 预测框的坐标 [x1, y1, x2, y2]
    classes = result.boxes.cls.cpu().numpy().astype(int)  # 类别索引
    scores = result.boxes.conf.cpu().numpy()      # 置信度

    # 统计识别结果
    class_counts = Counter(classes)

    # 构建描述字符串
    parts = []
    for cls_id, count in class_counts.items():
        cls_name = result.names[cls_id]
        parts.append(f"{count} {cls_name}")
    summary = "I have " + " and ".join(parts) + "."
    print(summary)

    # 在图像上绘制预测结果
    im_array = result.plot()
    im = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
    im.save(save_path)
    print(f"Saved annotated image to {save_path}")

    return summary


if __name__ == "__main__":
    # 模型路径和图片路径
    cfg_model_path = 'models/best 908.pt'
    image_path = 'image/upload.jpg'  # 修改为你的图片路径
    # image_path = 'image/1.png'  # 修改为你的图片路径
    output_path = 'image/output.jpg'
    confidence = 0.25

    if not os.path.exists(cfg_model_path):
        print("❌ 模型文件不存在！请检查路径。")
        exit(1)
    if not os.path.exists(image_path):
        print("❌ 图片文件不存在！请检查路径。")
        exit(1)

    model = load_model(cfg_model_path, device='cuda')
    # model = load_model(cfg_model_path, device='cpu')
    # model = load_model(cfg_model_path)
    summaryprompt = predict_and_save2(model, image_path, confidence, output_path)



