import os
import sys
import cv2
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import subprocess
import textwrap
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "D:\\Yolo_BitNet_Food\\agent\\BitNet\\models\\BitNet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
chatmodel = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

def run_command(command, shell=False):
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)


def load_model(model_path, device='cpu'):
    model = YOLO(model_path)
    model.to(device)
    print(f"✅ Model loaded to {device}")
    return model


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))


def draw_predictions(image, result):
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    names = result.names

    # 统计识别结果
    class_counts = Counter(classes)
    parts = [f"{count} {names[cls_id]}" for cls_id, count in class_counts.items()]
    summary = "I have " + " and ".join(parts) + "." if parts else "No objects detected."

    # 绘制检测框和类别标签
    for (box, cls_id, score) in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 根据图像尺寸动态设置字体大小
    image_height, image_width = image.shape[:2]
    font_scale = max(1.0, image_height / 720 * 1.2)
    thickness = max(2, int(image_height / 360))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # 自动换行逻辑
    approx_char_width = int(cv2.getTextSize("A", font, font_scale, thickness)[0][0])
    max_chars_per_line = max(1, image_width // (approx_char_width + 2))

    wrapped_lines = textwrap.wrap(summary, width=max_chars_per_line)

    # 增加行间距
    line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 20
    total_text_height = line_height * len(wrapped_lines)
    y_start = max(line_height + 10, (image_height - total_text_height) // 2)

    for i, line in enumerate(wrapped_lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (image_width - text_width) // 2
        y = y_start + i * line_height

        # 背景框
        cv2.rectangle(image,
                      (x - 10, y - text_height - 10),
                      (x + text_width + 10, y + 10),
                      bg_color,
                      thickness=-1)

        # 文本
        cv2.putText(image, line, (x, y),
                    font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    return image


def resize_for_display(img, width=1080, height=720):
    return cv2.resize(img, (width, height))


def predict_and_show(model, source_path, conf=0.25):
    if os.path.isdir(source_path):
        for file in sorted(os.listdir(source_path)):
            file_path = os.path.join(source_path, file)
            if is_image_file(file_path):
                results = model.predict(source=file_path, conf=conf, save=False)
                result = results[0]
                image = cv2.imread(file_path)
                image = draw_predictions(image, result)
                image = resize_for_display(image)
                cv2.imshow("Detection", image)
                if cv2.waitKey(0) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()

    elif is_image_file(source_path):
        results = model.predict(source=source_path, conf=conf, save=False)
        result = results[0]
        image = cv2.imread(source_path)
        image = draw_predictions(image, result)
        image = resize_for_display(image)
        cv2.imshow("Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print("❌ 无法打开视频")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf, save=False)
            result = results[0]
            frame = draw_predictions(frame, result)
            frame = resize_for_display(frame)
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("❌ 输入路径格式不支持，请输入图片、文件夹或视频路径")

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
    cfg_model_path = 'models/best 908.pt'        # 模型路径
    source_path = 'image/11.png'                 # 可为图片、视频或文件夹
    # source_path = 'D:\\Yolo_BitNet_Food\\image'                 # 可为图片、视频或文件夹
    # source_path = 'D:\\Yolo_BitNet_Food\\video\\videoplayback2.mp4'                 # 可为图片、视频或文件夹
    confidence = 0.25

    if not os.path.exists(cfg_model_path):
        print("❌ 模型文件不存在！请检查路径。")
        exit(1)
    if not os.path.exists(source_path):
        print("❌ 输入路径不存在！请检查路径。")
        exit(1)

    model = load_model(cfg_model_path, device='cuda')
    # predict_and_show(model, source_path, confidence)
    summaryprompt = predict_and_save2(model, source_path, confidence, "result.png")

    # Apply the chat template
    messages = [
        {"role": "system", "content": f"You are a helpful AI assistant"},
        {"role": "user","content": f"{summaryprompt} Please suggest a dish for me with instruction?"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    chat_outputs = chatmodel.generate(**chat_input, max_new_tokens=300,temperature=0.6)
    response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:],
                                skip_special_tokens=True)  # Decode only the response part
    print("\nAssistant Response:", response)
