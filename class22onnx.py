import cv2
import numpy as np
import onnxruntime
import time


providers = ['DnnlExecutionProvider', 'CPUExecutionProvider','OpenVINOExecutionProvider',"CUDAExecutionProvider"]
# 初始化ONNX Runtime推理会话
onnx_model_path = "D:\\Yolo_BitNet_Food\\models\\best908.onnx"  # 你的ONNX模型路径
# onnx_model_path = "D:\\Yolo_BitNet_Food\\models\\best908_int8.onnx"  # 你的ONNX模型路径
# session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

# YOLOv8 输入尺寸
input_width, input_height = 640, 640

# 视频读取
cap = cv2.VideoCapture("D:\\Yolo_BitNet_Food\\video\\videoplayback2.mp4")  # 视频文件或摄像头 0

# 计算FPS
prev_time = 0

def preprocess(img):
    h, w = img.shape[:2]
    scale = min(input_width / w, input_height / h)
    nw, nh = int(scale * w), int(scale * h)
    img_resized = cv2.resize(img, (nw, nh))
    img_padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    img_padded[(input_height - nh) // 2:(input_height - nh) // 2 + nh,
               (input_width - nw) // 2:(input_width - nw) // 2 + nw, :] = img_resized

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transpose = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(img_transpose, axis=0)
    return input_tensor, scale, (w, h)

def postprocess(outputs, scale, orig_shape, conf_threshold=0.3):
    # 这个函数仍保留，按需要调用（可以不调用也行）
    boxes = []
    scores = []
    class_ids = []

    preds = outputs[0]  # [N, 6]

    for pred in preds:
        score = pred[4]
        if score > conf_threshold:
            x1, y1, x2, y2 = pred[0:4]
            x1 = max(0, min(orig_shape[0], (x1 - (input_width - scale*orig_shape[0])/2) / scale))
            y1 = max(0, min(orig_shape[1], (y1 - (input_height - scale*orig_shape[1])/2) / scale))
            x2 = max(0, min(orig_shape[0], (x2 - (input_width - scale*orig_shape[0])/2) / scale))
            y2 = max(0, min(orig_shape[1], (y2 - (input_height - scale*orig_shape[1])/2) / scale))

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(score))
            class_ids.append(int(pred[5]))

    return boxes, scores, class_ids

# 你需要准备一个类别列表（class names）
class_names = ["person", "bicycle", "car", ...]  # 根据你模型对应的类别

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, scale, orig_shape = preprocess(frame)

    # ONNX推理
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

    # 计算FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # 显示FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()
