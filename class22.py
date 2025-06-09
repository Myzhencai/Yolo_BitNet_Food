# # import cv2
# # import time
# # import numpy as np
# # from openvino.runtime import Core
# #
# # # 初始化 OpenVINO
# # ie = Core()
# # model_ir = ie.read_model(model="D:/Yolo_BitNet_Food/models/best908_openvino_model/best908.xml")
# # compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
# # input_layer = compiled_model.input(0)
# # output_layer = compiled_model.output(0)
# #
# # # 参数
# # conf_threshold = 0.3
# # input_size = 640  # 你的模型输入尺寸
# #
# # def preprocess(frame):
# #     img = cv2.resize(frame, (input_size, input_size))
# #     img = img.transpose((2, 0, 1))  # HWC -> CHW
# #     img = np.expand_dims(img, axis=0)
# #     img = img.astype(np.float32) / 255.0
# #     return img
# #
# # def postprocess(results, original_shape):
# #     # results shape: (1, N, 85)
# #     boxes = results[0][:, :4]
# #     scores = results[0][:, 4] * results[0][:, 5:].max(axis=1)
# #     class_ids = results[0][:, 5:].argmax(axis=1)
# #     mask = scores > conf_threshold
# #
# #     boxes = boxes[mask]
# #     scores = scores[mask]
# #     class_ids = class_ids[mask]
# #
# #     # 将相对坐标转换为原图坐标
# #     h, w = original_shape
# #     boxes[:, [0, 2]] *= w  # x1, x2
# #     boxes[:, [1, 3]] *= h  # y1, y2
# #     boxes = boxes.astype(int)
# #
# #     return boxes, scores, class_ids
# #
# # # 打开视频或摄像头
# # cap = cv2.VideoCapture("D:\\Yolo_BitNet_Food\\video\\videoplayback2.mp4")  # 替换为0使用摄像头
# #
# # fps = 0
# # start_time = time.time()
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     h0, w0 = frame.shape[:2]
# #
# #     # 预处理
# #     input_tensor = preprocess(frame)
# #
# #     # 推理
# #     results = compiled_model([input_tensor])[output_layer]
# #
# #     # 解析结果
# #     boxes, scores, class_ids = postprocess(results, (h0, w0))
# #
# #     # 画框和标签
# #     for box, score, cls_id in zip(boxes, scores, class_ids):
# #         x1, y1, x2, y2 = box
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #         label = f"{cls_id}:{score:.2f}"
# #         cv2.putText(frame, label, (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# #
# #     # 计算FPS
# #     end_time = time.time()
# #     fps = 1 / (end_time - start_time)
# #     start_time = end_time
# #
# #     # 显示FPS
# #     cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #
# #     cv2.imshow("YOLOv8 OpenVINO Video Detection", frame)
# #
# #     if cv2.waitKey(1) & 0xFF == 27:  # 按Esc退出
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
# #
# #
#
#
# import os
# import sys
# import cv2
# from collections import Counter
# import textwrap
# import numpy as np
# import torch
# from openvino.runtime import Core
#
#
# def is_image_file(filename):
#     return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
#
#
# def draw_predictions(image, result):
#     boxes = result.boxes.xyxy.cpu().numpy()
#     classes = result.boxes.cls.cpu().numpy().astype(int)
#     scores = result.boxes.conf.cpu().numpy()
#     names = result.names
#
#     # 统计识别结果
#     class_counts = Counter(classes)
#     parts = [f"{count} {names.get(cls_id, str(cls_id))}" for cls_id, count in class_counts.items()]
#     summary = "I have " + " and ".join(parts) + "." if parts else "No objects detected."
#
#     # 绘制检测框和类别标签
#     for (box, cls_id, score) in zip(boxes, classes, scores):
#         x1, y1, x2, y2 = map(int, box)
#         label = f"{names.get(cls_id, str(cls_id))}: {score:.2f}"
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, label, (x1, max(0, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#     # 根据图像尺寸动态设置字体大小
#     image_height, image_width = image.shape[:2]
#     font_scale = max(1.0, image_height / 720 * 1.2)
#     thickness = max(2, int(image_height / 360))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text_color = (255, 255, 255)
#     bg_color = (0, 0, 0)
#
#     # 自动换行逻辑
#     approx_char_width = int(cv2.getTextSize("A", font, font_scale, thickness)[0][0])
#     max_chars_per_line = max(1, image_width // (approx_char_width + 2))
#
#     wrapped_lines = textwrap.wrap(summary, width=max_chars_per_line)
#
#     # 增加行间距
#     line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 20
#     total_text_height = line_height * len(wrapped_lines)
#     y_start = max(line_height + 10, (image_height - total_text_height) // 2)
#
#     for i, line in enumerate(wrapped_lines):
#         (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
#         x = (image_width - text_width) // 2
#         y = y_start + i * line_height
#
#         # 背景框
#         cv2.rectangle(image,
#                       (x - 10, y - text_height - 10),
#                       (x + text_width + 10, y + 10),
#                       bg_color,
#                       thickness=-1)
#
#         # 文本
#         cv2.putText(image, line, (x, y),
#                     font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
#
#     return image
#
#
# def resize_for_display(img, width=1080, height=720):
#     return cv2.resize(img, (width, height))
#
#
# class OpenVINOModel:
#     def __init__(self, xml_path):
#         self.core = Core()
#         self.model = self.core.read_model(model=xml_path)
#         self.compiled_model = self.core.compile_model(self.model, "CPU")
#         self.input_layer = self.compiled_model.input(0)
#         self.output_layer = self.compiled_model.output(0)
#         self.input_shape = self.input_layer.shape  # e.g. [1, 3, 640, 640]
#
#         # TODO: 替换成你自己类别名称字典
#         self.names = {
#             0: "class0",
#             1: "class1",
#             2: "class2",
#             # ...
#         }
#
#     def preprocess(self, image):
#         h, w = self.input_shape[2], self.input_shape[3]
#         img = cv2.resize(image, (w, h))
#         img = img.astype(np.float32) / 255.0
#         img = img.transpose(2, 0, 1)  # HWC -> CHW
#         img = np.expand_dims(img, axis=0)
#         return img
#
#     def postprocess(self, outputs, conf_threshold=0.25):
#         # 根据你的OpenVINO模型输出格式调整此处，以下是一个示例假设：
#         # 输出为 [1, N, 6] -> [x1, y1, x2, y2, score, class_id]
#         results = outputs[self.output_layer]
#         results = results[0]  # 去除batch维度
#
#         boxes = []
#         classes = []
#         scores = []
#
#         for det in results:
#             score = det[4]
#             if score > conf_threshold:
#                 x1, y1, x2, y2 = det[0:4]
#                 cls = int(det[5])
#                 boxes.append([x1, y1, x2, y2])
#                 classes.append(cls)
#                 scores.append(score)
#
#         return boxes, classes, scores
#
#     def predict(self, image, conf=0.25):
#         input_tensor = self.preprocess(image)
#         outputs = self.compiled_model.infer_new_request({self.input_layer: input_tensor})
#         boxes, classes, scores = self.postprocess(outputs, conf_threshold=conf)
#
#         class Box:
#             def __init__(self, boxes, classes, scores):
#                 self.xyxy = torch.tensor(boxes) if boxes else torch.empty((0, 4))
#                 self.cls = torch.tensor(classes) if classes else torch.empty((0,), dtype=torch.int64)
#                 self.conf = torch.tensor(scores) if scores else torch.empty((0,))
#
#         class Result:
#             def __init__(self, xyxy, cls, conf, names):
#                 self.boxes = Box(xyxy, cls, conf)
#                 self.names = names
#
#         return [Result(boxes, classes, scores, self.names)]
#
#
# def predict_and_show(model, source_path, conf=0.25):
#     if os.path.isdir(source_path):
#         for file in sorted(os.listdir(source_path)):
#             file_path = os.path.join(source_path, file)
#             if is_image_file(file_path):
#                 image = cv2.imread(file_path)
#                 results = model.predict(image, conf=conf)
#                 result = results[0]
#                 image = draw_predictions(image, result)
#                 image = resize_for_display(image)
#                 cv2.imshow("Detection", image)
#                 if cv2.waitKey(0) & 0xFF == 10:
#                     break
#         cv2.destroyAllWindows()
#
#     elif is_image_file(source_path):
#         image = cv2.imread(source_path)
#         results = model.predict(image, conf=conf)
#         result = results[0]
#         image = draw_predictions(image, result)
#         image = resize_for_display(image)
#         cv2.imshow("Detection", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#         cap = cv2.VideoCapture(source_path)
#         if not cap.isOpened():
#             print("❌ 无法打开视频")
#             return
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             results = model.predict(frame, conf=conf)
#             result = results[0]
#             frame = draw_predictions(frame, result)
#             frame = resize_for_display(frame)
#             cv2.imshow("Video Detection", frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#
#         cap.release()
#         cv2.destroyAllWindows()
#
#     else:
#         print("❌ 输入路径格式不支持，请输入图片、文件夹或视频路径")
#
#
# if __name__ == "__main__":
#     cfg_model_path_xml = "D:/Yolo_BitNet_Food/models/best908_openvino_model/best908.xml"
#     source_path = 'D:/Yolo_BitNet_Food/video/videoplayback2.mp4'
#     confidence = 0.25
#
#     if not os.path.exists(cfg_model_path_xml):
#         print("❌ OpenVINO模型文件不存在！请检查路径。")
#         sys.exit(1)
#     if not os.path.exists(source_path):
#         print("❌ 输入路径不存在！请检查路径。")
#         sys.exit(1)
#
#     model = OpenVINOModel(cfg_model_path_xml)
#
#     predict_and_show(model, source_path, confidence)


#
# import os
# import sys
# import cv2
# import numpy as np
# from openvino.runtime import Core
# from collections import Counter
# import textwrap
#
# # 类别映射（来自 metadata.yaml）
# CLASS_NAMES = {
#     0: "beet", 1: "bell_pepper", 2: "cabbage", 3: "carrot", 4: "cucumber",
#     5: "egg", 6: "eggplant", 7: "garlic", 8: "onion", 9: "potato",
#     10: "tomato", 11: "zucchini"
# }
#
# # 检查图像类型
# def is_image_file(filename):
#     return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
#
# # 加载 OpenVINO 模型
# def load_model(model_path, device='CPU'):
#     bin_path = model_path.replace(".xml", ".bin")
#     if not os.path.exists(bin_path):
#         raise FileNotFoundError("Missing .bin file for OpenVINO model.")
#     core = Core()
#     model = core.read_model(model=model_path)
#     compiled_model = core.compile_model(model=model, device_name=device)
#     input_layer = compiled_model.input(0)
#     output_layer = compiled_model.output(0)
#     print(f"✅ OpenVINO model loaded to {device}")
#     return compiled_model, input_layer, output_layer
#
# # 图像预处理
# def preprocess_image(image, input_shape):
#     img = cv2.resize(image, (input_shape[3], input_shape[2]))
#     img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
#     return img
#
# # 后处理（默认输出为 [N, 6]：x1,y1,x2,y2,score,class_id）
# # def parse_openvino_output(output, conf_threshold=0.25):
# #     detections = []
# #     for det in output[0]:
# #         score = float(det[4])
# #         if score >= conf_threshold:
# #             x1, y1, x2, y2, cls_id = map(int, det[:4]) + [int(det[5])]
# #             detections.append({
# #                 "box": [x1, y1, x2, y2],
# #                 "score": score,
# #                 "class_id": cls_id
# #             })
# #     return detections
#
# def parse_openvino_output(output, conf_threshold=0.25):
#     detections = []
#     for det in output[0]:
#         score = float(det[4])
#         if score >= conf_threshold:
#             x1, y1, x2, y2 = map(int, det[:4])
#             cls_id = int(det[5])
#             detections.append({
#                 "box": [x1, y1, x2, y2],
#                 "score": score,
#                 "class_id": cls_id
#             })
#     return detections
#
#
# # 结果绘图
# def draw_predictions(image, predictions):
#     h, w = image.shape[:2]
#     summary_counter = Counter()
#     for det in predictions:
#         x1, y1, x2, y2 = det["box"]
#         score = det["score"]
#         cls_id = det["class_id"]
#         label = f"{CLASS_NAMES.get(cls_id, cls_id)}: {score:.2f}"
#         summary_counter[cls_id] += 1
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#     # 生成文字总结
#     parts = [f"{count} {CLASS_NAMES.get(cls_id)}" for cls_id, count in summary_counter.items()]
#     summary = "I have " + " and ".join(parts) + "." if parts else "No objects detected."
#
#     # 自动换行
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = max(1.0, h / 720 * 1.2)
#     thickness = max(2, int(h / 360))
#     approx_char_width = int(cv2.getTextSize("A", font, font_scale, thickness)[0][0])
#     max_chars_per_line = max(1, w // (approx_char_width + 2))
#     wrapped_lines = textwrap.wrap(summary, width=max_chars_per_line)
#     line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 20
#     y_start = max(line_height + 10, (h - line_height * len(wrapped_lines)) // 2)
#     for i, line in enumerate(wrapped_lines):
#         (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
#         x = (w - text_width) // 2
#         y = y_start + i * line_height
#         cv2.rectangle(image, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
#         cv2.putText(image, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
#
#     return image
#
# # 显示尺寸缩放
# def resize_for_display(img, width=1080, height=720):
#     return cv2.resize(img, (width, height))
#
# # 推理并显示结果
# def predict_and_show_openvino(compiled_model, input_layer, output_layer, source_path, conf=0.25):
#     def run_inference(image):
#         input_tensor = preprocess_image(image, input_layer.shape)
#         output = compiled_model({input_layer.any_name: input_tensor})[output_layer]
#         return parse_openvino_output(output, conf)
#
#     if os.path.isdir(source_path):
#         for file in sorted(os.listdir(source_path)):
#             file_path = os.path.join(source_path, file)
#             if is_image_file(file_path):
#                 image = cv2.imread(file_path)
#                 predictions = run_inference(image)
#                 image = draw_predictions(image, predictions)
#                 image = resize_for_display(image)
#                 cv2.imshow("Detection", image)
#                 if cv2.waitKey(0) & 0xFF == 27:
#                     break
#         cv2.destroyAllWindows()
#
#     elif is_image_file(source_path):
#         image = cv2.imread(source_path)
#         predictions = run_inference(image)
#         image = draw_predictions(image, predictions)
#         image = resize_for_display(image)
#         cv2.imshow("Detection", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#         cap = cv2.VideoCapture(source_path)
#         if not cap.isOpened():
#             print("❌ 无法打开视频")
#             return
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             predictions = run_inference(frame)
#             frame = draw_predictions(frame, predictions)
#             frame = resize_for_display(frame)
#             cv2.imshow("Video Detection", frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("❌ 输入路径格式不支持，请输入图片、文件夹或视频路径")
#
# # 主入口
# if __name__ == "__main__":
#     cfg_model_path = "D:/Yolo_BitNet_Food/models/best908_openvino_model/best908.xml"  # OpenVINO 模型路径
#     source_path = 'D:\\Yolo_BitNet_Food\\video\\videoplayback2.mp4'            # 图片/视频/文件夹路径
#     confidence = 0.25
#
#     if not os.path.exists(cfg_model_path):
#         print("❌ 模型文件不存在！请检查路径。")
#         exit(1)
#     if not os.path.exists(source_path):
#         print("❌ 输入路径不存在！请检查路径。")
#         exit(1)
#
#     compiled_model, input_layer, output_layer = load_model(cfg_model_path, device="CPU")
#     predict_and_show_openvino(compiled_model, input_layer, output_layer, source_path, confidence)


import os
import sys
import cv2
import numpy as np
from openvino.runtime import Core
from collections import Counter
import textwrap

# 类别映射（来自 metadata.yaml）
CLASS_NAMES = {
    0: "beet", 1: "bell_pepper", 2: "cabbage", 3: "carrot", 4: "cucumber",
    5: "egg", 6: "eggplant", 7: "garlic", 8: "onion", 9: "potato",
    10: "tomato", 11: "zucchini"
}

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

def load_model(model_path, device='cuda'):
    bin_path = model_path.replace(".xml", ".bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError("Missing .bin file for OpenVINO model.")
    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print(f"✅ Model loaded on {device}")
    return compiled_model, input_layer, output_layer

def preprocess_image(image, input_shape):
    resized = cv2.resize(image, (input_shape[3], input_shape[2]))
    tensor = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
    return tensor, image

def parse_openvino_output(output, conf_threshold=0.25, image_shape=None):
    detections = []
    h, w = image_shape[:2] if image_shape is not None else (1, 1)
    for det in output[0]:
        score = float(det[4])
        if score >= conf_threshold:
            x1, y1, x2, y2 = det[:4]
            if max(x1, y1, x2, y2) <= 1.5:  # 判定是否归一化坐标
                x1 *= w
                x2 *= w
                y1 *= h
                y2 *= h
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(round(det[5]))
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": score,
                "class_id": cls_id
            })
    return detections

def draw_predictions(image, predictions):
    h, w = image.shape[:2]
    counter = Counter()
    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        score = pred["score"]
        cls_id = pred["class_id"]
        label = f"{CLASS_NAMES.get(cls_id, 'Unknown')}: {score:.2f}"
        counter[cls_id] += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 文本总结
    parts = [f"{count} {CLASS_NAMES.get(cls_id, 'Unknown')}" for cls_id, count in counter.items()]
    summary = "I have " + " and ".join(parts) + "." if parts else "No objects detected."

    # 动态文字绘制
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, h / 720 * 1.2)
    thickness = max(2, int(h / 360))
    approx_char_width = int(cv2.getTextSize("A", font, font_scale, thickness)[0][0])
    max_chars_per_line = max(1, w // (approx_char_width + 2))
    wrapped_lines = textwrap.wrap(summary, width=max_chars_per_line)
    line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 20
    y_start = max(line_height + 10, (h - line_height * len(wrapped_lines)) // 2)

    for i, line in enumerate(wrapped_lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (w - text_width) // 2
        y = y_start + i * line_height
        cv2.rectangle(image, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(image, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image

def resize_for_display(img, width=1080, height=720):
    return cv2.resize(img, (width, height))

def predict_and_show_openvino(compiled_model, input_layer, output_layer, source_path, conf=0.25):
    def run_inference(image):
        tensor, original = preprocess_image(image, input_layer.shape)
        output = compiled_model({input_layer.any_name: tensor})[output_layer]
        print(f"[DEBUG] output shape: {output.shape}, example row: {output[0][0]}")
        return parse_openvino_output(output, conf, image_shape=original.shape)

    if os.path.isdir(source_path):
        for file in sorted(os.listdir(source_path)):
            file_path = os.path.join(source_path, file)
            if is_image_file(file_path):
                image = cv2.imread(file_path)
                predictions = run_inference(image)
                image = draw_predictions(image, predictions)
                image = resize_for_display(image)
                cv2.imshow("Detection", image)
                if cv2.waitKey(0) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()

    elif is_image_file(source_path):
        image = cv2.imread(source_path)
        predictions = run_inference(image)
        image = draw_predictions(image, predictions)
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
            predictions = run_inference(frame)
            frame = draw_predictions(frame, predictions)
            frame = resize_for_display(frame)
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("❌ 输入路径格式不支持，请输入图片、文件夹或视频路径")

if __name__ == "__main__":
    cfg_model_path = "D:/Yolo_BitNet_Food/models/best908_openvino_model/best908.xml"
    source_path = "D:/Yolo_BitNet_Food/video/videoplayback2.mp4"
    confidence = 0.25

    if not os.path.exists(cfg_model_path):
        print("❌ 模型文件不存在！请检查路径。")
        sys.exit(1)
    if not os.path.exists(source_path):
        print("❌ 输入路径不存在！请检查路径。")
        sys.exit(1)

    compiled_model, input_layer, output_layer = load_model(cfg_model_path, device="CPU")
    predict_and_show_openvino(compiled_model, input_layer, output_layer, source_path, confidence)

