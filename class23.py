# import os
# import sys
# import cv2
# from ultralytics import YOLO
# from collections import Counter
# import subprocess
# import textwrap
# from concurrent.futures import ThreadPoolExecutor
# import threading
#
# lock = threading.Lock()
#
# def run_command(command, shell=False):
#     try:
#         subprocess.run(command, shell=shell, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred while running command: {e}")
#         sys.exit(1)
#
# def load_model(model_path, device='cpu'):
#     model = YOLO(model_path)
#     model.to(device)
#     print(f"✅ Model loaded to {device}")
#     return model
#
# def load_modelonnx(model_path, device='cpu'):
#     model = YOLO(model_path)
#     print(f"✅ Model loaded: {model_path}")
#     return model
#
# def is_image_file(filename):
#     return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
#
# def draw_predictions(image, result):
#     boxes = result.boxes.xyxy.cpu().numpy()
#     classes = result.boxes.cls.cpu().numpy().astype(int)
#     scores = result.boxes.conf.cpu().numpy()
#     names = result.names
#
#     class_counts = Counter(classes)
#     parts = [f"{count} {names[cls_id]}" for cls_id, count in class_counts.items()]
#     summary = "I have " + " and ".join(parts) + "." if parts else "No objects detected."
#
#     for (box, cls_id, score) in zip(boxes, classes, scores):
#         x1, y1, x2, y2 = map(int, box)
#         label = f"{names[cls_id]}: {score:.2f}"
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, label, (x1, max(0, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#     image_height, image_width = image.shape[:2]
#     font_scale = max(1.0, image_height / 720 * 1.2)
#     thickness = max(2, int(image_height / 360))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text_color = (255, 255, 255)
#     bg_color = (0, 0, 0)
#
#     approx_char_width = int(cv2.getTextSize("A", font, font_scale, thickness)[0][0])
#     max_chars_per_line = max(1, image_width // (approx_char_width + 2))
#     wrapped_lines = textwrap.wrap(summary, width=max_chars_per_line)
#
#     line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 20
#     total_text_height = line_height * len(wrapped_lines)
#     y_start = max(line_height + 10, (image_height - total_text_height) // 2)
#
#     for i, line in enumerate(wrapped_lines):
#         (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
#         x = (image_width - text_width) // 2
#         y = y_start + i * line_height
#
#         cv2.rectangle(image,
#                       (x - 10, y - text_height - 10),
#                       (x + text_width + 10, y + 10),
#                       bg_color,
#                       thickness=-1)
#         cv2.putText(image, line, (x, y),
#                     font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
#
#     return image
#
# def resize_for_display(img, width=1080, height=720):
#     return cv2.resize(img, (width, height))
#
# def process_image(file_path, model, conf):
#     results = model.predict(source=file_path, conf=conf, save=False)
#     result = results[0]
#     image = cv2.imread(file_path)
#     image = draw_predictions(image, result)
#     image = resize_for_display(image)
#
#     with lock:
#         cv2.imshow("Detection", image)
#         if cv2.waitKey(0) & 0xFF == 10:
#             return "stop"
#     return "continue"
#
# def predict_and_show(model, source_path, conf=0.25):
#     if os.path.isdir(source_path):
#         image_files = [f for f in sorted(os.listdir(source_path)) if is_image_file(f)]
#         image_paths = [os.path.join(source_path, f) for f in image_files]
#
#         with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#             futures = []
#             for path in image_paths:
#                 futures.append(executor.submit(process_image, path, model, conf))
#
#             for future in futures:
#                 result = future.result()
#                 if result == "stop":
#                     break
#
#         cv2.destroyAllWindows()
#
#     elif is_image_file(source_path):
#         results = model.predict(source=source_path, conf=conf, save=False)
#         result = results[0]
#         image = cv2.imread(source_path)
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
#             results = model.predict(source=frame, conf=conf, save=False)
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
# if __name__ == "__main__":
#     cfg_model_path = 'models/best908.onnx'
#     source_path = 'D:\\Yolo_BitNet_Food\\video\\videoplayback2.mp4'
#     confidence = 0.25
#
#     if not os.path.exists(cfg_model_path):
#         print("❌ 模型文件不存在！请检查路径。")
#         exit(1)
#     if not os.path.exists(source_path):
#         print("❌ 输入路径不存在！请检查路径。")
#         exit(1)
#
#     model = load_modelonnx(cfg_model_path)
#     predict_and_show(model, source_path, confidence)

import os
import cv2
import sys
import textwrap
from collections import Counter
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))


def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{i:05d}.jpg"), frame)
        i += 1
    cap.release()
    print(f"✅ Extracted {i} frames to {output_folder}")


def draw_predictions(image, result):
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    names = result.names

    class_counts = Counter(classes)
    parts = [f"{count} {names[cls_id]}" for cls_id, count in class_counts.items()]
    summary = "I have " + " and ".join(parts) + "." if parts else "No objects detected."

    for (box, cls_id, score) in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    image_height, image_width = image.shape[:2]
    font_scale = max(1.0, image_height / 720 * 1.2)
    thickness = max(2, int(image_height / 360))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    approx_char_width = int(cv2.getTextSize("A", font, font_scale, thickness)[0][0])
    max_chars_per_line = max(1, image_width // (approx_char_width + 2))
    wrapped_lines = textwrap.wrap(summary, width=max_chars_per_line)

    line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 20
    total_text_height = line_height * len(wrapped_lines)
    y_start = max(line_height + 10, (image_height - total_text_height) // 2)

    for i, line in enumerate(wrapped_lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (image_width - text_width) // 2
        y = y_start + i * line_height

        cv2.rectangle(image,
                      (x - 10, y - text_height - 10),
                      (x + text_width + 10, y + 10),
                      bg_color,
                      thickness=-1)
        cv2.putText(image, line, (x, y),
                    font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    return image


def process_frame(args):
    frame_path, model_path, conf, output_folder = args
    try:
        model = YOLO(model_path)
        result = model.predict(source=frame_path, conf=conf, save=False)[0]
        image = cv2.imread(frame_path)
        image = draw_predictions(image, result)
        cv2.imwrite(os.path.join(output_folder, os.path.basename(frame_path)), image)
        return frame_path
    except Exception as e:
        print(f"❌ Error processing {frame_path}: {e}")
        return None


def predict_frames_parallel(input_folder, model_path, output_folder, conf=0.25):
    os.makedirs(output_folder, exist_ok=True)
    frame_files = [f for f in os.listdir(input_folder) if is_image_file(f)]
    args_list = [(os.path.join(input_folder, f), model_path, conf, output_folder) for f in frame_files]

    print(f"🚀 Starting multiprocessing with {cpu_count()} cores for {len(args_list)} frames")

    with Pool(cpu_count()) as pool:
        pool.map(process_frame, args_list)

    print(f"✅ All frames processed and saved to {output_folder}")


def frames_to_video(input_folder, output_path, fps=30):
    image_files = sorted([f for f in os.listdir(input_folder) if is_image_file(f)])
    if not image_files:
        print("❌ No frames found.")
        return

    first_frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in image_files:
        frame = cv2.imread(os.path.join(input_folder, f))
        out.write(frame)

    out.release()
    print(f"🎉 Final video saved to {output_path}")


if __name__ == "__main__":
    # === 配置 ===
    video_path = "D:/Yolo_BitNet_Food/video/videoplayback2.mp4"
    model_path = "models/best908.onnx"
    extracted_dir = "video_frames"
    processed_dir = "processed_frames"
    output_video_path = "result_video.mp4"
    confidence = 0.25
    fps = 30  # 可修改为实际视频的帧率

    # === 流程 ===
    if not os.path.exists(video_path):
        print("❌ 视频路径不存在")
        sys.exit(1)

    if not os.path.exists(model_path):
        print("❌ 模型路径不存在")
        sys.exit(1)

    extract_frames(video_path, extracted_dir)
    predict_frames_parallel(extracted_dir, model_path, processed_dir, confidence)
    frames_to_video(processed_dir, output_video_path, fps)
