import os
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import os
import sys
import signal
import platform
import subprocess
from google import genai

# 初始化Google GenAI客户端
client = genai.Client(api_key="AIzaSyDnJ-HAfTYa7hdO4V2xXhuIAbElsjfxtSI")



def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference(promptsvalue):
    # 设定默认参数
    model = "D:\\JudyYolo\\angent\\BitNet\\models\\BitNet-b1.58-2B-4T\\ggml-model-i2_s.gguf"
    n_predict = 60000
    prompt = promptsvalue
    threads = 6
    ctx_size = 2048
    temperature = 0.8
    conversation = True

    build_dir = "D:\\JudyYolo\\angent\\BitNet\\build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")

    command = [
        f'{main_path}',
        '-m', model,
        '-n', str(n_predict),
        '-t', str(threads),
        '-p', prompt,
        '-ngl', '0',
        '-c', str(ctx_size),
        '--temp', str(temperature),
        "-b", "1",
    ]
    if conversation:
        command.append("-cnv")

    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)



def load_model(model_path, device='cpu'):
    model = YOLO(model_path)
    model.to(device)
    print(f"Model loaded to {device}")
    return model

def predict_and_save(model, img_path, conf=0.25, save_path="result.jpg"):
    results = model.predict(source=img_path, conf=conf, save=False)
    result = results[0]

    # 统计识别结果
    class_counts = Counter(result.boxes.cls.numpy().astype(int))

    # 构建描述字符串
    parts = []
    for cls_id, count in class_counts.items():
        cls_name = result.names[cls_id]
        parts.append(f"{count} {cls_name}")

    summary = "I have " + " and ".join(parts) + "."
    print(summary)

    return summary

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

    model = load_model(cfg_model_path, device='cpu')
    summaryprompt = predict_and_save2(model, image_path, confidence, output_path)

    # signal.signal(signal.SIGINT, signal_handler)
    # promptsvalue = summaryprompt + " can you suggest me a dish ?"
    # run_inference(promptsvalue)
    promptdemo = """
    材料准备
    🥔 主料
    土豆 2 个（约 400 克）
    下厨房

    青椒/尖椒 2–3 个（约 100 克）
    下厨房

    🥩 可选辅料
    肉丝 50 克（不放肉也可，全素版本）
    下厨房

    🌱 调料
    食用油 适量
    葱、蒜 各 1 小把（切末）
    下厨房
    料酒 1 茶勺
    生抽 1 大勺
    白糖 ½ 茶勺（可选）
    盐 适量

    操作步骤
    🔪 切丝泡水
    土豆去皮切细丝，放入清水中浸泡 5 分钟以去除多余淀粉，水清后捞出沥干
    下厨房

    辣椒去蒂切丝，葱蒜切末备用

    🍳 过油或干炒
    锅中倒入适量油，烧至五成热（约 150℃）
    下入土豆丝大火快速翻炒 1–2 分钟，至土豆“断生”即可捞出控油
    搜狐

    🍳 爆香调味
    锅留底油，先下葱末、蒜末爆香约 10 秒
    倒入肉丝翻炒至变色，加入料酒、生抽快速翻炒均匀

    🔄 合炒收汁
    将控干水的土豆丝倒回锅中，大火翻炒至入味
    加入辣椒丝继续翻炒 30 秒
    撒入盐、白糖，再翻炒 10–15 秒即可出锅
    下厨房

    小贴士
    💧 控水要充分：土豆丝切好后务必控净表面水分，否则易“爆油”或出水​
    m.meishiq.com

    🔥 火候要足：整个过程均用大火快炒，才能保持土豆脆爽口感​
    m.meishiq.com

    🕒 时间要短：土豆丝和辣椒都不宜久炒，保持爽脆最佳​
    下厨房

    🥄 调料随喜好添加：可根据口味增减糖、醋或辣椒种类

    通过以上简单几步，您就能在家快速做出这道下饭又开胃的土豆炒辣椒。享用时可搭配米饭或馒头，一同品尝家常的魅力！
    """

    # 构造新的内容
    # prompt = f"{summaryprompt} Can you suggest a dish for me? 用中文回答并参考{promptdemo}内容格式生成结果注意保留小图标"
    prompt = f"{summaryprompt} Can you suggest a dish for me? 用中文答复，请在回答过程中使用类似于🔪这些厨房类的小图标，每个步骤都添加可以参照的格式是:{promptdemo}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    print(response.text)  # 输出生成的内容
