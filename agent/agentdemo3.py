from llama_cpp import Llama

# 指定 .gguf 模型文件路径
# model_path = "D:\\Yolo_BitNet_Food\\agent\\BitNet\\models\\BitNet-b1.58-2B-4T\\ggml-model-i2_s.gguf"
model_path = "D:\\Yolo_BitNet_Food\\agent\\BitNet\\models\\BitNet-b1.58-2B-4TSs\\llama-2-7b.Q4_K_M.gguf"

# 加载模型（建议根据你显卡的 VRAM 设置 n_ctx 和 n_gpu_layers）
llm = Llama(
    model_path=model_path,
    n_ctx=2048,               # 上下文窗口大小，可根据显存适当调整
    n_gpu_layers=30,          # 使用 GPU 加速的层数（也可以设为 0 使用 CPU）
    verbose=True              # 显示详细信息
)

# 构建 prompt（没有 chat template 功能需自己构造 prompt）
prompt = "User:I hava a potato can you suggest a meal for me with instructions?\nAssistant:"

# 推理生成
output = llm(prompt, max_tokens=100, stop=["User:", "Assistant:"])

# 打印输出
print("Assistant Response:", output["choices"][0]["text"].strip())
