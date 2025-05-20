import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "D:\\Yolo_BitNet_Food\\agent\\BitNet\\models\\BitNet-b1.58-2B-4T"

# 关闭 TorchDynamo（重要）
torch._dynamo.config.suppress_errors = True  # 避免报错
# torch._dynamo.disable()  # 如果仍然启用优化，可以尝试强制禁用（需要较新版本的 PyTorch）

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Windows 上建议使用 float16（视显卡支持情况）
    device_map="auto"  # 不用 "cuda"，可避免某些调度错误
)

# 构造输入
prompt = "System: You are a helpful AI assistant.\nUser: what is your name ?\nAssistant:"
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():  # 防止某些优化路径被触发
    chat_outputs = model.generate(**chat_input, max_new_tokens=50)

response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
print("\nAssistant Response:", response)
