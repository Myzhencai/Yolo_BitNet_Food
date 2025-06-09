from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "D:/Yolo_BitNet_Food/models/best908.onnx"
# model_fp32 = "D:/Yolo_BitNet_Food/runs/train/yoloPConv/weights/best.onnx"
model_int8 = "D:/Yolo_BitNet_Food/models/best908_int8.onnx"
# model_int8 = "D:/Yolo_BitNet_Food/runs/train/yoloPConv/weights/best_int8.onnx"

quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)
