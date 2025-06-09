from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import onnx
import numpy as np

class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        self.data = [{"images": np.random.rand(1, 3, 640, 640).astype(np.float32)}]  # 适配你的输入名
        self.enum_data = iter(self.data)

    def get_next(self):
        return next(self.enum_data, None)

model_fp32 = "D:/Yolo_BitNet_Food/models/best908.onnx"
model_int8 = "D:/Yolo_BitNet_Food/models/best908_int8static.onnx"

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=DummyCalibrationDataReader(),
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
