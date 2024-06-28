import onnxruntime
import numpy as np
import cv2

# 加载模型
onnx_model_path = 'resources/models/operators_det.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)


# 图形处理部分
input_image_path = 'resources/input/operators_det/1.png'
image = cv2.imread(input_image_path)
resized_image = cv2.resize(image, (640, 640))  # 缩放到640x640

# 将图像转换为模型所需的格式（BGR到RGB，并转换为浮点数）
input_image = resized_image[:, :, ::-1].astype(np.float32) / 255.0
input_image = np.transpose(input_image, [2, 0, 1])  # 转换为CHW格式
input_image = np.expand_dims(input_image, axis=0)  # 添加batch维度

# 进行推理
outputs = ort_session.run(None, {"input": input_image})

# 解析输出结果
# 假设YOLOv8 N的输出在outputs[0]中
predictions = parse_yolo_output(outputs[0])

# 处理预测结果
for pred in predictions:
    class_label = pred['class']
    confidence = pred['confidence']
    bbox = pred['bbox']
    # 进一步处理角色血量的逻辑

# 输出预测结果
print(predictions)