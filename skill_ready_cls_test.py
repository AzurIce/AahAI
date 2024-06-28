import onnxruntime
import numpy as np
import PIL.Image

# 加载模型
onnx_model_path = 'resources/models/skill_ready_cls.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)


# 图形处理部分
input_image_path = 'resources/input/skill_ready/unready/1.png'
input_image = PIL.Image.open(input_image_path)
# 调整图像大小为模型期望的输入尺寸，这里假设模型期望输入为 (64, 64)
input_image = input_image.resize((64, 64))
input_image = np.array(input_image, dtype=np.float32)
# 将通道维度调整为第二个维度
input_image = np.transpose(input_image, (2, 0, 1))
# 添加批处理维度
input_image = np.expand_dims(input_image, axis=0)

# 预测输出
outputs = ort_session.run(None, {'input': input_image})

# 处理输出
predicted_class = np.argmax(outputs[0])

# 打印预测结果
print(f'预测类别索引: {predicted_class}')