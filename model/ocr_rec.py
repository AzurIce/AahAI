import onnxruntime
import numpy as np
from PIL import Image

# 定义模型路径和图像路径
model_path = 'resources/models/model_rec.onnx'
image_path = 'resources/input/ocr/ocr8.png'
# image_path = 'output_with_boxes.png'
# 加载模型
ort_session = onnxruntime.InferenceSession(model_path)

# 读取并预处理输入图像
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 如果图像不是RGB格式，进行转换
    image = image.resize((16, 48))  # 调整图像大小为模型期望的大小
    image.show()
    image = np.array(image).astype(np.float32)  # 转换为numpy数组并转换为float32类型
    image = np.transpose(image, (2, 0, 1))  # 调整通道顺序以匹配模型输入
    image = np.expand_dims(image, axis=0)  # 添加批处理维度
    return image

# 加载并预处理图像
input_data = preprocess_image(image_path)

# 运行模型
outputs = ort_session.run(None, {'x': input_data})

# 处理模型输出
output_tensor = outputs[0]

# print(f"模型的输出形状为: {outputs}")
# 假设输出是一个字符序列的概率分布，获取最可能的字符序列
decoded_text = ""

print(outputs[0].shape)
# print(outputs[0][0][:,2095])
# keys = None
with open("keys.txt", encoding='utf-8') as f:
    keys = f.readlines()


for output in output_tensor[0]:
    print(output)
    # 假设使用 argmax 获取每个位置上概率最高的字符索引
    char_index = np.argmax(output)
    # 假设字符索引对应 ASCII 码，可以用 chr() 转换为字符
    decoded_text += keys[max(char_index, 1124)]

print(f'预测的文本为: {decoded_text}')

#%%
