import onnx
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# 加载 ONNX 模型
model_path = "./resources/models/model_det.onnx"  # ch_PP-OCRv3_det_infer
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# 使用 ONNX Runtime 进行推理
ort_session = ort.InferenceSession(model_path)

# 准备输入数据
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (960, 960))
    img = img.astype('float32')
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

input_data = preprocess("resources/input/ocr/ocr5.png")

# 推理
input_name = ort_session.get_inputs()[0].name
result = ort_session.run(None, {input_name: input_data})

# 处理结果
print(result)

# 提取二维数组 (假设输出形状是 (1, 1, H, W))
result = result[0][0][0]

# 获取数组的高度和宽度
height, width = result.shape

# 创建一张新的黑白图片
image = Image.new('1', (width, height))

# 遍历二维数组并设置像素值
for i in range(height):
    for j in range(width):
        if result[i, j] != 0:
            image.putpixel((j, i), 0)  # 黑色像素
        else:
            image.putpixel((j, i), 1)  # 白色像素

# 保存图片
output_path = 'black_white_image.png'
image.save(output_path)

# 显示图片
image.show()

print(f'Black and white image saved to {output_path}')
