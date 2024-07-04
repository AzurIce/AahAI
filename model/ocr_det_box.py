import onnx
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image, ImageDraw

# 加载 ONNX 模型
model_path = "./resources/models/model_det.onnx"  # ch_PP-OCRv3_det_infer
print("Loading ONNX model...")
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print("Model loaded and checked successfully.")

# 使用 ONNX Runtime 进行推理
print("Initializing ONNX Runtime session...")
ort_session = ort.InferenceSession(model_path)

# 准备输入数据
def preprocess(image_path):
    print(f"Preprocessing image: {image_path}")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (960, 960))
    img_normalized = img_resized.astype('float32') / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_expanded = np.expand_dims(img_transposed, axis=0)
    return img_expanded, img_resized

input_data, img_resized = preprocess("resources/input/ocr/ocr2.png")
print("Input data prepared.")

# 推理
input_name = ort_session.get_inputs()[0].name
print(f"Running inference...")
result = ort_session.run(None, {input_name: input_data})
print("Inference completed.")

# 提取二维数组 (假设输出形状是 (1, 1, H, W))
print("Processing output result...")
result = result[0][0][0]

# 将结果转换为二值图像
binary_image = (result > 0).astype(np.uint8) * 255

# 查找所有文本区域的边界框
print("Finding contours...")
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = [cv2.boundingRect(contour) for contour in contours]
print(f"Found {len(boxes)} bounding boxes.")

# 在预处理后的图像上绘制检测框
img_pil = Image.fromarray(img_resized)
draw = ImageDraw.Draw(img_pil)
for box in boxes:
    x, y, w, h = box
    # 扩大边界框
    padding = 0  # 可以调整这个值以控制边界框的扩展量
    draw.rectangle([x - padding, y - padding, x + w + padding, y + h + padding], outline="red")
    cropped = img_pil.crop([x - padding, y - padding, x + w + padding, y + h + padding])
    # cropped.save("output.png")

# 保存和显示结果图像
output_image_path = 'output_with_boxes.png'
img_pil.save(output_image_path)
img_pil.show()

print(f'Image with detection boxes saved to {output_image_path}')

#%%
