import onnx
import onnxruntime as ort
import numpy as np
import cv2

# 加载 ONNX 模型
model_path = "./resources/models/model_det.onnx"  # ch_PP-OCRv3_det_infer
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# 使用 ONNX Runtime 进行推理
ort_session = ort.InferenceSession(model_path)

# 准备输入数据
def preprocess(image_path):
    img = cv2.imread(image_path)
    original_img = img.copy()  # 保留原始图像用于可视化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (960, 960))  # 根据实际模型输入大小调整
    img = img.astype('float32')
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, original_img

input_data, original_img = preprocess("resources/input/ocr/ocr2.png")

# 推理
input_name = ort_session.get_inputs()[0].name
result = ort_session.run(None, {input_name: input_data})

# 解析检测结果并绘制边界框
def postprocess(boxes, original_img):
    # 确保 boxes 是一个二维数组
    if boxes.ndim == 3:
        boxes = boxes[0]

    # 查找轮廓并绘制边界框
    contours, _ = cv2.findContours((boxes * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1 = int(x * original_img.shape[1] / 960)
        y1 = int(y * original_img.shape[0] / 960)
        x2 = int((x + w) * original_img.shape[1] / 960)
        y2 = int((y + h) * original_img.shape[0] / 960)
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return original_img

# 处理推理结果并绘制边界框
result_img = postprocess(result[0][0], original_img)

# 保存和显示结果图片
output_path = 'detected_text_boxes.png'
cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
cv2.imshow("Detected Text Boxes", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Detected text boxes saved to {output_path}')
