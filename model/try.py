import cv2
import numpy as np
import onnxruntime as ort

# 初始化ONNX运行时会话
det_model_path = "./resources/models/model_det.onnx"
rec_model_path = "./resources/models/model_rec.onnx"
cls_model_path = "./resources/models/model_cls.onnx"

det_sess = ort.InferenceSession(det_model_path)
rec_sess = ort.InferenceSession(rec_model_path)
cls_sess = ort.InferenceSession(cls_model_path)

# 加载图像
image_path = './resources/input/ocr/ocr2.png'
image = cv2.imread(image_path)

# 预处理图像进行检测
def preprocess_det_image(image):
    h, w, _ = image.shape
    scale = 640 / max(h, w)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))
    image = image.astype('float32')
    image = image / 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# 检测文本区域
def detect_text(det_sess, image):
    det_input = preprocess_det_image(image)
    det_output = det_sess.run(None, {'x': det_input})[0]
    # TODO: 解析det_output以获得文本框位置
    # 假设我们得到了文本框 boxes
    boxes = []  # 这里你需要根据det_output解析出实际的文本框
    return boxes

# 预处理图像进行方向分类
def preprocess_cls_image(image, box):
    x, y, w, h = cv2.boundingRect(np.array(box))
    cropped = image[y:y+h, x:x+w].copy()
    cropped = cv2.resize(cropped, (cls_sess.get_inputs()[0].shape[3], cls_sess.get_inputs()[0].shape[2]))
    cropped = cropped.astype('float32')
    cropped = cropped / 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cropped = (cropped - mean) / std
    cropped = np.transpose(cropped, (2, 0, 1))
    cropped = np.expand_dims(cropped, axis=0)
    return cropped

# 校正文本方向
def classify_text_direction(cls_sess, image, boxes):
    corrected_boxes = []
    for box in boxes:
        cls_input = preprocess_cls_image(image, box)
        cls_output = cls_sess.run(None, {'x': cls_input})[0]
        # TODO: 解析cls_output以校正文本方向
        corrected_boxes.append(box)  # 这里你需要根据cls_output校正box
    return corrected_boxes

# 预处理图像进行识别
def preprocess_rec_image(image, box):
    x, y, w, h = cv2.boundingRect(np.array(box))
    cropped = image[y:y+h, x:x+w].copy()
    cropped = cv2.resize(cropped, (rec_sess.get_inputs()[0].shape[3], rec_sess.get_inputs()[0].shape[2]))
    cropped = cropped.astype('float32')
    cropped = cropped / 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cropped = (cropped - mean) / std
    cropped = np.transpose(cropped, (2, 0, 1))
    cropped = np.expand_dims(cropped, axis=0)
    return cropped

# 识别文本
def recognize_text(rec_sess, image, boxes):
    results = []
    for box in boxes:
        rec_input = preprocess_rec_image(image, box)
        rec_output = rec_sess.run(None, {'x': rec_input})[0]
        # TODO: 解析rec_output以获得文本和置信度
        text = "dummy_text"  # 这里你需要根据rec_output解析出实际的文本
        confidence = 1.0     # 这里你需要根据rec_output解析出实际的置信度
        results.append((text, confidence))
    return results

# 主函数
def main():
    # 步骤1：文本检测
    boxes = detect_text(det_sess, image)

    # 步骤2：方向分类（如果需要）
    corrected_boxes = classify_text_direction(cls_sess, image, boxes)

    # 步骤3：文本识别
    results = recognize_text(rec_sess, image, corrected_boxes)

    # 打印识别结果
    for text, confidence in results:
        print(f'Text: {text}, Confidence: {confidence}')

if __name__ == '__main__':
    main()
