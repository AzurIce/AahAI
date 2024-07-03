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


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


from shapely.geometry import Polygon
import pyclipper


def unclip(box, unclip_ratio):
    poly = Polygon(box)
    # print(poly)
    if poly.length < 0.001:
        return None
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded


# 解析检测结果并绘制边界框
def postprocess(boxes, original_img):
    # 查找轮廓并绘制边界框
    contours, _ = cv2.findContours((boxes * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        box, _ = get_mini_boxes(contour)
        # print(box)
        points = [[int(x * original_img.shape[1] / 960), int(y * original_img.shape[0] / 960)] for [x, y] in box]
        # box = unclip(points, 2.0)
        # print(box)
        # x, y, w, h = cv2.boundingRect(contour)
        # x1 = int(x * original_img.shape[1] / 960)
        # y1 = int(y * original_img.shape[0] / 960)
        # x2 = int((x + w) * original_img.shape[1] / 960)
        # y2 = int((y + h) * original_img.shape[0] / 960)
        # print(points)
        if box is None or len(box) > 1:
            continue
        try:
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = get_mini_boxes(box)
            box = [[int(x), int(y)] for [x, y] in box]
            print(box)
            cv2.rectangle(original_img, box[0], box[2], (0, 255, 0), 2)
        except:
            pass

    return original_img


# 处理推理结果并绘制边界框
result_img = postprocess(result[0][0][0], original_img)

# 保存和显示结果图片
output_path = 'detected_text_boxes.png'
cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
cv2.imshow("Detected Text Boxes", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Detected text boxes saved to {output_path}')
