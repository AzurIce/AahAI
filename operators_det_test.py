import cv2
import numpy as np
import onnxruntime as ort


# 定义函数用于将图像转换为模型输入的张量
def image_to_tensor(image):
    image = image.astype(np.float32)
    image /= 255.0  # 图像归一化
    return image.transpose(2, 0, 1)  # 调整通道顺序，转换为CHW格式


# 加载模型
session = ort.InferenceSession("resources/models/operators_det.onnx")


# 读取并预处理输入图像
image_path = "resources/input/operators_det/1.png"
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
input_data = image_to_tensor(image_resized)

# 构建模型输入
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = (1,) + input_data.shape
input_data = input_data.reshape((1,) + input_data.shape).astype(np.float32)

# 运行推理
outputs = session.run([output_name], {input_name: input_data})
output_data = outputs[0]

# 解析模型输出
raw_output = output_data.flatten()
output_shape = output_data.shape
output = [[] for _ in range(output_shape[1])]
for i in range(output_shape[1]):
    output[i] = raw_output[i * output_shape[2]:(i + 1) * output_shape[2]]

print(output[-1])
print(len(output[-1]))

# exit(0)

# 根据模型输出生成结果
results = []
confidences = output[-1]
for i in range(len(confidences)):
    score = confidences[i]
    if score < 0.3:  # 阈值设定
        continue

    center_x = int(output[0][i] / (640 / image.shape[1]))
    center_y = int(output[1][i] / (640 / image.shape[0]))
    w = int(output[2][i] / (640 / image.shape[1]))
    h = int(output[3][i] / (640 / image.shape[0]))

    x = center_x - w // 2
    y = center_y - h // 2
    results.append((x, y, w, h, score))

# 在原始图像上绘制结果
for (x, y, w, h, score) in results:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 显示或保存结果图像
cv2.imshow("Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = 'resources/output/operators_det/operators_det_1.png'  # Specify the path where you want to save the image
cv2.imwrite(output_path, image)  # Save the image to the specified path

print(f"Image saved successfully at {output_path}")