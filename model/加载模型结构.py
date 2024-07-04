import onnx

# 加载 ONNX 模型
model = onnx.load('resources/models/model_rec.onnx')

# 打印模型的图结构
print(onnx.helper.printable_graph(model.graph))