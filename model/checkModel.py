import onnx

def print_model_info(model_path):
    # 加载模型
    model = onnx.load(model_path)
    print(f"Model: {model_path}")

    # 打印模型的输入信息
    print("Model inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        input_type = input.type.tensor_type
        print(f"Type: {onnx.TensorProto.DataType.Name(input_type.elem_type)}")
        shape = [dim.dim_value for dim in input_type.shape.dim]
        print(f"Shape: {shape}")

    # 打印模型的输出信息
    print("Model outputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        output_type = output.type.tensor_type
        print(f"Type: {onnx.TensorProto.DataType.Name(output_type.elem_type)}")
        shape = [dim.dim_value for dim in output_type.shape.dim]
        print(f"Shape: {shape}")
    print("\n")

# 模型路径
model_paths = [
    # 'resources/models/skill_ready_cls.onnx',
    # 'resources/models/deploy_direction_cls.onnx',
    # 'resources/models/operators_det.onnx',
    # 'resources/models/model_det.onnx',
    'resources/models/model_rec.onnx'
]

# 打印每个模型的信息
for model_path in model_paths:
    print_model_info(model_path)
