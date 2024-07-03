import torch

# Load your PyTorch model
model = torch.load("resources/models/Operator_class.pt")
model.eval()

# Create example input tensor
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(model, dummy_input, "model.onnx")