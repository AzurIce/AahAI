from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO("models/DeploymentReadyModel/ready6-6.pt")
model = YOLO("models/MapDeploymentModel/map1-1.pt")
# Export the model to ONNX format
model.export(format="onnx")
