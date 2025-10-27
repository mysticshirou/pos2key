from ultralytics import YOLO

hello = YOLO("/models/yolo11n.pt")
hello.export(format="onnx")