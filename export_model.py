"""
Script to export YOLOv8 model to ONNX format for faster inference
Run this once to create the optimized model
"""
from ultralytics import YOLO

print("Exporting YOLOv8n to ONNX format...")
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=320)
print("✅ Export complete! yolov8n.onnx created")

print("\nOptional: Exporting to OpenVINO format (best for Intel CPUs)...")
try:
    model.export(format="openvino", imgsz=320)
    print("✅ OpenVINO export complete!")
except Exception as e:
    print(f"⚠️ OpenVINO export failed: {e}")
    print("Install with: pip install openvino-dev")
