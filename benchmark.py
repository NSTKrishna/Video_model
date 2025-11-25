"""
Performance benchmark script to compare different optimization levels
"""
import time
from ultralytics import YOLO
import cv2

def benchmark_inference():
    print("🔥 PERFORMANCE BENCHMARK 🔥\n")
    
    # Load a test frame
    cap = cv2.VideoCapture("videos/sample.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not load video")
        return
    
    # Test 1: PyTorch with default size (640)
    print("Test 1: PyTorch YOLOv8n @ 640px (Baseline)")
    model = YOLO("yolov8n.pt")
    start = time.time()
    for _ in range(10):
        _ = model(frame, verbose=False)
    avg_time = (time.time() - start) / 10 * 1000
    print(f"  Average: {avg_time:.1f}ms per frame\n")
    baseline = avg_time
    
    # Test 2: PyTorch with imgsz=320
    print("Test 2: PyTorch YOLOv8n @ 320px (Fix 1)")
    start = time.time()
    for _ in range(10):
        _ = model(frame, imgsz=320, verbose=False)
    avg_time = (time.time() - start) / 10 * 1000
    speedup = baseline / avg_time
    print(f"  Average: {avg_time:.1f}ms per frame")
    print(f"  Speedup: {speedup:.1f}x faster\n")
    
    # Test 3: ONNX with imgsz=320
    try:
        print("Test 3: ONNX YOLOv8n @ 320px (Fix 2)")
        model_onnx = YOLO("yolov8n.onnx")
        start = time.time()
        for _ in range(10):
            _ = model_onnx(frame, imgsz=320, verbose=False)
        avg_time = (time.time() - start) / 10 * 1000
        speedup = baseline / avg_time
        print(f"  Average: {avg_time:.1f}ms per frame")
        print(f"  Speedup: {speedup:.1f}x faster\n")
    except Exception as e:
        print(f"  ONNX test failed: {e}\n")
    
    # Test 4: OpenVINO with imgsz=320
    try:
        print("Test 4: OpenVINO YOLOv8n @ 320px (Fix 3) ⭐")
        model_ov = YOLO("yolov8n_openvino_model")
        start = time.time()
        for _ in range(10):
            _ = model_ov(frame, imgsz=320, verbose=False)
        avg_time = (time.time() - start) / 10 * 1000
        speedup = baseline / avg_time
        print(f"  Average: {avg_time:.1f}ms per frame")
        print(f"  Speedup: {speedup:.1f}x faster ⚡\n")
    except Exception as e:
        print(f"  OpenVINO test failed: {e}\n")
    
    print("=" * 50)
    print("Benchmark complete!")

if __name__ == "__main__":
    benchmark_inference()
