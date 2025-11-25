# 🎥 Video Inventory Detection System

An optimized object detection and tracking system for counting inventory items in videos using YOLO models. Features advanced optimizations for 29x faster processing on Apple Silicon.

## 🚀 Features

- **Real-time Object Detection** - Detects and counts objects in video frames
- **Object Tracking** - Tracks unique objects to avoid duplicate counts
- **Multi-Model Support** - Supports grocery_yolov5s.pt, YOLOv8n, ONNX, and OpenVINO formats
- **High Performance** - 29x faster than baseline with optimizations
- **Automatic Fallback** - Intelligently falls back to available models

## 📊 Performance

| Configuration                   | Speed per Frame | Speedup           |
| ------------------------------- | --------------- | ----------------- |
| Original (YOLOv8x @ 640px)      | ~700ms          | 1x                |
| **Optimized (YOLOv8n @ 320px)** | **~24ms**       | **29x faster** ⚡ |

### Optimizations Applied:

- ✅ Reduced image size from 640px to 320px (3x speedup)
- ✅ Using YOLOv8n instead of YOLOv8x (smaller, faster model)
- ✅ ONNX Runtime support (3x-6x faster on some systems)
- ✅ OpenVINO support (5x-10x faster on Intel CPUs)
- ✅ Optimized for Apple Silicon (M1/M2/M3)

## 📁 Project Structure

```
Video_model/
├── src/
│   ├── detect_inventory.py    # Frame-based object detection
│   ├── track_inventory.py     # Video-based object tracking
│   ├── extract_frames.py      # Video frame extraction
│   ├── run.py                 # Main execution script
│   └── utils.py               # Utility functions
├── videos/
│   └── sample.mp4             # Input video files
├── output/
│   └── inventory.json         # Detection results
├── grocery_yolov5s.pt         # Grocery-specific model (optional)
├── yolov8n.pt                 # Default YOLO model
├── yolov8n.onnx              # ONNX export (optional)
├── yolov8n_openvino_model/   # OpenVINO export (optional)
├── requirements.txt           # Python dependencies
└── readme.md                  # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/NSTKrishna/Video_model.git
cd Video_model
```

2. **Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your model** (Optional - for grocery detection)
   - Place `grocery_yolov5s.pt` in the project root directory
   - If not available, the system will automatically use YOLOv8n

## 🎯 Usage

### Basic Usage

Run the inventory detection on your video:

```bash
python src/run.py
```

### Expected Output

```json
📦 Final Inventory:
{
    "frame_based_count": {
        "mouse": 1,
        "book": 2
    },
    "tracking_based_count": {
        "mouse": 1,
        "book": 4
    }
}
```

### Using Custom Video

Place your video file in the `videos/` directory and update `src/run.py`:

```python
VIDEO_PATH = "videos/your_video.mp4"
```

## 🔧 Configuration

### Model Selection

The system automatically selects the best available model:

1. **grocery_yolov5s.pt** (if present) - Grocery-specific detection
2. **yolov8n.pt** (fallback) - General object detection

### Adjusting Detection Settings

Edit `src/detect_inventory.py` to customize:

```python
# Change image size (smaller = faster, larger = more accurate)
results = model(frame, imgsz=320, verbose=False)

# Adjust frame extraction rate in extract_frames.py
frames = extract_frames(VIDEO_PATH, fps_interval=1)  # Extract 1 frame per second
```

## 🧪 Advanced Features

### Export Models for Better Performance

Run the export script to create optimized model formats:

```bash
python export_model.py
```

This creates:

- `yolov8n.onnx` - ONNX format (portable, faster)
- `yolov8n_openvino_model/` - OpenVINO format (fastest on Intel CPUs)

### Benchmark Performance

Test different model formats on your system:

```bash
python benchmark.py
```

## 📋 Requirements

Main dependencies:

- `ultralytics` - YOLO implementation
- `opencv-python` - Video processing
- `numpy` - Numerical operations
- `onnxruntime` - ONNX inference (optional)
- `openvino-dev` - OpenVINO optimization (optional)

See `requirements.txt` for complete list.

## 🎓 How It Works

### 1. Frame-Based Detection

- Extracts frames from video at specified intervals
- Runs YOLO detection on each frame
- Counts all detected objects across frames

### 2. Tracking-Based Detection

- Processes video stream continuously
- Assigns unique IDs to objects using ByteTrack
- Counts only unique objects (avoids duplicates)

### 3. Optimization Pipeline

```
Video Input → Frame Extraction → YOLO Detection (320px) → Object Counting
                                      ↓
                            Tracking (ByteTrack) → Unique ID Assignment
```

## 🐛 Troubleshooting

### Issue: "grocery_yolov5s.pt not found" warning

**Solution**: This is normal if you don't have the grocery model. The system will use YOLOv8n as fallback. To remove the warning, either:

- Add the `grocery_yolov5s.pt` file to the project root, OR
- The code will continue working with YOLOv8n

### Issue: Slow processing

**Solutions**:

1. Ensure `imgsz=320` is set in detection/tracking calls
2. Use YOLOv8n instead of larger models (YOLOv8x)
3. Extract fewer frames (increase `fps_interval`)
4. Run `python export_model.py` and use ONNX/OpenVINO

### Issue: CUDA/GPU errors

**Solution**: The code is optimized for CPU inference. GPU support is automatic if PyTorch detects CUDA.

## 📈 Performance Tips

1. **For Speed**: Use `imgsz=320`, YOLOv8n model, reduce frame extraction rate
2. **For Accuracy**: Use `imgsz=640`, grocery_yolov5s.pt or larger models, extract more frames
3. **For Balance**: Current default settings (320px, YOLOv8n, 1 fps)

## 📚 Documentation

Additional documentation files:

- `FINAL_RESULTS.md` - Complete optimization results
- `GROCERY_MODEL_SETUP.md` - Grocery model setup guide
- `OPTIMIZATIONS.md` - Detailed optimization explanations
- `QUICK_REFERENCE.md` - Quick command reference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Author

**NSTKrishna**

- GitHub: [@NSTKrishna](https://github.com/NSTKrishna)

## 🙏 Acknowledgments

- **Ultralytics** for YOLO implementation
- **OpenVINO** for CPU optimization
- **ONNX Runtime** for cross-platform inference

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review documentation in the project
3. Open an issue on GitHub

---

**Last Updated**: November 25, 2025

**Version**: 2.0 (Optimized)

⭐ If you find this project useful, please give it a star!
