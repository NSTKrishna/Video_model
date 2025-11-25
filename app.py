from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

app = FastAPI()

# Load model once
model = YOLO("best.pt")

# Class mapping (optional)
model.model.names = {
    0: "Chips",
    1: "Ice Cream",
    2: "Noodles",
    3: "Cold Drinks"
}
names = model.model.names


# -------------------------------
# Home route (fixes Not Found)
# -------------------------------
@app.get("/")
def home():
    return {"message": "YOLO Video API is running!"}


# -------------------------------
# Extract frames from video
# -------------------------------
def extract_frames_from_video(video_bytes):
    # Write uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# -------------------------------
# YOLO Counting Logic
# -------------------------------
def detect_inventory_frames(frames):
    counts = {}

    for frame in frames:
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                name = names[cls_id]
                counts[name] = counts.get(name, 0) + 1

    return counts


# -------------------------------
# Detect endpoint
# -------------------------------
@app.post("/detect")
async def detect_inventory(file: UploadFile = File(...)):
    data = await file.read()

    # First try to decode as IMAGE
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # If image decoding fails → treat as VIDEO
    if frame is None:
        frames = extract_frames_from_video(data)
        if len(frames) == 0:
            return {"error": "Invalid image/video"}
        counts = detect_inventory_frames(frames)
        return JSONResponse(counts)

    # IMAGE path
    results = model(frame)
    counts = {}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            name = names[cls_id]
            counts[name] = counts.get(name, 0) + 1

    return JSONResponse(counts)
