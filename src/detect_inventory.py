from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# Override mapping
model.model.names = {
    0: "Chips",
    1: "Ice Cream",
    2: "Noodles",
    3: "Cold Drinks"
}

names = model.model.names


def extract_frames_from_video(video_bytes):
    import tempfile

    # Write video bytes to a real temp file with correct extension
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



def detect_inventory_in_frames(frames):
    """Your existing logic"""
    counts = {}

    for frame in frames:
        results = model(frame)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]

                counts[class_name] = counts.get(class_name, 0) + 1

    return counts


@app.post("/detect")
async def detect_inventory(file: UploadFile = File(...)):
    video_bytes = await file.read()

    frames = extract_frames_from_video(video_bytes)

    if len(frames) == 0:
        return {"error": "Could not extract frames"}

    counts = detect_inventory_in_frames(frames)

    return JSONResponse(counts)
