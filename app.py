from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# load model one time
model = YOLO("best.pt")

@app.post("/detect")
async def detect_inventory(file: UploadFile = File(...)):
    # read uploaded video/image
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image/video"}

    results = model(frame)
    names = model.model.names

    counts = {}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            name = names[cls_id]
            counts[name] = counts.get(name, 0) + 1

    return JSONResponse(counts)
