from ultralytics import YOLO

def track_inventory(video_path):
    print("🔥 Using Normal YOLOv8n for tracking @ 320px")

    # Load normal YOLOv8n (not HF)
    model = YOLO("best.pt")

    print(model.names)

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        save=False,
        show=False,
        stream=True,
        imgsz=320,
        verbose=False
    )

    ids_seen = set()
    classes_seen = {}

    for frame in results:
        if not frame.boxes:
            continue

        for det in frame.boxes:
            if det.id is None:
                continue

            obj_id = int(det.id.item())
            cls = frame.names[int(det.cls)]

            if obj_id not in ids_seen:
                ids_seen.add(obj_id)
                classes_seen[cls] = classes_seen.get(cls, 0) + 1

    return classes_seen
