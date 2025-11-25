from ultralytics import YOLO

model = YOLO("best.pt")

# override mapping
model.model.names = {
    0: "Chips",
    1: "Ice Cream",
    2: "Noodles",
    3: "Cold Drinks"
}

names = model.model.names

def detect_inventory_in_frames(frames):
    counts = {}

    for frame in frames:
        results = model(frame)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])   
                class_name = names[class_id]

                counts[class_name] = counts.get(class_name, 0) + 1

    return counts
