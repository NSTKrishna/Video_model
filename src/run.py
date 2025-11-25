import json
import os
from extract_frames import extract_frames
from detect_inventory import detect_inventory_in_frames

VIDEO_PATH = "videos/sample2.mp4"
OUTPUT_JSON = "output/inventory.json"

if __name__ == "__main__":
    print("Extracting frames...")
    frames = extract_frames(VIDEO_PATH)

    if len(frames) == 0:
        print("❌ No frames extracted. Check your video path.")
        exit()

    print(f"✓ Extracted {len(frames)} frames")

    print("\nRunning object detection...")
    counts = detect_inventory_in_frames(frames)

    print("\n📦 Final Detected Objects:")
    print(json.dumps(counts, indent=4))

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(counts, f, indent=4)

    print(f"\n💾 Saved to: {OUTPUT_JSON}")
