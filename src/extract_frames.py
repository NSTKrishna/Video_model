import cv2

def extract_frames(video_path, fps_interval=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % (video_fps * fps_interval) == 0:
            frames.append(frame)

        i += 1

    cap.release()
    return frames
