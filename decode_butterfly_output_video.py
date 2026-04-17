import cv2

# Open the extracted video
cap = cv2.VideoCapture("output_video.mp4")

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR (OpenCV default) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

cap.release()
print(f"Decoded {len(frames)} frames.")