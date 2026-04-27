import cv2
import numpy as np
from collections import deque

VIDEO_PATH = "data/phillips_no_color.mp4"

# we're keeping track of the components in the image over time based on centroid movement

# ignore tiny specks/noise, connected components smaller than this pixel area are discarded
MIN_AREA = 100

# width/height shape gate for components. rejecting components that are too wide vs. their height
MAX_W_OVER_H = 2.0

# maximum centroid jump (in pixels) allowed to match a detection to an existing track
MAX_MATCH_DISTANCE = 60

# how many consecutive frames a track can be unmatched before we delete it
MAX_MISSES = 20

# minimum number of matched frames before drawing a track
MIN_TRACK_AGE_TO_DRAW = 5

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("components")
cv2.createTrackbar("thresh", "components", 28, 255, lambda _: None)

# temporal smoothing state.
recent_frames = deque(maxlen=11)
ema = None

# each track is our best guess for "the same object over time".
tracks = {}
next_track_id = 1

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # smooth in time so thresholding is less flickery.
    recent_frames.append(gray)
    mean_frame = np.mean(recent_frames, axis=0)
    ema = gray if ema is None else (0.93 * ema + 0.07 * gray)
    smooth = (0.35 * mean_frame + 0.65 * ema).astype(np.uint8)

    threshold = cv2.getTrackbarPos("thresh", "components")
    _, mask = cv2.threshold(smooth, threshold, 255, cv2.THRESH_BINARY_INV)

    # turn the mask into connected components and keep likely vessel candidates.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    for label in range(1, n_labels):  # 0 is background
        x, y, w, h, area = stats[label]
        if area < MIN_AREA or w > MAX_W_OVER_H * h:
            continue

        component_mask = ((labels[y:y + h, x:x + w] == label).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours and len(contours[0]) >= 5:
            contour = contours[0] + [x, y]
            center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
            detections.append((cv2.fitEllipse(contour), center))

    # age tracks every frame; unmatched ones eventually expire.
    for track_id in list(tracks):
        tracks[track_id]["misses"] += 1
        if tracks[track_id]["misses"] > MAX_MISSES:
            del tracks[track_id]

    # match detections to nearest existing tracks.
    for ellipse, center in detections:
        best_track_id, best_distance = None, 1e9
        for track_id, track in tracks.items():
            distance = np.linalg.norm(center - track["center"])
            if distance < best_distance:
                best_distance, best_track_id = distance, track_id

        if best_track_id is not None and best_distance < MAX_MATCH_DISTANCE:
            track = tracks[best_track_id]
            track["center"] = center
            track["ellipse"] = ellipse
            track["misses"] = 0
            track["age"] += 1
        else:
            tracks[next_track_id] = {"center": center, "ellipse": ellipse, "misses": 0, "age": 1}
            next_track_id += 1

    # only draw stable tracks so one-frame noise does not show up.
    output = frame.copy()
    for track in tracks.values():
        if track["age"] >= MIN_TRACK_AGE_TO_DRAW:
            cv2.ellipse(output, track["ellipse"], (0, 255, 0), 2)

    cv2.imshow("components", output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()