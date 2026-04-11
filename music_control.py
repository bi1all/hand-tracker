import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import keyboard
import time
import os
import urllib.request
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Download model if needed
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Detector setup
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# Skeleton connections
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Gesture cooldown
last_gesture_time = 0
COOLDOWN = 1.2  # seconds between gesture triggers

def finger_states(hand):
    # Returns list of booleans: [thumb, index, middle, ring, pinky] extended
    tips = [4, 8, 12, 16, 20]
    pip  = [3, 6, 10, 14, 18]
    extended = []
    # Thumb (compare x axis)
    extended.append(hand[4].x < hand[3].x)
    # Other fingers (compare y axis, lower y = higher on screen = extended)
    for i in range(1, 5):
        extended.append(hand[tips[i]].y < hand[pip[i]].y)
    return extended

def detect_gesture(hand):
    f = finger_states(hand)
    thumb, index, middle, ring, pinky = f

    # Open palm — all fingers extended
    if all(f):
        return "PLAY_PAUSE"

    # Peace — index + middle up, others down
    if not thumb and index and middle and not ring and not pinky:
        return "NEXT_TRACK"

    # Thumbs up — only thumb extended
    if thumb and not index and not middle and not ring and not pinky:
        return "PREV_TRACK"

    return None

def get_pinch_distance(hand, w, h):
    x1, y1 = int(hand[4].x * w), int(hand[4].y * h)
    x2, y2 = int(hand[8].x * w), int(hand[8].y * h)
    return np.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)

cap = cv2.VideoCapture(0)
print("Running. Show gestures to camera.")
print("Open palm=Play/Pause | Peace=Next | Thumbs Up=Prev | Pinch=Volume")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    status_text = ""

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        now = time.time()

        # --- Volume via pinch ---
        dist, pt1, pt2 = get_pinch_distance(hand, w, h)
        # Map pinch distance (20px=min, 200px=max) to volume
        vol_db = np.interp(dist, [20, 200], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol_db, None)
        vol_pct = int(np.interp(dist, [20, 200], [0, 100]))

        # Draw pinch line
        cv2.line(frame, pt1, pt2, (0, 255, 100), 2)
        cv2.circle(frame, pt1, 8, (0, 255, 100), -1)
        cv2.circle(frame, pt2, 8, (0, 255, 100), -1)
        status_text = f"Volume: {vol_pct}%"

        # --- Gesture controls ---
        if now - last_gesture_time > COOLDOWN:
            gesture = detect_gesture(hand)
            if gesture == "PLAY_PAUSE":
                keyboard.send("play/pause media")
                status_text = "PLAY / PAUSE"
                last_gesture_time = now
            elif gesture == "NEXT_TRACK":
                keyboard.send("next track")
                status_text = "NEXT TRACK"
                last_gesture_time = now
            elif gesture == "PREV_TRACK":
                keyboard.send("previous track")
                status_text = "PREV TRACK"
                last_gesture_time = now

        # Draw skeleton
        for a, b in CONNECTIONS:
            ax, ay = int(hand[a].x * w), int(hand[a].y * h)
            bx, by = int(hand[b].x * w), int(hand[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

    # HUD
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.putText(frame, "Q to quit", (w - 120, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Music Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()