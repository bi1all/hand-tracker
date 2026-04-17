import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pythonosc import udp_client
import urllib.request
import os
import sys

# ---- Model download ----
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)

# ---- OSC client ----
client = udp_client.SimpleUDPClient("127.0.0.1", 10000)

# ---- Skeleton connections ----
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ---- Detector setup ----
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

detector = mp_vision.HandLandmarker.create_from_options(options)

# ---- Camera setup ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    # Always send both hands — send -1 if hand not detected
    hands = result.hand_landmarks if result.hand_landmarks else []

    for hand_idx in range(2):
        if hand_idx < len(hands):
            hand = hands[hand_idx]
            for lm_id in range(21):
                lm = hand[lm_id]
                client.send_message(f"/h{hand_idx}lm{lm_id}x", float(lm.x))
                client.send_message(f"/h{hand_idx}lm{lm_id}y", float(1.0 - lm.y))
        else:
            for lm_id in range(21):
                client.send_message(f"/h{hand_idx}lm{lm_id}x", -1.0)
                client.send_message(f"/h{hand_idx}lm{lm_id}y", -1.0)

    # Draw on preview
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand in result.hand_landmarks:
            for a, b in CONNECTIONS:
                ax, ay = int(hand[a].x * w), int(hand[a].y * h)
                bx, by = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
