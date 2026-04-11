import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client
import urllib.request
import os

# Download hand landmark model if not present
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Downloaded.")

# OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# Skeleton connections
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Setup detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_idx, hand in enumerate(result.hand_landmarks):
            h, w, _ = frame.shape

            # Send OSC data to TouchDesigner
            for lm_id, lm in enumerate(hand):
                base = f"/hand/{hand_idx}/lm/{lm_id}"
                client.send_message(f"{base}/x", lm.x)
                client.send_message(f"{base}/y", lm.y)
                client.send_message(f"{base}/z", lm.z)

            client.send_message(f"/hand/{hand_idx}/wrist/x", hand[0].x)
            client.send_message(f"/hand/{hand_idx}/wrist/y", hand[0].y)

            # Draw skeleton
            for a, b in CONNECTIONS:
                ax, ay = int(hand[a].x * w), int(hand[a].y * h)
                bx, by = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)

            # Draw joint dots
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()