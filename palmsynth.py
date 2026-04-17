"""
PalmSynth - Windows Port
Hand-controlled audio player using MediaPipe + OpenCV + pygame

Controls:
  Left hand pinch  → Volume (0 to 1)
  Right hand pinch → Bass boost (-12 to +12 dB simulated via EQ)
  Press O          → Open MP3 file
  Press SPACE      → Play / Pause
  Press Q or ESC   → Quit
"""

import cv2
import numpy as np
import pygame
import pygame.mixer
import threading
import tkinter as tk
from tkinter import filedialog
import math
import time
import os
import sys
import queue
from pydub import AudioSegment
from pydub.effects import low_pass_filter
import io
import urllib.request

# ── MediaPipe 0.10.x compatible import ──────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    USE_TASKS_API = True
except Exception:
    USE_TASKS_API = False

# Fallback: try legacy solutions API
if not USE_TASKS_API:
    try:
        import mediapipe as mp
        _hands_legacy = mp.solutions.hands
        USE_TASKS_API = False
    except Exception as e:
        print(f"[ERROR] MediaPipe import failed: {e}")
        sys.exit(1)

# ─────────────────────────────────────────────
# AUDIO ENGINE
# ─────────────────────────────────────────────

class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.channel = None
        self.sound = None
        self.original_seg = None   # pydub AudioSegment (unmodified)
        self.loaded = False
        self.playing = False
        self.volume = 0.5          # 0.0 – 1.0
        self.bass_gain = 0.0       # -12.0 – +12.0 dB
        self.filename = "None"
        self._bass_dirty = False
        self._lock = threading.Lock()
        self._bass_thread = None
        self._pending_bass = None
        self._bass_queue = queue.Queue(maxsize=1)
        self._start_bass_worker()

    # ── file loading ──────────────────────────
    def load(self, path):
        was_playing = self.playing
        self.stop()
        try:
            self.original_seg = AudioSegment.from_file(path)
            self.filename = os.path.basename(path)
            self._apply_bass_sync(self.bass_gain)
            self.loaded = True
            if was_playing:
                self.play()
        except Exception as e:
            print(f"[Audio] Load error: {e}")
            self.loaded = False
            self.filename = "None"

    def play(self):
        if not self.loaded or self.sound is None:
            return
        if self.channel and self.channel.get_busy():
            self.channel.stop()
        self.channel = pygame.mixer.find_channel(True)
        self.channel.set_volume(self.volume)
        self.channel.play(self.sound, loops=-1)
        self.playing = True

    def pause(self):
        if self.channel and self.channel.get_busy():
            self.channel.pause()
            self.playing = False
        elif self.channel:
            self.channel.unpause()
            self.playing = True

    def stop(self):
        if self.channel:
            self.channel.stop()
        self.playing = False

    # ── volume ───────────────────────────────
    def set_volume(self, v):
        self.volume = max(0.0, min(1.0, v))
        if self.channel:
            self.channel.set_volume(self.volume)

    # ── bass (async so hand movement stays smooth) ──
    def set_bass(self, gain_db):
        gain_db = max(-12.0, min(12.0, float(gain_db)))
        if abs(gain_db - self.bass_gain) < 0.4:   # dead-band
            return
        self.bass_gain = gain_db
        try:
            self._bass_queue.put_nowait(gain_db)
        except queue.Full:
            pass  # drop; worker is busy, it'll pick up next change

    def _start_bass_worker(self):
        t = threading.Thread(target=self._bass_worker, daemon=True)
        t.start()

    def _bass_worker(self):
        while True:
            gain_db = self._bass_queue.get()          # blocks until work arrives
            if self.original_seg is None:
                continue
            self._apply_bass_sync(gain_db)

    def _apply_bass_sync(self, gain_db):
        """Rebuild the pygame Sound with bass EQ applied."""
        if self.original_seg is None:
            return
        try:
            seg = self.original_seg
            # Simple shelving: boost lows with low-pass then blend
            if gain_db > 0:
                lows = low_pass_filter(seg, 200)
                boosted = lows + gain_db          # pydub dB addition
                seg = seg.overlay(boosted)
            elif gain_db < 0:
                lows = low_pass_filter(seg, 200)
                cut = lows + gain_db              # negative = cut
                seg = seg.overlay(cut)

            # Convert to pygame Sound via in-memory WAV
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            buf.seek(0)
            with self._lock:
                was_playing = self.playing
                pos = 0
                if self.channel and self.channel.get_busy():
                    self.channel.stop()
                self.sound = pygame.mixer.Sound(buf)
                if was_playing:
                    self.channel = pygame.mixer.find_channel(True)
                    self.channel.set_volume(self.volume)
                    self.channel.play(self.sound, loops=-1)
        except Exception as e:
            print(f"[Audio] Bass apply error: {e}")


# ─────────────────────────────────────────────
# HAND TRACKER
# ─────────────────────────────────────────────

mp_hands = mp.solutions.hands

# MediaPipe landmark indices
WRIST       = 0
THUMB_CMC   = 1; THUMB_MCP  = 2; THUMB_IP   = 3; THUMB_TIP  = 4
INDEX_MCP   = 5; INDEX_PIP  = 6; INDEX_DIP  = 7; INDEX_TIP  = 8
MIDDLE_MCP  = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP  = 14; RING_DIP   = 15; RING_TIP  = 16
PINKY_MCP   = 17; PINKY_PIP = 18; PINKY_DIP  = 19; PINKY_TIP = 20

# Bone connections for skeleton rendering — matches HandOverlay.swift
BONES = [
    # Palm
    (WRIST, INDEX_MCP), (WRIST, MIDDLE_MCP), (WRIST, RING_MCP), (WRIST, PINKY_MCP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
    # Thumb
    (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # Index
    (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # Middle
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # Ring
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # Pinky
    (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

class Smoother:
    """Exponential moving average — mirrors Swift Smoother class."""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def step(self, x):
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.value * (1 - self.alpha) + x * self.alpha
        return self.value


def pinch_distance(lm, w, h):
    """Normalized pinch distance between thumb tip and index tip."""
    tx, ty = lm[THUMB_TIP].x * w, lm[THUMB_TIP].y * h
    ix, iy = lm[INDEX_TIP].x * w, lm[INDEX_TIP].y * h
    dist = math.hypot(tx - ix, ty - iy)
    # Normalize by hand span (wrist to middle MCP)
    wx_, wy_ = lm[WRIST].x * w, lm[WRIST].y * h
    mx_, my_ = lm[MIDDLE_MCP].x * w, lm[MIDDLE_MCP].y * h
    span = math.hypot(wx_ - mx_, wy_ - my_) + 1e-6
    return min(dist / (span * 1.5), 1.0)


def finger_curl(lm, mcp_i, pip_i, tip_i):
    """Curl value 0 (open) to 1 (curled) — mirrors Swift computeCurl."""
    mcp = lm[mcp_i]; pip = lm[pip_i]; tip = lm[tip_i]
    ab = math.hypot(mcp.x - pip.x, mcp.y - pip.y)
    bc = math.hypot(pip.x - tip.x, pip.y - tip.y)
    if ab < 1e-4:
        return 0.0
    ratio = bc / ab
    curled = 1.0 - max(0.0, min(1.0, (ratio - 0.2) / 1.2))
    return max(0.0, min(1.0, curled))


def draw_hand(frame, lm, h, w, color=(255, 255, 255)):
    """Draw skeletal overlay matching HandOverlay.swift style."""
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]

    # Glow layer — thick soft lines
    glow = frame.copy()
    for a, b in BONES:
        cv2.line(glow, pts[a], pts[b], color, 12)
    cv2.addWeighted(glow, 0.25, frame, 0.75, 0, frame)

    # Core bones — bright thin lines on top
    for a, b in BONES:
        cv2.line(frame, pts[a], pts[b], color, 2)

    # Joints
    for pt in pts:
        # Outer glow dot
        cv2.circle(frame, pt, 8, color, 1)
        # Filled center dot
        cv2.circle(frame, pt, 4, color, -1)


# ─────────────────────────────────────────────
# HUD RENDERING
# ─────────────────────────────────────────────

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def draw_hud(frame, audio, tracker_state):
    h, w = frame.shape[:2]

    # Stats panel (top-left) — matches PerformanceView
    lines = [
        f"frames: {tracker_state['frames']}",
        f"hands: {tracker_state['hands']}",
        f"trackingOK: {str(tracker_state['tracking_ok']).lower()}",
        f"volume: {audio.volume:.2f}",
        f"bass: {audio.bass_gain:+.1f} dB",
    ]
    pad = 10
    line_h = 22
    panel_h = pad * 2 + line_h * len(lines)
    panel_w = 210

    # Dark panel background
    panel = frame[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(panel, 0.25, frame[0:panel_h, 0:panel_w], 0.75, 0, frame[0:panel_h, 0:panel_w])

    for i, line in enumerate(lines):
        y = pad + (i + 1) * line_h - 4
        cv2.putText(frame, line, (pad, y), FONT_BOLD, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Top bar — filename + controls hint
    bar_text = tracker_state['filename']
    cv2.putText(frame, bar_text, (panel_w + 15, 28), FONT, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Controls hint (bottom)
    hints = "[O] Open   [SPACE] Play/Pause   [Q] Quit"
    cv2.putText(frame, hints, (10, h - 12), FONT, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    # Volume bar (right side)
    bar_x = w - 28
    bar_top = 40
    bar_bot = h - 40
    bar_len = bar_bot - bar_top
    fill = int(bar_len * audio.volume)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 12, bar_bot), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_bot - fill), (bar_x + 12, bar_bot), (220, 220, 220), -1)
    cv2.putText(frame, "VOL", (bar_x - 2, bar_top - 8), FONT, 0.35, (160, 160, 160), 1)

    # Bass bar (right side, next to volume)
    bar2_x = bar_x - 24
    mid_y = bar_top + bar_len // 2
    bass_norm = (audio.bass_gain + 12) / 24.0   # 0..1
    fill2 = int(bar_len * bass_norm)
    cv2.rectangle(frame, (bar2_x, bar_top), (bar2_x + 12, bar_bot), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar2_x, bar_bot - fill2), (bar2_x + 12, bar_bot), (180, 200, 255), -1)
    cv2.line(frame, (bar2_x - 2, mid_y), (bar2_x + 14, mid_y), (100, 100, 100), 1)
    cv2.putText(frame, "BASS", (bar2_x - 6, bar_top - 8), FONT, 0.35, (160, 160, 160), 1)

    # Play/pause indicator
    status_color = (100, 255, 100) if audio.playing else (255, 100, 100)
    status_text = "PLAYING" if audio.playing else "PAUSED"
    if not audio.loaded:
        status_text = "NO FILE"
        status_color = (120, 120, 120)
    cv2.putText(frame, status_text, (w // 2 - 35, h - 12), FONT, 0.48, status_color, 1, cv2.LINE_AA)

    # Left/Right hand labels
    if tracker_state.get('left_pinch') is not None:
        cv2.putText(frame, f"L-PINCH: {tracker_state['left_pinch']:.2f}",
                    (10, h - 35), FONT, 0.42, (200, 220, 255), 1, cv2.LINE_AA)
    if tracker_state.get('right_pinch') is not None:
        cv2.putText(frame, f"R-PINCH: {tracker_state['right_pinch']:.2f}",
                    (10, h - 55), FONT, 0.42, (255, 220, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# FILE PICKER (runs in main thread via Tk)
# ─────────────────────────────────────────────

def pick_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    audio = AudioEngine()

    vol_smoother  = Smoother(alpha=0.12)
    bass_smoother = Smoother(alpha=0.10)

    frame_count   = 0
    tracking_ok   = False
    left_pinch_v  = None
    right_pinch_v = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    win_name = "PalmSynth"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    print("PalmSynth running.")
    print("  [O]     — Open audio file")
    print("  [SPACE] — Play / Pause")
    print("  [Q/ESC] — Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mirror flip for natural feel
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Run hand detection on real camera frame BEFORE blanking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands_model.process(rgb)
        rgb.flags.writeable = True

        # Pure black background — no camera feed visible
        frame = np.zeros_like(frame)

        num_hands = 0
        left_pinch_v = None
        right_pinch_v = None

        if results.multi_hand_landmarks and results.multi_handedness:
            tracking_ok = True
            num_hands = len(results.multi_hand_landmarks)

            for lm_data, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lm = lm_data.landmark
                label = handedness.classification[0].label  # "Left" or "Right"

                # Draw skeleton
                draw_hand(frame, lm, h, w)

                # Compute pinch distance
                pd = pinch_distance(lm, w, h)

                if label == "Left":
                    # Mirrored cam: MediaPipe "Left" = user's RIGHT
                    # We want: left hand = volume, right hand = bass
                    # After flip, labels swap — so MediaPipe "Left" = user's left
                    smooth_pd = vol_smoother.step(pd)
                    left_pinch_v = smooth_pd
                    # Map pinch: 0 (pinched) = max, 1 (open) = min
                    volume = 1.0 - smooth_pd
                    audio.set_volume(volume)
                else:
                    smooth_pd = bass_smoother.step(pd)
                    right_pinch_v = smooth_pd
                    # Map: 0 (pinched) = max bass (+12), 1 (open) = 0
                    bass_db = (1.0 - smooth_pd) * 12.0 - 0.0
                    # Center at open hand = 0 dB
                    bass_db = (0.5 - smooth_pd) * 24.0
                    audio.set_bass(bass_db)
        else:
            tracking_ok = False

        # ── HUD ────────────────────────────────
        tracker_state = {
            'frames':      frame_count,
            'hands':       num_hands,
            'tracking_ok': tracking_ok,
            'filename':    audio.filename,
            'left_pinch':  left_pinch_v,
            'right_pinch': right_pinch_v,
        }
        draw_hud(frame, audio, tracker_state)

        cv2.imshow(win_name, frame)

        # ── Key handling ───────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # Q or ESC
            break

        elif key == ord('o') or key == ord('O'):
            # Pause render briefly while file picker opens
            path = pick_file()
            if path:
                audio.load(path)
                audio.play()

        elif key == ord(' '):
            if audio.loaded:
                audio.pause()

    # ── Cleanup ─────────────────────────────
    cap.release()
    hands_model.close()
    cv2.destroyAllWindows()
    audio.stop()
    pygame.mixer.quit()
    print("PalmSynth closed.")


if __name__ == "__main__":
    main()
