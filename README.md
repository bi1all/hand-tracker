PalmSynth
PalmSynth is a gesture-based audio controller inspired by the BMW iDrive AirGesture system. It translates real-time hand geometry into tactile sound manipulation, allowing for a "touchless" mixing experience. While currently a functional demo, it establishes a high-fidelity bridge between computer vision and digital signal processing.

<img width="1920" height="1080" alt="Screenshot 2026-04-17 131205" src="https://github.com/user-attachments/assets/374308c5-7e0b-460f-a82b-1dc2ba0c393d" />


Core Functionality
Dual-Hand Vision Integration: Utilizes the MediaPipe framework to track 21 hand landmarks at sub-millisecond latency.

Dynamic Parameter Mapping:

Left Hand: Euclidean distance between index and thumb mapped to Master Gain.

Right Hand: Pinch distance mapped to a Low-Shelf EQ (Bass) for real-time frequency sculpting.

Privacy-First Aesthetic: Renders a high-contrast skeletal overlay on a pure black void, completely isolating the user's environment and face from the display.

Setup & Installation
To run this on Windows, you must have Python 3.11 installed. Follow these steps exactly:

1. Install FFmpeg (Required for MP3 processing)

Download the "essentials" build from gyan.dev.

Extract the folder and copy the path to the bin folder.

Add this path to your System Environment Variables.

2. Install Dependencies
Open your terminal and run:

Bash
pip install mediapipe==0.10.9 opencv-python numpy pygame pydub
3. Clone and Upload to GitHub


Load Music: Press 'O' on your keyboard to open the file picker and select an MP3.

Control Audio:

Spacebar: Play/Pause.

Left Hand Pinch: Close your thumb and index finger to lower volume; open them to increase it.

Right Hand Pinch: Close your thumb and index finger to pull back the Bass; open them to boost it.

Release: When you let go of the pinch, the setting stays locked at that level.

Exit: Press 'Q' to close the app.

Technical Architecture
The system runs on a Python-based signal chain:

Input: OpenCV captures the raw feed, processed by MediaPipe for landmark detection.

Logic: Coordinate normalization and hysteresis loops prevent jitter, ensuring smooth transitions.

Audio Engine: Powered by pydub and pygame.mixer, utilizing FFmpeg for high-fidelity decoding.

Known Limitations
Hand Swapping: Rapid crossing of hands may briefly invert control assignments.

Lux Dependency: Tracking stability decreases in low-light environments.

Occlusion: Extreme wrist angles may lead to landmark dropouts.

Author: Bilal
[Inspiration: Momin (Logic & Concept ported to Windows)](https://github.com/mominwaleed9089/PalmSynth#)
Concept: Original macOS logic by Momin; rebuilt for Windows architecture.
