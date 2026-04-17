# hand-tracker
Built a hand tracking thing that reads your fingers through a webcam and controls your music.
Palm Synth is a high-fidelity spatial audio controller that transforms the human hand into a dynamic MIDI-equivalent interface. By leveraging Google’s MediaPipe landmarking and OpenCV, the system strips away the physical environment to render a clean, holographic skeletal overlay against a pure black void, ensuring total user privacy while maintaining a "vanish but be noticed" aesthetic. This module enables tactile manipulation of sound through precise gestural mapping: the left hand functions as a master volume fader via pinch-distance calculation, while the right hand operates as a real-time bass EQ and filter sweep. Engineered for Windows with a Python-based audio engine powered by Pygame and Pydub, Palm Synth merges computer vision with live signal processing to create a seamless, air-gap performance tool where every micro-movement of the fingers translates into immediate acoustic sculpture.
(How to Run It)
Paste this into your README.md file. It explains the requirements and the controls to anyone looking at your repo.

Markdown
### Setup & Installation
1. **Python 3.11** is required.
2. **FFmpeg**: Must be installed and added to your System PATH to process MP3 files.
3. **Dependencies**:
   ```bash
   pip install mediapipe==0.10.9 opencv-python numpy pygame pydub
Operational Controls
O Key: Open file explorer to load an MP3.

Space: Play / Pause toggle.

Left Hand Pinch: Dynamic Volume control.

Right Hand Pinch: Bass EQ / Filter control.

Q Key: Safe exit and release webcam.


---

### 2. The Commands (How to Upload)
Run these in your command prompt while inside your project folder to push the new file to your GitHub:

```bash
git add palmsynth.py
git commit -m "Add palm synth module and privacy-mode rendering"
git push origin main
3. Verification
Before you close the terminal, run the script one last time to ensure the environment is stable:
python palmsynth.py

If the window opens and the background is pure black while tracking your hands, the build is complete. No further configuration is required.
