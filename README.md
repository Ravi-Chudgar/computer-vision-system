# Real-Time Computer Vision System

A single-file Python application that combines gesture recognition, background manipulation, air drawing, lip detection, speech-to-text transcription, and LinkedIn publishing — all running at ~30 fps on a standard laptop webcam.

---

## Features

### 👻 Invisibility Effects
- **Pinch gesture** → ghost / partial transparency (background subtraction + alpha blending)
- **L-hand gesture** → full invisibility cloak (person pixels replaced with captured background)

### ✍️ Air Drawing with Shape Recognition
- Point one **index finger** in draw mode to trace shapes mid-air
- System auto-recognises: **circle, rectangle, square, triangle, line, polygon**
- Snaps messy finger traces to clean geometric shapes using OpenCV contour approximation

### 👆👆 Frame Square Gesture
- Hold both index fingers up to define two corners of a rectangle
- Hold for ~1 second → rectangle locks and saves to the frame

### 🖼️ Focus Box Mode
- Click 4 points to define a visible region
- Everything **outside** the box is replaced with the clean background — hides distractions during video calls

### 🎙️ Speech → Structured Notes
- Press `S` to start/stop microphone recording
- Audio captured with **sounddevice** (no PyAudio required)
- Transcribed via **Google Speech-to-Text** in 5-second chunks
- Each utterance classified by **Claude claude-sonnet-4-6** as `sentence`, `paragraph`, or `bullet`
- Transcription panel displayed live **next to the camera feed**
- Auto-saved to a **Word document** (`.docx`) after every entry

### 👄 Lip Detection
- MediaPipe FaceLandmarker draws mouth landmark ring
- Detects speaking vs silent state in real time

### 🔗 LinkedIn Publishing
- Press `P` to publish transcription to LinkedIn via OAuth 2.0 + UGC Posts API
- **Claude claude-sonnet-4-6** rewrites the raw notes into an engaging post with hashtags
- Token saved locally — authorize once, post forever

---

## Controls

| Key | Action |
|-----|--------|
| `S` | Start / Stop microphone (speech → text) |
| `D` | Toggle air-drawing mode |
| `E` | Snap current stroke to recognised shape |
| `L` | Toggle lip landmark overlay |
| `F` | Toggle focus mode |
| `R` | Define focus region (4 mouse clicks) |
| `W` | Save transcription as `.docx` Word document |
| `P` | Post transcription to LinkedIn |
| `B` | Re-capture background |
| `C` | Clear all shapes and transcription |
| `Q` | Quit |
| Pinch | Toggle partial invisibility |
| L-hand | Toggle full invisibility |
| Both index fingers (hold 1s) | Save frame rectangle |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| OpenCV | Video capture, display, drawing |
| MediaPipe Tasks API | Hand landmarks, face landmarks, gesture detection |
| NumPy | Frame blending, mask operations |
| sounddevice | Microphone recording (no PyAudio) |
| SpeechRecognition | Google STT via AudioFile |
| Anthropic Claude claude-sonnet-4-6 | Text classification, LinkedIn post formatting |
| python-docx | Word document generation |
| LinkedIn OAuth 2.0 | UGC Posts API for publishing |

---

## Installation

```bash
git clone https://github.com/Ravi-Chudgar/computer-vision-system.git
cd computer-vision-system
pip install -r requirements.txt
python3 invisibility_system.py
```

### Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
sounddevice>=0.4.0
SpeechRecognition>=3.10.0
python-docx>=1.0.0
anthropic>=0.20.0
```

### Optional: AI + LinkedIn features

Set these environment variables to enable Claude classification and LinkedIn posting:

```bash
export ANTHROPIC_API_KEY=your_api_key
export LINKEDIN_CLIENT_ID=your_client_id
export LINKEDIN_CLIENT_SECRET=your_client_secret
```

> **LinkedIn setup:** Create an app at [developer.linkedin.com](https://www.linkedin.com/developers/apps), request the *Share on LinkedIn* product, and add `http://localhost:8765/callback` as a redirect URL.

---

## How It Works

### Invisibility
A clean background frame is captured at startup. MediaPipe segments the foreground person using the Selfie Segmentation model. In the main loop, per-pixel alpha blending replaces foreground pixels with the captured background — creating a real-time invisibility effect.

### Gesture Detection
MediaPipe Hand Landmarker returns 21 3D landmarks per hand per frame. Geometric rules detect:
- **Pinch** — Euclidean distance between thumb tip and index tip < 6% of frame width
- **L-hand** — index tip above PIP and MCP joints; middle/ring/pinky tips below PIP; thumb extended horizontally
- **Pointing** — index up, all other fingers down

### Shape Recognition
A finger-traced path is stored as a list of pixel coordinates. On finalization, `cv2.convexHull` computes the outer boundary, `cv2.approxPolyDP` reduces it to vertices, and circularity `(4π·area) / perimeter²` classifies circular vs polygonal shapes.

### Speech Pipeline
`sounddevice.InputStream` captures raw `int16` audio. Every 5 seconds the buffer is flushed to a temp WAV file via stdlib `wave`, passed to `speech_recognition.AudioFile` (bypassing PyAudio entirely), and sent to Google STT. The result is sent to Claude claude-sonnet-4-6 for error correction and type classification, then appended to the live panel and the auto-save `.docx`.

---

## Screenshots

> Camera feed (left) with hand landmarks and HUD badges · Transcription panel (right) with live notes

---

## License

MIT
