# 🎯 AI Face & Gesture Attendance System

A touchless, kiosk-ready Attendance System powered by **Computer Vision**, **InsightFace (`buffalo_l`)**, and **Google MediaPipe**. 

Employees authenticate via high-accuracy facial recognition (512-D embeddings) and trigger actions using natural hand gestures. An animated hold-to-confirm progress bar prevents accidental triggers, and audio-visual cues provide instant punch confirmation.

---

## ⚡ Features

- **Touchless & Contactless**: Check in and check out completely hands-free using camera gestures.
- **Biometric Face Recognition**: Built on InsightFace's `buffalo_l` deep face representation model with cosine similarity matching and L2 unit-vector normalization.
- **Gesture Control**:
  - ✌️ **Peace Sign (2 fingers)**: `CHECK_IN`
  - ✋ **Open Palm (5 fingers)**: `CHECK_OUT`
- **Hold-to-Confirm Mechanism**: Requires maintaining the gesture for ~0.8 seconds (20 consecutive frames) to eliminate false positives and accidental triggers.
- **Smart Anti-Spam / Cooldown**: 30-second cooldown per employee per action to prevent double punches.
- **Interactive Heads-Up Display (HUD)**:
  - Real-time bounding boxes with identity labels and match confidence.
  - Animated circular / linear progress bar during confirmation.
  - Success / error banner toasts and chime feedback.
  - Live footer showing recent punches and active enrolled user count.
- **Guided Multi-Angle Enrollment**: Step-by-step CLI tool that guides employees through frontal, left, right, and vertical head poses to produce robust averaged embeddings.
- **Automated CSV Logging**: Records timestamps, dates, employee names, actions, and similarity scores to `attendance.csv`.

---

## 📥 Model Download Instructions (Important)

> **Note:** Pretrained neural network models are **not included in this repository** to keep the Git history lightweight and respect GitHub repository size limits. You must download the required models before running the application.

You have two ways to get the models:

### Option A: Automatic Download (Recommended)

Run the included download helper script:

```bash
python download_models.py
```

This script will automatically fetch the required MediaPipe model bundles into the project root.

---

### Option B: Manual Download

#### 1. MediaPipe Gesture Recognizer Model (`gesture_recognizer.task`)
- **Required for**: `gesture_detector.py` and `attendance_system.py`
- **Direct Link**: [Download gesture_recognizer.task](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task)
- **Destination**: Place the downloaded file directly inside the root folder of this project (`Face_Attendance/gesture_recognizer.task`).

**Terminal Command:**
- **Windows (PowerShell):**
  ```powershell
  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task" -OutFile "gesture_recognizer.task"
  ```
- **macOS / Linux (curl):**
  ```bash
  curl -Lo gesture_recognizer.task https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task
  ```

---

#### 2. InsightFace Face Recognition Model (`buffalo_l`)
- **Required for**: Face detection, facial landmarks, and 512-dimensional embedding extraction.
- **Automatic behavior**: InsightFace will automatically download the `buffalo_l` model pack (~288 MB) on your first run of `enroll_employee.py` or `attendance_system.py`. It is saved into `~/.insightface/models/buffalo_l/`.
- **Manual download** (if the auto-download times out or fails):
  1. Download the archive: [buffalo_l.zip (GitHub Release)](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)
  2. Extract the zip into your local user cache directory:
     - **Windows**: `C:\Users\<Your-Username>\.insightface\models\buffalo_l\`
     - **Linux / macOS**: `~/.insightface/models/buffalo_l/`
  3. Ensure the folder contains the following ONNX files:
     - `det_10g.onnx` (Face detector)
     - `w600k_r50.onnx` (Feature recognition embedding)
     - `1k3d68.onnx`
     - `2d106det.onnx`
     - `genderage.onnx`

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/abdurraafay90/Face_Attendance.git
cd Face_Attendance
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
```bash
python download_models.py
```

---

## 👤 Enrolling an Employee

Before launching the attendance kiosk, enroll at least one face:

```bash
python enroll_employee.py --name "Your Name"
```

1. Look directly into the webcam with good lighting.
2. Press **`S`** to start capturing.
3. Follow the on-screen prompts:
   - *Look directly at camera*
   - *Turn slightly left*
   - *Turn slightly right*
   - *Tilt slightly up / down*
4. Once 20 multi-angle frames are captured, the tool computes a normalized embedding and saves it as `<your_name>_embedding.npy`.

---

## 🖥️ Running the Attendance System

Start the live kiosk:

```bash
python attendance_system.py
```

### How to Use the Kiosk:
1. **Stand in front of the camera**: The system detects your face and displays your name with a green bounding box once recognized.
2. **Perform the gesture**:
   - Show a **Peace Sign ✌️** to **Check In**.
   - Show an **Open Palm ✋** to **Check Out**.
3. **Hold for ~0.8s**: Watch the animated hold-to-confirm progress bar fill up.
4. **Done!**: A success banner appears, an audio chime plays, and your punch is saved to `attendance.csv`.

### Keyboard Shortcuts:
- **`Q`**: Quit the application.
- **`R`**: Hot-reload employee embeddings from disk without restarting.

---

## 📊 Attendance Logs

Attendance records are automatically logged to `attendance.csv` with the following schema:

| Timestamp | Date | Time | Employee_Name | Action | Confidence |
|---|---|---|---|---|---|
| `2026-10-16T09:00:15.123456` | `2026-10-16` | `09:00:15 AM` | `John Doe` | `CHECK_IN` | `0.78` |
| `2026-10-16T17:30:22.654321` | `2026-10-16` | `05:30:22 PM` | `John Doe` | `CHECK_OUT` | `0.81` |

---

## 📂 Repository Structure

```
Face_Attendance/
├── attendance_system.py     # Main real-time attendance kiosk application
├── attendance_logger.py     # CSV logging, audio chimes, and cooldown state
├── gesture_detector.py      # MediaPipe gesture detection & rule-based engine
├── enroll_employee.py       # Multi-angle guided employee enrollment tool
├── enroll_multiframe.py     # Fast multiframe enrollment utility
├── download_models.py       # Helper script to download pretrained model tasks
├── requirements.txt         # Python package dependencies
├── .gitignore               # Excludes virtual environments, models, and biometrics
│
│   # Educational / Development Milestones
├── day1_camera.py           # Camera connectivity & OpenCV setup
├── day2_face_detection.py   # MediaPipe face detection demo
├── day3_embedding.py        # InsightFace feature extraction demo
├── enroll.py                # Single-frame baseline enrollment
├── recognize.py             # Baseline face recognizer
└── recognize_stable.py      # Smoothed face recognition with buffer
```

---

## 🛠️ Tech Stack

- **[InsightFace](https://github.com/deepinsight/insightface)**: State-of-the-art 2D and 3D face analysis.
- **[Google MediaPipe](https://developers.google.com/mediapipe)**: Ultra-fast hand landmark and gesture recognition.
- **[OpenCV (cv2)](https://opencv.org/)**: Video capture, frame preprocessing, and HUD overlay rendering.
- **[NumPy](https://numpy.org/)**: Vector math and cosine similarity computations.
- **[ONNX Runtime](https://onnxruntime.ai/)**: High-performance inference engine for deep learning models.

---

## 📄 License

This project is licensed under the MIT License.
