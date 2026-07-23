# 🛡️ Human Fall Detection System using YOLO Pose Estimation

An AI-powered Human Fall Detection Web Application built using **Flask**, **YOLO Pose Estimation**, and **OpenCV**. The system detects human falls from uploaded videos, generates an annotated output video, provides a fall detection timeline, and visualizes the fall confidence score through interactive graphs.

---

## 📌 Features

- Upload and analyze videos through a web interface
- Human pose estimation using YOLO Pose models
- Fall detection using body posture analysis and Finite State Machine (FSM)
- Supports multiple YOLO models:
  - YOLOv8n-Pose
  - YOLO11n-Pose
  - YOLO26n-Pose
- Generates annotated output video
- Displays detection timeline
- Interactive confidence graph
- Benchmark comparison between different YOLO models
- Browser-compatible video playback using FFmpeg

---

## 🏗️ Project Architecture

```
Upload Video
      │
      ▼
YOLO Pose Detection
      │
      ▼
Pose Keypoint Extraction
      │
      ▼
Feature Extraction
      │
      ▼
Fall Confidence Score
      │
      ▼
Finite State Machine (FSM)
      │
      ▼
Fall Detection Decision
      │
      ▼
Annotated Video + Timeline + Graph
```

---

## 🖥️ Tech Stack

### Backend

- Python
- Flask

### Computer Vision

- OpenCV
- Ultralytics YOLO Pose

### Frontend

- HTML
- CSS
- JavaScript

### Visualization

- Plotly

### Video Processing

- FFmpeg

---

## 📁 Project Structure

```
Fall-Detection-Web-App/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
├── static/
│
├── uploads/
├── outputs/
│
├── weights/
│   ├── yolov8n-pose.pt
│   ├── yolo11n-pose.pt
│   └── yolo26n-pose.pt
│
└── benchmark_reports/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/drcdebtosh/Fall-Detection-Web-App.git

cd Fall-Detection-Web-App
```

### Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install FFmpeg (Required)

The application converts processed videos into a browser-compatible format using FFmpeg.

### Windows

Download FFmpeg:

https://ffmpeg.org/download.html

Add the **ffmpeg/bin** directory to your system PATH.

Verify installation:

```bash
ffmpeg -version
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5050
```

---

## 🚀 How It Works

1. Upload a video.
2. Select the YOLO pose model.
3. The system detects human poses frame-by-frame.
4. Body posture features are extracted.
5. A Fall Confidence Score is calculated.
6. The Finite State Machine validates temporal state transitions.
7. The annotated video, timeline, and graph are displayed.

---

## 📊 Supported Models

| Model | Purpose |
|--------|----------|
| YOLOv8n-Pose | Pose Detection |
| YOLO11n-Pose | Pose Detection |
| YOLO26n-Pose | Pose Detection |

---

## 📈 Output

The application generates:

- Annotated Detection Video
- Detection Timeline
- Confidence Score Graph
- Fall Detection Summary

---

## 📷 Screenshots

### Home Page

(Add Screenshot)

---

### Analysis Dashboard

(Add Screenshot)

---

### Detection Timeline

(Add Screenshot)

---

### Confidence Graph

(Add Screenshot)

---

## 📚 Research Background

This project was developed as a Final Year Major Project to investigate real-time human fall detection using YOLO Pose Estimation combined with posture-based analysis and a Finite State Machine for temporal validation.

The application also provides benchmarking across multiple YOLO Pose models to evaluate performance.

---

## 👨‍💻 Authors

**Debatosh Roychowdhury**

Final Year B.Tech (Computer Science & Engineering)

Techno International New Town

Kolkata, India

---

## 📄 License

This project is intended for educational and research purposes.

```

---

# ⭐ Before uploading to GitHub

I also recommend adding these:

- A project banner image at the top.
- 4–5 screenshots of the UI.
- A demo GIF (30–60 seconds) showing the upload and analysis flow.
- Topics on GitHub such as `flask`, `computer-vision`, `yolo`, `pose-estimation`, `fall-detection`, `opencv`, `python`, and `deep-learning`.

A repository with screenshots and a short demo video looks much more polished and makes a stronger impression than code alone.