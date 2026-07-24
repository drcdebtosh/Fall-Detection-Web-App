![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-black)
![YOLO](https://img.shields.io/badge/YOLO-Pose-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-Educational-orange)

# 🛡️ Nexus FallNet AI
### Intelligent Human Fall Detection using YOLO Pose Estimation

An AI-powered Human Fall Detection Web Application built with **Flask**, **YOLO Pose Estimation**, **OpenCV**, and **PyTorch**. The system detects human falls from uploaded videos or live streams using posture analysis and a Finite State Machine (FSM), then generates an annotated output video, confidence graph, detection timeline, and model benchmark comparisons.

---

## 📸 Application Preview

### Home Page

![Home Screen](screenshots/Home%20Screen.png)

### Fall Detection Result

![Output](screenshots/EMA%20UR%20FALL%20Output.png)

### Model Benchmark Comparison

![Benchmark](screenshots/Model%20Comparison.png)

### Confusion Matrix

![Confusion Matrix](screenshots/Confusion%20Matrix.png)

---

# ✨ Features

- 🎥 Upload and analyze videos directly through the web interface
- 📹 Real-time fall detection using Webcam or IP Camera
- 🧍 Human pose estimation using YOLO Pose models
- 🧠 Posture-based feature extraction
- 📈 Fall Confidence Score calculation
- 🔄 Finite State Machine (FSM) based temporal validation
- 📊 Interactive benchmark comparison across multiple YOLO models
- 📉 Confusion Matrix generation
- 🎬 Annotated output video generation
- 📅 Detection timeline visualization
- 🌐 Browser-compatible video conversion using FFmpeg
- 🐳 Docker support for cross-platform deployment

---

# 🏗️ System Architecture

```
                   Input Video / Live Camera
                             │
                             ▼
                  YOLO Pose Estimation
                             │
                             ▼
                 Human Pose Keypoint Detection
                             │
                             ▼
                 Posture Feature Extraction
                             │
                             ▼
                 Fall Confidence Calculation
                             │
                             ▼
               Finite State Machine (FSM)
                             │
                             ▼
                  Fall Detection Decision
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   Annotated Video     Detection Timeline   Confidence Graph
```

---

# 🤖 Supported YOLO Models

| Model | Purpose |
|--------|----------|
| YOLOv8n-Pose | Pose Estimation |
| YOLO11n-Pose | Pose Estimation |
| YOLO26n-Pose | Pose Estimation |

---

# 🛠️ Technology Stack

## Backend

- Python
- Flask

## Computer Vision

- OpenCV
- Ultralytics YOLO Pose
- PyTorch

## Frontend

- HTML5
- CSS3
- JavaScript

## Video Processing

- FFmpeg

## Deployment

- Docker
- Docker Compose

---

# 📁 Project Structure

```
Fall-Detection-Web-App
│
├── animation1/            # Frontend animation assets
├── benchmarks/            # Benchmark reports
├── screenshots/           # README screenshots
├── static/                # CSS, JavaScript & images
├── templates/             # HTML templates
├── uploads/               # Uploaded videos
├── outputs/               # Processed videos
│
├── app.py                 # Flask web application
├── main.py                # Fall detection engine
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── yolo11n-pose.pt
├── yolov8n-pose.pt
└── yolo26n-pose.pt
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/drcdebtosh/Fall-Detection-Web-App.git

cd Fall-Detection-Web-App
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install FFmpeg

FFmpeg is required for browser-compatible video conversion.

Verify installation:

```bash
ffmpeg -version
```

---

## 5. Run the Application

```bash
python app.py
```

Open your browser:

```
http://localhost:5050
```

---

# 🐳 Docker Deployment

Build the Docker image:

```bash
docker compose build
```

Run the application:

```bash
docker compose up
```

Open:

```
http://localhost:5050
```

---

# 🚀 How It Works

1. Upload a video or connect a live camera.
2. Select a YOLO Pose model.
3. Detect human pose keypoints.
4. Extract posture-based features.
5. Calculate Fall Confidence Score.
6. Validate state transitions using the Finite State Machine.
7. Generate the final detection result.
8. Produce:
   - Annotated Video
   - Detection Timeline
   - Confidence Graph
   - Confusion Matrix
   - Benchmark Results

---

# 📊 Outputs

The application generates:

- ✅ Annotated Video
- ✅ Detection Timeline
- ✅ Fall Confidence Graph
- ✅ Confusion Matrix
- ✅ Model Benchmark Report
- ✅ Fall Detection Summary

---

# 🎯 Benchmarking

The application allows benchmarking of multiple YOLO Pose models based on:

- Accuracy
- Precision
- Recall
- F1 Score
- Processing Speed
- Confusion Matrix

This enables comparative evaluation of different pose estimation models for fall detection.

---

# 📚 Research Background

This project was developed as a **Final Year B.Tech Major Project** at **Techno International New Town**.

The objective is to investigate intelligent human fall detection using posture analysis, YOLO Pose Estimation, and Finite State Machine-based temporal validation for reliable fall recognition.

---

# 👨‍💻 Authors

### Students

- Debatosh Roychowdhury
- Rajnaya Ghosh
- Sneha Shree
- Diganta Das

### Project Guide

**Dr. Chinmoy Kar**

Associate Professor

Department of Computer Science & Engineering

Techno International New Town

Kolkata, India

---

# 📄 License

This project is intended solely for educational and research purposes.

---

## ⭐ If you found this project useful, consider giving it a Star!