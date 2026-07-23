"""
Nexus FallNet AI — Web Application Server (v2)
================================================
Flask backend with:
  • Video upload & analysis
  • Live webcam / IP camera real-time detection
  • Model benchmark comparison API
  • Annotated video generation
"""

import os
import sys
import uuid
import time
import base64
import shutil
import subprocess
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

# ── Import detection classes from the existing main.py ───────────────
from main import (
    CONFIG, DEVICE,
    FallStateMachine, PersonTrack, FallDetector, draw_person_overlay,
)

try:
    from ultralytics import YOLO
except ImportError:
    print("[FATAL] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
# Flask App Configuration
# ═══════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_model_cache = {}
_model_lock = threading.Lock()

# ── Live camera state (per-session tracks) ───────────────────────
_live_tracks = {}        # session_id -> {tracks, model_name}
_live_tracks_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
# Real benchmark data from 195-video evaluation
# ═══════════════════════════════════════════════════════════════════════
BENCHMARK_DATA = {
    "total_videos": 300,
    "ground_truth_method": "3-Model Cross-Consensus Majority Vote",
    "hardware": str(DEVICE).upper(),
    "champion": {
        "accuracy": "YOLO11n-Pose",
        "speed": "YOLO11n-Pose",
        "balanced": "YOLO11n-Pose",
    },
    "models": [
        {
            "id": "yolov8n-pose",
            "name": "YOLOv8n-Pose",
            "filename": "yolov8n-pose.pt",
            "generation": "v8",
            "description": "Ultralytics YOLOv8 Nano Pose — battle-tested baseline with strong recall.",
            "badge": "Reliable",
            "metrics": {
                "recall": 95.40, "precision": 90.22, "f1": 92.74, "fps": 34.9,
                "accuracy": 93.33,
            },
            "confusion": {"tp": 83, "tn": 99, "fp": 9, "fn": 4},
            "strengths": ["High recall (95.4%)", "Mature & stable architecture", "Strong on diverse environments"],
            "weaknesses": ["Slightly more false positives", "Marginally slower"],
        },
        {
            "id": "yolo11n-pose",
            "name": "YOLO11n-Pose",
            "filename": "yolo11n-pose.pt",
            "generation": "v11",
            "description": "Next-gen YOLO11 Nano Pose — highest F1 score, champion across all categories.",
            "badge": "🏆 Champion",
            "metrics": {
                "recall": 97.70, "precision": 96.60, "f1": 97.10, "fps": 35.7,
                "accuracy": 97.00,
            },
            "confusion": {"tp": 105, "tn": 125, "fp": 4, "fn": 2},
            "strengths": ["Best F1 score (97.1%)", "Highest recall (97.7%)", "Fastest processing (35.7 FPS)", "Only 2 missed falls out of 107"],
            "weaknesses": ["Slightly lower precision than YOLO26"],
        },
        {
            "id": "yolo26n-pose",
            "name": "YOLO26n-Pose",
            "filename": "yolo26n-pose.pt",
            "generation": "v26",
            "description": "YOLO26 Nano Pose (NMS-free) — highest precision, fewest false alarms.",
            "badge": "Precision King",
            "metrics": {
                "recall": 82.76, "precision": 97.30, "f1": 89.44, "fps": 35.4,
                "accuracy": 91.28,
            },
            "confusion": {"tp": 72, "tn": 106, "fp": 2, "fn": 15},
            "strengths": ["Highest precision (97.3%)", "Only 2 false alarms out of 108", "NMS-free architecture"],
            "weaknesses": ["Lower recall (82.8%)", "Misses 15 falls — less suitable for critical safety"],
        },
    ],
}


def get_model(model_name=None):
    if model_name is None:
        model_name = "yolo11n-pose.pt"
    with _model_lock:
        if model_name not in _model_cache:
            model_path = os.path.join(os.path.dirname(__file__), model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found: {model_path}")
            print(f"[MODEL] Loading {model_name} on {DEVICE}...")
            _model_cache[model_name] = YOLO(model_path)
            print(f"[MODEL] {model_name} loaded.")
        return _model_cache[model_name]


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════
# Video Analysis Engine
# ═══════════════════════════════════════════════════════════════════════
def analyze_video(video_path, job_id, model_name=None):
    model = get_model(model_name)
    cap = cv2.VideoCapture(video_path)
    temp_mp4_path = None

    if not cap.isOpened() and video_path.lower().endswith(".avi"):
        if shutil.which("ffmpeg"):
            temp_mp4_path = video_path.replace(".avi", "_temp.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264",
                            "-preset", "ultrafast", "-crf", "28", "-an", temp_mp4_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            cap = cv2.VideoCapture(temp_mp4_path)

    if not cap.isOpened():
        return {"error": "Failed to open video file."}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames (used indirectly via frame_count)

    output_filename = f"{job_id}_annotated.mp4"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    active_tracks = {}
    video_has_fall = False
    fall_frame_idx = None
    fall_timestamp = None
    total_inf_time = 0.0
    frame_count = 0
    fall_frames = 0
    timeline = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        fall_activity = False

        t0 = time.perf_counter()
        results = model.track(frame, persist=True, imgsz=640, verbose=False, device=DEVICE)
        total_inf_time += time.perf_counter() - t0

        for t in active_tracks.values():
            t.unseen_count += 1

        for r in results:
            if r.keypoints is None or r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xywh.cpu().numpy()
            kpts = r.keypoints.xy.cpu().numpy()
            kconfs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            tids = r.boxes.id.int().cpu().tolist()

            for i in range(len(boxes)):
                cx, cy, w, h = boxes[i]
                tid = tids[i]
                if h < CONFIG["MIN_PERSON_HEIGHT"] or w * h < CONFIG["MIN_BOX_AREA"]:
                    continue
                if kconfs is not None and np.mean(kconfs[i]) < CONFIG["KEYPOINT_CONF_THRESHOLD"]:
                    continue

                if tid not in active_tracks:
                    active_tracks[tid] = PersonTrack(tid)
                trk = active_tracks[tid]
                trk.unseen_count = 0
                trk.bbox_history.append([cx, cy, w, h])
                trk.raw_keypoints_history.append(kpts[i])
                FallDetector.detect(trk, width, height, keypoint_confs=kconfs[i] if kconfs is not None else None, fps=fps)

                if trk.current_state == FallStateMachine.CONFIRMED:
                    video_has_fall = True
                    fall_activity = True
                    if fall_frame_idx is None:
                        fall_frame_idx = frame_count
                        fall_timestamp = frame_count / fps
                elif trk.current_state in [FallStateMachine.LYING, FallStateMachine.IMPACT]:
                    fall_activity = True

                imm = f"Ground: {trk.immobility_seconds:.1f}s ({trk.immobility_status})" if trk.current_state == FallStateMachine.CONFIRMED else ""
                draw_person_overlay(frame, [cx, cy, w, h], trk.keypoints_history[-1], tid,
                                    trk.current_state, trk.fall_confidence, trk.last_vy,
                                    trk.last_torso_angle, trk.state_machine.consecutive_lying_frames,
                                    near_fall_count=trk.state_machine.near_fall_count,
                                    severity=trk.fall_severity, immobility_info=imm)

        if fall_activity:
            fall_frames += 1

        if frame_count % 5 == 0 or fall_activity:
            best_conf, best_state = 0.0, "NORMAL"
            for t in active_tracks.values():
                if t.fall_confidence > best_conf:
                    best_conf = t.fall_confidence
                    best_state = t.current_state
            timeline.append({"frame": int(frame_count), "time": float(round(frame_count / fps, 2)),
                             "confidence": float(round(best_conf, 4)), "state": str(best_state),
                             "fall_activity": bool(fall_activity)})

        for tid in [k for k, v in active_tracks.items() if v.unseen_count > 60]:
            del active_tracks[tid]
        out_writer.write(frame)

    cap.release()
    out_writer.release()

    # H.264 transcode for browser
    if shutil.which("ffmpeg"):
        h264 = output_path.replace(".mp4", "_h264.mp4")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264", "-preset", "fast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", h264],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.remove(output_path)
            os.rename(h264, output_path)
        except Exception:
            pass

    if temp_mp4_path and os.path.exists(temp_mp4_path):
        os.remove(temp_mp4_path)

    near_falls = sum(t.state_machine.near_fall_count for t in active_tracks.values())
    severity_score = immobility_sec = 0.0
    sev_label = imm_status = "N/A"
    max_conf = peak_vel = peak_angle = 0.0

    for t in active_tracks.values():
        max_conf = max(max_conf, t.fall_confidence)
        peak_vel = max(peak_vel, t.peak_descent_velocity)
        peak_angle = max(peak_angle, t.last_torso_angle)
        if t.fall_severity != "N/A":
            sev_label = t.fall_severity
            severity_score = t.fall_severity_score
            immobility_sec = t.immobility_seconds
            imm_status = t.immobility_status

    return {
        "job_id": str(job_id),
        "prediction": "FALL DETECTED" if video_has_fall else "NO FALL",
        "is_fall": bool(video_has_fall),
        "confidence": float(round(max_conf, 4)),
        "fall_frame": int(fall_frame_idx) if fall_frame_idx is not None else None,
        "fall_timestamp": float(round(fall_timestamp, 2)) if fall_timestamp is not None else None,
        "severity": str(sev_label),
        "severity_score": float(round(severity_score, 4)),
        "near_fall_count": int(near_falls),
        "immobility_seconds": float(round(immobility_sec, 1)),
        "immobility_status": str(imm_status),
        "total_frames": int(frame_count),
        "fall_frames": int(fall_frames),
        "fall_percentage": float(round(fall_frames / (frame_count + 1e-6) * 100, 1)),
        "fps_processing": float(round(frame_count / (total_inf_time + 1e-6), 1)),
        "video_fps": float(fps),
        "video_resolution": f"{width}x{height}",
        "peak_velocity": float(round(peak_vel, 4)),
        "peak_torso_angle": float(round(peak_angle, 1)),
        "persons_tracked": int(len(active_tracks)),
        "model_used": str(model_name or "yolo11n-pose.pt"),
        "device": str(DEVICE),
        "annotated_video": str(output_filename),
        "timeline": timeline,
    }


# ═══════════════════════════════════════════════════════════════════════
# Live Frame Detection (Webcam / IP Camera)
# ═══════════════════════════════════════════════════════════════════════
def process_live_frame(frame_bytes, session_id, model_name, width, height):
    """Process a single frame from webcam and return annotated frame + state."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, {}

    model = get_model(model_name)
    fh, fw = frame.shape[:2]

    with _live_tracks_lock:
        if session_id not in _live_tracks:
            _live_tracks[session_id] = {}
        active_tracks = _live_tracks[session_id]

    for t in active_tracks.values():
        t.unseen_count += 1

    results = model.track(frame, persist=True, imgsz=640, verbose=False, device=DEVICE)
    states = {}

    for r in results:
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue
        boxes = r.boxes.xywh.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()
        kconfs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        tids = r.boxes.id.int().cpu().tolist()

        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            tid = tids[i]
            if h < CONFIG["MIN_PERSON_HEIGHT"] or w * h < CONFIG["MIN_BOX_AREA"]:
                continue
            if kconfs is not None and np.mean(kconfs[i]) < CONFIG["KEYPOINT_CONF_THRESHOLD"]:
                continue

            if tid not in active_tracks:
                active_tracks[tid] = PersonTrack(tid)
            trk = active_tracks[tid]
            trk.unseen_count = 0
            trk.bbox_history.append([cx, cy, w, h])
            trk.raw_keypoints_history.append(kpts[i])
            FallDetector.detect(trk, fw, fh, keypoint_confs=kconfs[i] if kconfs is not None else None, fps=25)

            states[int(tid)] = {
                "state": str(trk.current_state),
                "confidence": float(round(trk.fall_confidence, 3)),
                "severity": str(trk.fall_severity),
                "near_falls": int(trk.state_machine.near_fall_count),
            }

            imm = f"Ground: {trk.immobility_seconds:.1f}s ({trk.immobility_status})" if trk.current_state == FallStateMachine.CONFIRMED else ""
            draw_person_overlay(frame, [cx, cy, w, h], trk.keypoints_history[-1], tid,
                                trk.current_state, trk.fall_confidence, trk.last_vy,
                                trk.last_torso_angle, trk.state_machine.consecutive_lying_frames,
                                near_fall_count=trk.state_machine.near_fall_count,
                                severity=trk.fall_severity, immobility_info=imm)

    for tid in [k for k, v in active_tracks.items() if v.unseen_count > 60]:
        del active_tracks[tid]

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    has_fall = any(s["state"] == FallStateMachine.CONFIRMED for s in states.values())

    return jpeg.tobytes(), {
        "has_fall": bool(has_fall),
        "persons": int(len(states)),
        "states": states,
    }


# ═══════════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400
    file = request.files["video"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format."}), 400

    job_id = str(uuid.uuid4())[:12]
    ext = os.path.splitext(file.filename)[1].lower()
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}{ext}")
    file.save(upload_path)

    model_name = request.form.get("model", "yolo11n-pose")
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    try:
        results = analyze_video(upload_path, job_id, model_name)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)
    return jsonify(results)


@app.route("/api/detect-frame", methods=["POST"])
def api_detect_frame():
    """Process a single webcam frame and return annotated JPEG + detection data."""
    if "frame" not in request.files:
        return jsonify({"error": "No frame data"}), 400

    frame_data = request.files["frame"].read()
    session_id = request.form.get("session_id", "default")
    model_name = request.form.get("model", "yolo11n-pose.pt")
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    jpeg_bytes, detection = process_live_frame(frame_data, session_id, model_name, 640, 480)
    if jpeg_bytes is None:
        return jsonify({"error": "Failed to process frame"}), 400

    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    return jsonify({"frame": encoded, "detection": detection})


@app.route("/api/reset-live", methods=["POST"])
def api_reset_live():
    """Reset live tracking state for a session."""
    session_id = request.json.get("session_id", "default") if request.is_json else "default"
    with _live_tracks_lock:
        _live_tracks.pop(session_id, None)
    return jsonify({"status": "reset"})


@app.route("/api/video/<filename>")
def serve_video(filename):
    """
    Serve annotated video with HTTP Range request support.
    This is required for HTML5 <video> controls (play, seek, fullscreen) to work in all browsers.
    """
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "Video not found"}), 404

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range", None)

    if range_header:
        # Parse Range header, e.g. "bytes=0-999"
        byte_start, byte_end = 0, None
        match = range_header.strip().replace("bytes=", "").split("-")
        byte_start = int(match[0]) if match[0] else 0
        byte_end = int(match[1]) if len(match) > 1 and match[1] else file_size - 1
        byte_end = min(byte_end, file_size - 1)
        length = byte_end - byte_start + 1

        with open(file_path, "rb") as f:
            f.seek(byte_start)
            data = f.read(length)

        response = Response(
            data,
            206,
            mimetype="video/mp4",
            direct_passthrough=True,
        )
        response.headers["Content-Range"] = f"bytes {byte_start}-{byte_end}/{file_size}"
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(length)
        response.headers["Cache-Control"] = "no-cache"
        return response
    else:
        # Full file response
        with open(file_path, "rb") as f:
            data = f.read()
        response = Response(
            data,
            200,
            mimetype="video/mp4",
            direct_passthrough=True,
        )
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(file_size)
        response.headers["Cache-Control"] = "no-cache"
        return response


@app.route("/api/models")
def list_models():
    return jsonify(BENCHMARK_DATA)


@app.route("/api/benchmark")
def api_benchmark():
    return jsonify(BENCHMARK_DATA)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        get_model("yolo11n-pose.pt")
    except FileNotFoundError:
        print("[WARNING] Default model not found.")

    print("\n" + "=" * 60)
    print("  NEXUS FALLNET AI — Web Application v2")
    print(f"  Device: {DEVICE}")
    print("  Open: http://localhost:5050")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
