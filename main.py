import os
import cv2  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import time
import random
import shutil
import subprocess
import re
from pathlib import Path
from collections import deque
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore

# ===================== CONFIGURATION =====================
CONFIG = {
    "DATASET_PATH": "./DATASET",                   # Local path fallback
    # ── VIDEO FOLDERS ─────────────────────────────────────────────────
    # Folders to auto-scan for video files (.mp4, .avi, .mov, .mkv).
    # All videos found inside these folders will be added automatically.
    "VIDEO_FOLDERS": [
        "./Dataset2",
        "./Dataset",
    ],
    # ── ADDITIONAL TEST VIDEOS ────────────────────────────────────────
    # Extra loose video files (in the project root) to include.
    # Ground truth is determined automatically via cross-model consensus.
    "TEST_VIDEOS": [],
    "POSE_MODELS": [
        "yolov8n-pose.pt",                        # YOLOv8 Nano Pose
        "yolo11n-pose.pt",                        # YOLOv11 Nano Pose
        "yolo26n-pose.pt"                         # YOLO26 Nano Pose (NMS-free)
    ],
    "SAVE_ANNOTATED_VIDEOS": False,               # FALSE: Avoids filling up SSD
    "SHOW_LIVE_PREVIEW": False,                    # FALSE: Disables window popups
    
    # Ignore criteria for noise / background people
    "MIN_PERSON_HEIGHT": 30,                      # Minimum height in pixels
    "MIN_BOX_AREA": 800,                          # Minimum bounding box area
    "KEYPOINT_CONF_THRESHOLD": 0.35,              # Minimum average confidence of keypoints
    
    # Confidence-Aware EMA (CA-EMA) smoothing parameters
    "EMA_ALPHA_MIN": 0.15,                        # Alpha for low-confidence detections (heavy smoothing)
    "EMA_ALPHA_MAX": 0.50,                        # Alpha for high-confidence detections (fast response)
    
    # Detection thresholds
    "VELOCITY_THRESHOLD": 0.45,                   # Downward velocity threshold (normalized by height/sec)
    "ACCELERATION_THRESHOLD": 0.20,               # Downward acceleration threshold (normalized)
    "ASPECT_RATIO_THRESHOLD": 1.25,               # Width/height ratio threshold indicating horizontal posture
    "BODY_ANGLE_THRESHOLD": 60.0,                 # Torso angle in degrees from vertical indicating lying
    "GROUND_THRESHOLD": 0.50,                     # Ratio of screen height below which person center must sit
    
    # Lying verification duration
    "LYING_TIME_THRESHOLD": 20,                   # Consecutive frames lying/impact to confirm fall (~1 sec at 20-30 FPS)
    
    # Recovery parameters
    "RECOVERY_ANGLE_THRESHOLD": 40.0,             # Torso angle below this triggers recovery check
    "RECOVERY_ASPECT_RATIO": 0.85,                # Aspect ratio below this triggers recovery check
    "RECOVERY_HEIGHT_RATIO": 0.75,                # Height restoration threshold relative to standing height
    
    # ── USP: Near-Fall / Stumble Detection ────────────────────────
    "NEAR_FALL_MIN_VELOCITY": 0.30,               # Minimum velocity during descent to qualify as near-fall
    
    # ── USP: Fall Severity Classification ─────────────────────────
    "SEVERITY_SOFT_THRESHOLD": 0.35,              # Severity score below this = SOFT
    "SEVERITY_SEVERE_THRESHOLD": 0.65,            # Severity score above this = SEVERE (else MODERATE)
    
    # ── USP: Post-Fall Immobility Timer ───────────────────────────
    "IMMOBILITY_DELAYED_SEC": 30.0,               # Seconds on ground before DELAYED_RECOVERY
    "IMMOBILITY_EMERGENCY_SEC": 120.0,            # Seconds on ground before IMMOBILE_EMERGENCY
    
    # Benchmark settings
    "MAX_VIDEOS_PER_ENV": None,                   # None to run all videos, or int to sample
}

# Terminal Formatting Color Codes
CLR_RESET = "\033[0m"   
CLR_BOLD = "\033[1m"
CLR_RED = "\033[91m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"

# Determine local hardware accelerator
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# MS COCO 17 Keypoints Adjacency Pairs for visual rendering
COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (11, 12),                            # Hips
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16),                  # Right leg
    (5, 11), (6, 12)                     # Torso
]

# ===================== SKELETON SMOOTHING FILTER (CA-EMA) =====================
class EMAFilter:
    """
    Confidence-Aware Exponential Moving Average (CA-EMA) filter.
    
    Adapts the smoothing factor α per-frame based on the detection confidence
    of each keypoint:
        α_effective = α_min + (α_max - α_min) × confidence
    
    - High confidence → α closer to α_max → trusts new detection (faster response)
    - Low confidence  → α closer to α_min → trusts history (heavier smoothing)
    """
    def __init__(self, alpha_min=0.15, alpha_max=0.50):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.value = None
        
    def update(self, raw_val, confidence=1.0):
        # Clamp confidence to [0, 1]
        confidence = float(np.clip(confidence, 0.0, 1.0))
        # Compute adaptive alpha: interpolate between α_min and α_max
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * confidence
        
        if self.value is None:
            self.value = np.array(raw_val, dtype=np.float32)
        else:
            self.value = alpha * np.array(raw_val, dtype=np.float32) + (1.0 - alpha) * self.value
        return self.value

# ===================== STATE MACHINE =====================
class FallStateMachine:
    """
    Tracks state transitions: NORMAL -> DESCENDING -> IMPACT -> LYING -> CONFIRMED FALL
    Also tracks: near-fall events, post-fall immobility duration.
    """
    NORMAL = "NORMAL"
    DESCENDING = "DESCENDING"
    IMPACT = "IMPACT"
    LYING = "LYING"
    CONFIRMED = "CONFIRMED FALL"
    
    def __init__(self, lying_threshold=20):
        self.state = FallStateMachine.NORMAL
        self.lying_threshold = lying_threshold
        self.consecutive_lying_frames = 0
        
        # USP: Near-Fall Detection
        self.near_fall_count = 0
        self._was_descending = False         # Was in DESCENDING state last frame
        self._was_impact = False             # Was in IMPACT state last frame
        self._near_fall_this_frame = False   # Flag set on the frame a near-fall occurs
        
        # USP: Post-Fall Immobility Timer
        self.confirmed_immobility_frames = 0  # Frames spent in CONFIRMED state
        self.peak_immobility_frames = 0       # Longest continuous immobility observed
        
    def update(self, is_descending, is_horizontal, is_grounded, has_recovered):
        prev_state = self.state
        self._near_fall_this_frame = False
        
        if has_recovered:
            # If recovering from a fall-related state (DESCENDING/IMPACT), count as near-fall
            if prev_state in [FallStateMachine.DESCENDING, FallStateMachine.IMPACT]:
                self.near_fall_count += 1
                self._near_fall_this_frame = True
            # Track peak immobility before recovery
            if prev_state == FallStateMachine.CONFIRMED:
                self.peak_immobility_frames = max(self.peak_immobility_frames, self.confirmed_immobility_frames)
            self.state = FallStateMachine.NORMAL
            self.consecutive_lying_frames = 0
            self.confirmed_immobility_frames = 0
            return self.state
            
        if self.state == FallStateMachine.NORMAL:
            if is_descending:
                self.state = FallStateMachine.DESCENDING
            elif is_horizontal and is_grounded:
                self.state = FallStateMachine.IMPACT
                
        elif self.state == FallStateMachine.DESCENDING:
            if is_horizontal and is_grounded:
                self.state = FallStateMachine.IMPACT
            elif not is_descending:
                # Descent stopped without impact → near-fall (stumble/balance recovery)
                self.near_fall_count += 1
                self._near_fall_this_frame = True
                self.state = FallStateMachine.NORMAL
                
        elif self.state == FallStateMachine.IMPACT:
            if is_horizontal and is_grounded:
                self.state = FallStateMachine.LYING
                self.consecutive_lying_frames = 1
            else:
                # Impact without follow-through → near-fall
                self.near_fall_count += 1
                self._near_fall_this_frame = True
                self.state = FallStateMachine.NORMAL
                
        elif self.state == FallStateMachine.LYING:
            if is_horizontal and is_grounded:
                self.consecutive_lying_frames += 1
                if self.consecutive_lying_frames >= self.lying_threshold:
                    self.state = FallStateMachine.CONFIRMED
            else:
                self.state = FallStateMachine.NORMAL
                self.consecutive_lying_frames = 0
                
        elif self.state == FallStateMachine.CONFIRMED:
            # USP: Post-Fall Immobility Timer — count frames in CONFIRMED state
            self.confirmed_immobility_frames += 1
            self.peak_immobility_frames = max(self.peak_immobility_frames, self.confirmed_immobility_frames)
            if not (is_horizontal and is_grounded):
                self.state = FallStateMachine.NORMAL
                self.consecutive_lying_frames = 0
                # Don't reset confirmed_immobility_frames here; keep peak for reporting
                
        return self.state

# ===================== INDEPENDENT PERSON TRACK =====================
class PersonTrack:
    """
    Stores metrics, history, filters, and state machine for a tracked person.
    """
    def __init__(self, track_id, max_history=30):
        self.track_id = track_id
        self.max_history = max_history
        self.bbox_history = deque(maxlen=max_history)
        self.raw_keypoints_history = deque(maxlen=max_history)
        self.keypoints_history = deque(maxlen=max_history)
        self.filters = [EMAFilter(alpha_min=CONFIG["EMA_ALPHA_MIN"], alpha_max=CONFIG["EMA_ALPHA_MAX"]) for _ in range(17)]
        
        self.velocity_history = deque(maxlen=max_history)
        self.acceleration_history = deque(maxlen=max_history)
        
        self.state_machine = FallStateMachine(lying_threshold=CONFIG["LYING_TIME_THRESHOLD"])
        self.current_state = FallStateMachine.NORMAL
        self.fall_confidence = 0.0
        self.unseen_count = 0
        
        self.normal_standing_height = None
        self.normal_standing_width = None
        
        self.last_vy = 0.0
        self.last_v_top = 0.0
        self.last_ay = 0.0
        self.last_torso_angle = 0.0
        self.last_aspect_ratio = 0.0
        self.reason = "Normal activity"
        
        # USP: Fall Severity Classification
        self.peak_descent_velocity = 0.0      # Maximum downward velocity observed during this fall
        self.peak_deceleration = 0.0          # Maximum deceleration (sudden stop) at impact
        self.min_head_ground_dist = float('inf')  # Minimum head-to-ground distance ratio
        self.fall_severity = "N/A"            # SOFT / MODERATE / SEVERE
        self.fall_severity_score = 0.0        # 0.0–1.0 severity score
        
        # USP: Post-Fall Immobility Assessment
        self.immobility_status = "N/A"        # SELF_RECOVERED / DELAYED_RECOVERY / IMMOBILE_EMERGENCY
        self.immobility_seconds = 0.0         # Duration on ground in seconds

    def update(self, bbox, keypoints, keypoint_confs=None):
        cx, cy, w, h = bbox
        self.bbox_history.append(bbox)
        self.raw_keypoints_history.append(keypoints)
        
        # Smooth keypoints using Confidence-Aware EMA (CA-EMA)
        smoothed_kpts = np.zeros_like(keypoints)
        for idx in range(17):
            conf = keypoint_confs[idx] if keypoint_confs is not None else 1.0
            smoothed_kpts[idx] = self.filters[idx].update(keypoints[idx], confidence=conf)
        self.keypoints_history.append(smoothed_kpts)
        
        # Update standing statistics
        aspect_ratio = w / (h + 1e-6)
        if aspect_ratio < 0.70:
            if self.normal_standing_height is None or h > self.normal_standing_height:
                self.normal_standing_height = h
                self.normal_standing_width = w
                
        # Calculate velocity and acceleration
        vy = 0.0
        v_top = 0.0
        ay = 0.0
        if len(self.bbox_history) >= 5:
            cys = [b[1] for b in list(self.bbox_history)[-5:]]
            # Velocity relative to current height per frame
            vy = (cys[-1] - cys[0]) / (4.0 * h + 1e-6)
            self.velocity_history.append(vy)
            
            # Top-edge velocity relative to current height per frame
            tops = [b[1] - b[3]/2.0 for b in list(self.bbox_history)[-5:]]
            v_top = (tops[-1] - tops[0]) / (4.0 * h + 1e-6)
            
            if len(self.velocity_history) >= 3:
                vels = list(self.velocity_history)[-3:]
                ay = vels[-1] - vels[0]
                self.acceleration_history.append(ay)
        else:
            self.velocity_history.append(0.0)
            self.acceleration_history.append(0.0)
            
        # Compute Torso Angle (Neck/Shoulder midpoint to Hip midpoint)
        nose = smoothed_kpts[0]
        l_shoulder = smoothed_kpts[5]
        r_shoulder = smoothed_kpts[6]
        l_hip = smoothed_kpts[11]
        r_hip = smoothed_kpts[12]
        
        torso_angle = 0.0
        if l_shoulder[0] > 0 and r_shoulder[0] > 0 and l_hip[0] > 0 and r_hip[0] > 0:
            neck_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
            neck_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            hip_x = (l_hip[0] + r_hip[0]) / 2.0
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
            dx = hip_x - neck_x
            dy = hip_y - neck_y
            torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
        elif nose[0] > 0 and l_hip[0] > 0:
            hip_x = (l_hip[0] + r_hip[0]) / 2.0 if r_hip[0] > 0 else l_hip[0]
            hip_y = (l_hip[1] + r_hip[1]) / 2.0 if r_hip[1] > 0 else l_hip[1]
            dx = hip_x - nose[0]
            dy = hip_y - nose[1]
            torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
            
        self.last_vy = vy
        self.last_v_top = v_top
        self.last_ay = ay
        self.last_torso_angle = torso_angle
        self.last_aspect_ratio = aspect_ratio
        
        return vy, ay, torso_angle, aspect_ratio

# ===================== FALL DETECTOR =====================
class FallDetector:
    """
    Evaluates tracks, updates states, and calculates fused fall confidence score.
    Also computes: fall severity, near-fall events, post-fall immobility status.
    """
    @staticmethod
    def detect(track, frame_width, frame_height, keypoint_confs=None, fps=25):
        if len(track.bbox_history) < 1:
            return
            
        cx, cy, w, h = track.bbox_history[-1]
        vy, ay, torso_angle, aspect_ratio = track.update(track.bbox_history[-1], track.raw_keypoints_history[-1], keypoint_confs=keypoint_confs)
        
        y_bottom = cy + h/2.0
        ground_ratio = y_bottom / (frame_height + 1e-6)
        
        # Binary state triggers
        is_descending = (vy > CONFIG["VELOCITY_THRESHOLD"]) or \
                        (track.last_v_top > CONFIG["VELOCITY_THRESHOLD"]) or \
                        (ay > CONFIG["ACCELERATION_THRESHOLD"])
                        
        is_shrunk = False
        if track.normal_standing_height is not None:
            is_shrunk = h < (track.normal_standing_height * 0.65)
            
        is_horizontal = (torso_angle > CONFIG["BODY_ANGLE_THRESHOLD"]) or \
                        (aspect_ratio > CONFIG["ASPECT_RATIO_THRESHOLD"]) or \
                        is_shrunk
                        
        is_grounded = ground_ratio > CONFIG["GROUND_THRESHOLD"]
        
        # Recovery check
        has_recovered = False
        if track.normal_standing_height is not None and track.current_state != FallStateMachine.NORMAL:
            if (torso_angle < CONFIG["RECOVERY_ANGLE_THRESHOLD"] and aspect_ratio < CONFIG["RECOVERY_ASPECT_RATIO"]) or \
               (h > track.normal_standing_height * CONFIG["RECOVERY_HEIGHT_RATIO"]):
                has_recovered = True
                
        # Fused Fall Confidence (0.0 to 1.0)
        vel_score = min(1.0, max(0.0, max(vy, track.last_v_top)) / (CONFIG["VELOCITY_THRESHOLD"] + 1e-6)) * 0.30
        accel_score = min(1.0, max(0.0, ay) / (CONFIG["ACCELERATION_THRESHOLD"] + 1e-6)) * 0.15
        angle_score = min(1.0, torso_angle / (CONFIG["BODY_ANGLE_THRESHOLD"] + 1e-6)) * 0.25
        ar_score = min(1.0, aspect_ratio / (CONFIG["ASPECT_RATIO_THRESHOLD"] + 1e-6)) * 0.15
        ground_score = min(1.0, ground_ratio / (CONFIG["GROUND_THRESHOLD"] + 1e-6)) * 0.15
        
        track.fall_confidence = vel_score + accel_score + angle_score + ar_score + ground_score
        
        # ── USP: Track peak metrics for severity classification ──
        current_vel = max(vy, track.last_v_top)
        if current_vel > track.peak_descent_velocity:
            track.peak_descent_velocity = current_vel
        
        # Deceleration = sudden drop in velocity (large negative change)
        if len(track.velocity_history) >= 2:
            vel_change = list(track.velocity_history)[-2] - list(track.velocity_history)[-1]
            if vel_change > track.peak_deceleration:
                track.peak_deceleration = vel_change
        
        # Head-to-ground proximity (nose keypoint Y relative to frame height)
        smoothed_kpts = track.keypoints_history[-1] if track.keypoints_history else None
        if smoothed_kpts is not None:
            nose_y = smoothed_kpts[0][1]
            if nose_y > 0:
                head_ground_ratio = nose_y / (frame_height + 1e-6)
                if head_ground_ratio < track.min_head_ground_dist:
                    track.min_head_ground_dist = head_ground_ratio
        
        # Update State Machine
        track.current_state = track.state_machine.update(is_descending, is_horizontal, is_grounded, has_recovered)
        
        # ── USP: Fall Severity Classification (on CONFIRMED state) ──
        if track.current_state == FallStateMachine.CONFIRMED and track.fall_severity == "N/A":
            # Compute severity score from peak metrics
            sv_velocity = min(1.0, track.peak_descent_velocity / (CONFIG["VELOCITY_THRESHOLD"] * 3.0 + 1e-6))
            sv_decel = min(1.0, track.peak_deceleration / (CONFIG["ACCELERATION_THRESHOLD"] * 3.0 + 1e-6))
            sv_head = min(1.0, max(0.0, track.min_head_ground_dist - 0.3) / 0.7) if track.min_head_ground_dist != float('inf') else 0.0
            # Head closer to ground (higher ratio) = worse → invert
            sv_head = 1.0 - sv_head  # Head near ground = high severity
            
            severity_score = (sv_velocity * 0.40) + (sv_decel * 0.30) + (sv_head * 0.30)
            track.fall_severity_score = severity_score
            
            if severity_score < CONFIG["SEVERITY_SOFT_THRESHOLD"]:
                track.fall_severity = "SOFT"
            elif severity_score >= CONFIG["SEVERITY_SEVERE_THRESHOLD"]:
                track.fall_severity = "SEVERE"
            else:
                track.fall_severity = "MODERATE"
        
        # ── USP: Post-Fall Immobility Assessment ──
        if track.current_state == FallStateMachine.CONFIRMED:
            immobility_frames = track.state_machine.confirmed_immobility_frames
            track.immobility_seconds = immobility_frames / (fps + 1e-6)
            
            if track.immobility_seconds >= CONFIG["IMMOBILITY_EMERGENCY_SEC"]:
                track.immobility_status = "IMMOBILE_EMERGENCY"
            elif track.immobility_seconds >= CONFIG["IMMOBILITY_DELAYED_SEC"]:
                track.immobility_status = "DELAYED_RECOVERY"
            else:
                track.immobility_status = "MONITORING"
        elif has_recovered and track.immobility_status != "N/A":
            # Person got back up
            if track.immobility_seconds < CONFIG["IMMOBILITY_DELAYED_SEC"]:
                track.immobility_status = "SELF_RECOVERED"
        
        # Assemble reasoning string for diagnostic logs
        reasons = []
        if is_descending:
            reasons.append(f"Descent velocity (vy={vy:.2f}, v_top={track.last_v_top:.2f})")
        if is_horizontal:
            reasons.append(f"Horizontal (angle={torso_angle:.1f}, AR={aspect_ratio:.2f}, shrunk={is_shrunk})")
        if is_grounded:
            reasons.append(f"Grounded ({ground_ratio:.2f})")
        if has_recovered:
            reasons.append("Recovery")
        if track.state_machine._near_fall_this_frame:
            reasons.append(f"⚡ NEAR-FALL #{track.state_machine.near_fall_count}")
            
        if not reasons:
            if aspect_ratio < 0.70:
                track.reason = "Vertical posture"
            else:
                track.reason = "Stable vertical/sitting"
        else:
            track.reason = ", ".join(reasons)

# ===================== VISUAL OVERLAY UTILITY =====================
def draw_person_overlay(frame, bbox, smoothed_kpts, track_id, state, confidence, velocity, torso_angle, lying_frames,
                        near_fall_count=0, severity="N/A", immobility_info=""):
    cx, cy, w, h = bbox
    x_min, y_min = int(cx - w/2), int(cy - h/2)
    x_max, y_max = int(cx + w/2), int(cy + h/2)
    
    # State-based coloring
    if state == FallStateMachine.CONFIRMED:
        color = (0, 0, 255)       # Red
    elif state in [FallStateMachine.IMPACT, FallStateMachine.LYING]:
        color = (0, 165, 255)    # Orange
    elif state == FallStateMachine.DESCENDING:
        color = (0, 255, 255)    # Yellow
    else:
        color = (0, 255, 0)      # Green
        
    # Draw bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Draw skeletal lines
    for kp1_idx, kp2_idx in COCO_PAIRS:
        kp1 = smoothed_kpts[kp1_idx]
        kp2 = smoothed_kpts[kp2_idx]
        if kp1[0] > 0 and kp1[1] > 0 and kp2[0] > 0 and kp2[1] > 0:
            cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), color, 2)
            
    # Draw joint dots
    for kp in smoothed_kpts:
        if kp[0] > 0 and kp[1] > 0:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 255, 255), -1)
            
    # Draw status overlay
    meta_text = [
        f"ID: {track_id} | {state}",
        f"Conf: {confidence*100:.1f}%",
        f"Vel: {velocity:.2f} | Ang: {torso_angle:.1f}°",
    ]
    if state == FallStateMachine.LYING:
        meta_text[0] += f" ({lying_frames}/{CONFIG['LYING_TIME_THRESHOLD']})"
    
    # USP: Near-fall count
    if near_fall_count > 0:
        meta_text.append(f"⚡ Near-Falls: {near_fall_count}")
    
    # USP: Fall severity + immobility
    if state == FallStateMachine.CONFIRMED:
        sev_label = f"Severity: {severity}"
        if immobility_info:
            sev_label += f" | {immobility_info}"
        meta_text.append(sev_label)
        
    y_offset = y_min - 10 if y_min - 10 > 20 else y_min + 20
    for idx, text in enumerate(meta_text):
        cv2.putText(frame, text, (x_min, y_offset + idx * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# ===================== PERFORMANCE SUB-TRACKER =====================
class SubTracker:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.total_inference_time = 0.0
        self.total_frames = 0

# ===================== PERFORMANCE TRACKER =====================
class PerformanceTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.global_tracker = SubTracker()
        self.env_trackers = {}
        self.video_details = []
        
    def _get_or_create_env_tracker(self, env_name):
        if env_name not in self.env_trackers:
            self.env_trackers[env_name] = SubTracker()
        return self.env_trackers[env_name]
        
    def update_metrics(self, predicted_state, ground_truth_state, env_name):
        p = 1 if predicted_state == "FALLEN" else 0
        g = 1 if ground_truth_state == "FALLEN" else 0
        
        for tracker in [self.global_tracker, self._get_or_create_env_tracker(env_name)]:
            if p == 1 and g == 1:
                tracker.tp += 1
            elif p == 0 and g == 0:
                tracker.tn += 1
            elif p == 1 and g == 0:
                tracker.fp += 1
            elif p == 0 and g == 1:
                tracker.fn += 1
                
    def add_latency(self, latency_seconds, frames_count, env_name):
        for tracker in [self.global_tracker, self._get_or_create_env_tracker(env_name)]:
            tracker.total_inference_time += latency_seconds
            tracker.total_frames += frames_count
            
    def calculate_metrics_for_tracker(self, tracker):
        tp, tn, fp, fn = tracker.tp, tracker.tn, tracker.fp, tracker.fn
        recall = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        avg_fps = tracker.total_frames / (tracker.total_inference_time + 1e-6)
        
        return {
            "recall": recall * 100,
            "precision": precision * 100,
            "f1": f1 * 100,
            "fps": avg_fps,
            "total_frames": tracker.total_frames,
            "matrix": (tp, tn, fp, fn)
        }
        
    def get_results_summary(self):
        summary = {"GLOBAL": self.calculate_metrics_for_tracker(self.global_tracker)}
        for env_name, tracker in self.env_trackers.items():
            summary[env_name] = self.calculate_metrics_for_tracker(tracker)
        return summary

# ===================== DATASET SCANNER & PARSER =====================
class DatasetScanner:
    @staticmethod
    def discover_dataset_path():
        if os.path.exists(CONFIG["DATASET_PATH"]) and len(os.listdir(CONFIG["DATASET_PATH"])) > 0:
            return CONFIG["DATASET_PATH"]
            
        home = os.path.expanduser("~")
        cloud_storage_dir = os.path.join(home, "Library", "CloudStorage")
        if os.path.exists(cloud_storage_dir):
            for item in os.listdir(cloud_storage_dir):
                if item.startswith("GoogleDrive-"):
                    drive_root = os.path.join(cloud_storage_dir, item, "My Drive")
                    potential_paths = [
                        os.path.join(drive_root, "Le2i Dataset", "DATASET"),
                        os.path.join(drive_root, "Le2i Dataset"),
                        os.path.join(drive_root, "DATASET")
                    ]
                    for path in potential_paths:
                        if os.path.exists(path):
                            print(f"[AUTO-DISCOVER] Natively linked to Google Drive mount: {path}")
                            return path
        try:
            import kagglehub  # type: ignore
            print("\n[INFO] No local or Google Drive dataset discovered.")
            print("[INFO] Checking Kaggle Cache / Downloading 'Le2i Fall Dataset' via kagglehub...")
            cache_path = kagglehub.dataset_download("tuyenldvn/falldataset-imvia")
            print(f"[KAGGEHUB DISCOVERY] Successfully verified Kaggle dataset root: {cache_path}")
            return cache_path
        except Exception as e:
            print(f"[ERROR] Kaggle auto-download failed: {e}")
            return None

    @classmethod
    def scan_dataset(cls):
        video_files = []
        dataset_path = cls.discover_dataset_path()
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"[INFO] Scanning directory: {dataset_path} ...")
            env_buckets = {}
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                        full_path = os.path.join(root, file)
                        normalized_path = os.path.normpath(full_path)
                        
                        parts = normalized_path.split(os.sep)
                        parts_lower = [p.lower() for p in parts]
                        file_lower = file.lower()
                        
                        relative_path = os.path.relpath(full_path, dataset_path)
                        relative_parts_lower = [p.lower() for p in relative_path.split(os.sep)]
                        
                        # Determine Environment
                        environment = "General"
                        known_envs = {
                            "coffee": "Coffee Room",
                            "home": "Home",
                            "lecture": "Lecture Room",
                            "office": "Office"
                        }
                        for key, val in known_envs.items():
                            if any(key in p for p in parts_lower):
                                environment = val
                                break
                                
                        # Decode Annotations
                        ground_truth = "NORMAL"
                        video_num = None
                        match = re.search(r'\d+', file)
                        if match:
                            video_num = int(match.group())
                            
                        parent_dir = Path(root).parent
                        annotation_dir = None
                        if os.path.exists(parent_dir):
                            for entry in os.listdir(parent_dir):
                                if "annotation" in entry.lower():
                                    annotation_dir = os.path.join(parent_dir, entry)
                                    break
                                    
                        annotation_path = None
                        if annotation_dir:
                            annotation_file = Path(file).with_suffix(".txt")
                            test_path = os.path.join(annotation_dir, str(annotation_file))
                            if os.path.exists(test_path):
                                annotation_path = test_path
                            else:
                                for f in os.listdir(annotation_dir):
                                    if f.lower() == str(annotation_file).lower():
                                        annotation_path = os.path.join(annotation_dir, f)
                                        break
                                        
                        if annotation_path and os.path.exists(annotation_path):
                            try:
                                with open(annotation_path, "r") as af:
                                    lines = [line.strip() for line in af.readlines() if line.strip()]
                                    if lines:
                                        first_line_parts = lines[0].split()
                                        if first_line_parts:
                                            try:
                                                start_frame_val = int(first_line_parts[0])
                                                if start_frame_val > 0:
                                                    ground_truth = "FALLEN"
                                                else:
                                                    ground_truth = "NORMAL"
                                            except ValueError:
                                                nums = re.findall(r'\d+', first_line_parts[0])
                                                if nums and int(nums[0]) > 0:
                                                    ground_truth = "FALLEN"
                                                else:
                                                    ground_truth = "NORMAL"
                            except Exception:
                                pass
                        else:
                            is_fall = any("fall" in p or "chute" in p for p in relative_parts_lower) or "fall" in file_lower or "fallen" in file_lower or "chute" in file_lower
                            is_adl = any("adl" in p or "normal" in p or "sans" in p for p in relative_parts_lower)
                            if is_fall and not is_adl:
                                ground_truth = "FALLEN"
                            elif environment == "Lecture Room" and video_num is not None:
                                ground_truth = "FALLEN" if (10 <= video_num <= 22) else "NORMAL"
                            elif environment == "Office" and video_num is not None:
                                ground_truth = "FALLEN" if (11 <= video_num <= 22) else "NORMAL"
                                
                        if environment not in env_buckets:
                            env_buckets[environment] = []
                        env_buckets[environment].append((full_path, ground_truth, environment))
                        
            for env, items in env_buckets.items():
                if CONFIG["MAX_VIDEOS_PER_ENV"] is not None and len(items) > CONFIG["MAX_VIDEOS_PER_ENV"]:
                    print(f" -> Env '{env}' has {len(items)} clips. Sampling exactly {CONFIG['MAX_VIDEOS_PER_ENV']} randomized clips...")
                    sampled_items = random.sample(items, CONFIG["MAX_VIDEOS_PER_ENV"])
                    video_files.extend(sampled_items)
                else:
                    video_files.extend(items)
                    
        if not video_files:
            print("\n[WARNING] Discovered dataset folder is empty or has no supported video files.")
            print("          Falling back to configured test videos...")
            for vid in CONFIG.get("TEST_VIDEOS", ["test_fall.mp4"]):
                if os.path.exists(vid):
                    video_files.append((vid, "UNKNOWN", "Unknown_Environment"))
                    
        return video_files, dataset_path

# ===================== BENCHMARK RUNNER =====================
class BenchmarkRunner:
    @staticmethod
    def run():
        print("=========================================================")
        print("   NEXUS FALLNET AI - REDESIGNED FALL BENCHMARK PIPELINE")
        print(f"   Running on local hardware accelerator: {DEVICE}")
        print("=========================================================")
        
        # ── Phase 0: Load Videos ────────────────────────────────────
        import sys
        video_dataset = []  # list of (path, environment)
        VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
        seen_paths = set()  # avoid duplicates
        
        if len(sys.argv) > 1:
            # CLI mode: use command-line arguments
            for v_in in sys.argv[1:]:
                abs_path = os.path.abspath(v_in)
                if abs_path not in seen_paths and os.path.exists(v_in):
                    video_dataset.append((v_in, "CLI_Inputs"))
                    seen_paths.add(abs_path)
                    print(f"  [CLI] Added '{os.path.basename(v_in)}'")
                elif abs_path not in seen_paths:
                    print(f"[WARNING] Video file '{v_in}' not found! Skipping.")
        else:
            # 1. Auto-scan VIDEO_FOLDERS
            for folder in CONFIG.get("VIDEO_FOLDERS", []):
                if os.path.isdir(folder):
                    folder_name = os.path.basename(os.path.abspath(folder))
                    folder_videos = sorted([
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
                    ])
                    for vpath in folder_videos:
                        abs_path = os.path.abspath(vpath)
                        if abs_path not in seen_paths:
                            video_dataset.append((vpath, folder_name))
                            seen_paths.add(abs_path)
                    print(f"  [FOLDER] Scanned '{folder}' → {len(folder_videos)} videos found (env: {folder_name})")
                else:
                    print(f"[WARNING] Video folder '{folder}' not found! Skipping.")
            
            # 2. Add additional TEST_VIDEOS (loose files)
            test_videos = CONFIG.get("TEST_VIDEOS", [])
            added_test = 0
            for vid in test_videos:
                abs_path = os.path.abspath(vid)
                if abs_path in seen_paths:
                    continue  # already added from a folder scan
                if os.path.exists(vid):
                    video_dataset.append((vid, "Test_Videos"))
                    seen_paths.add(abs_path)
                    added_test += 1
                else:
                    print(f"[WARNING] Test video '{vid}' not found! Skipping.")
            if added_test > 0:
                print(f"  [CONFIG] Added {added_test} additional test videos (env: Test_Videos)")
                
        if not video_dataset:
            print("[ERROR] No valid videos provided to process.")
            return
            
        dataset_src_path = "Multiple Video Inputs" if len(video_dataset) > 1 else "Single Video Input"
        CONFIG["SHOW_LIVE_PREVIEW"] = True
        
        if len(video_dataset) > 1:
            print(f"\n[INFO] Loaded {len(video_dataset)} videos for evaluation.")
        else:
            print(f"\n[INFO] Loaded single video: {video_dataset[0][0]}")
        
        report_path = "benchmark_report.txt"
        
        # Reset the executive report file
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 18 + "EXECUTIVE BENCHMARK REPORT: FALL DETECTOR ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Source : {dataset_src_path}\n")
            f.write(f"Total Videos   : {len(video_dataset)}\n")
            f.write(f"Hardware Accel : {DEVICE}\n")
            f.write(f"Ground Truth   : Auto-consensus (cross-model majority vote)\n")
            f.write("=" * 80 + "\n\n")
            
        # ════════════════════════════════════════════════════════════
        # PHASE 1: DETECTION — Run all models on all videos
        # ════════════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print(f"{CLR_BOLD}{CLR_CYAN}   PHASE 1: DETECTION — Running all models on all videos{CLR_RESET}")
        print("=" * 80)
        
        all_detections = {}  # model_name -> list of per-video detection dicts
        
        for model_filename in CONFIG["POSE_MODELS"]:
            if model_filename == "yolo11n-pose_stgcn":
                model_name = "yolo11n-pose_stgcn"
                yolo_weights = "yolo11n-pose.pt"
            else:
                model_name = model_filename.replace(".pt", "")
                yolo_weights = model_filename
                
            print(f"\n" + "="*60)
            print(f" DETECTING WITH MODEL: {model_name.upper()}")
            print("="*60)
            
            try:
                model = YOLO(yolo_weights)
            except Exception as e:
                print(f"[ERROR] Failed to load {yolo_weights}: {e}. Skipping.")
                continue
                
            model_detections = []
            
            for idx, (video_path, environment) in enumerate(video_dataset):
                video_name = os.path.basename(video_path)
                temp_mp4_path = None
                cap = cv2.VideoCapture(video_path)
                
                # macOS AVI Decoder Fallback
                if not cap.isOpened() and video_path.lower().endswith('.avi'):
                    if shutil.which("ffmpeg") is not None:
                        temp_mp4_path = video_path.replace(".avi", "_temp_transcoded.mp4")
                        cmd = [
                            "ffmpeg", "-y", "-i", video_path, 
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", 
                            "-an", temp_mp4_path
                        ]
                        try:
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            cap = cv2.VideoCapture(temp_mp4_path)
                        except Exception:
                            pass
                            
                if not cap.isOpened():
                    print(f" -> [{idx+1}/{len(video_dataset)}] [SKIP] Failed to load {video_name}.")
                    if cap: cap.release()
                    model_detections.append({
                        "video_path": video_path, "video_name": video_name,
                        "environment": environment, "prediction": "SKIPPED",
                        "fall_frame_idx": None, "fall_timestamp": None,
                        "inference_time": 0.0, "frames_count": 0,
                        "fall_state_frames": 0, "max_fall_confidence": 0.0,
                        "verdict_msg": "SKIPPED", "reason": "Failed to open video",
                    })
                    continue
                    
                print(f" 🎬 [{idx+1}/{len(video_dataset)}] Scanning: {video_name} | Env: {environment}")
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps == 0: fps = 25
                
                out_writer = None
                if CONFIG["SAVE_ANNOTATED_VIDEOS"]:
                    os.makedirs("benchmark_outputs", exist_ok=True)
                    output_filename = f"benchmark_outputs/{model_name}_{environment}_{video_name}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                    
                # Active tracks for multiple people
                active_tracks = {}
                video_has_confirmed_fall = False
                fall_frame_idx = None
                fall_timestamp = None
                video_inference_time = 0.0
                video_frames_count = 0
                fall_state_frame_count = 0  # Count frames with any fall-related activity
                early_stop = False
                
                if CONFIG["SHOW_LIVE_PREVIEW"]:
                    cv2.namedWindow("Nexus Redesigned Fall Benchmark", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Nexus Redesigned Fall Benchmark", 960, 540)
                    
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    video_frames_count += 1
                    frame_has_fall_activity = False
                    
                    # Inference step
                    start_time = time.perf_counter()
                    results = model.track(frame, persist=True, imgsz=640, verbose=False, device=DEVICE)
                    end_time = time.perf_counter()
                    video_inference_time += (end_time - start_time)
                    
                    # Mark all active tracks as unseen in this frame
                    for track in active_tracks.values():
                        track.unseen_count += 1
                        
                    for r in results:
                        if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                            boxes = r.boxes.xywh.cpu().numpy()
                            keypoints = r.keypoints.xy.cpu().numpy()
                            keypoint_confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                            track_ids = r.boxes.id.int().cpu().tolist()
                            
                            for i in range(len(boxes)):
                                cx, cy, w, h = boxes[i]
                                person_kpts = keypoints[i]
                                track_id = track_ids[i]
                                
                                # Box filter
                                if h < CONFIG["MIN_PERSON_HEIGHT"] or w * h < CONFIG["MIN_BOX_AREA"]:
                                    continue
                                    
                                # Pose confidence filter
                                if keypoint_confs is not None:
                                    avg_conf = np.mean(keypoint_confs[i])
                                    if avg_conf < CONFIG["KEYPOINT_CONF_THRESHOLD"]:
                                        continue
                                        
                                # Fetch or instantiate track
                                if track_id not in active_tracks:
                                    active_tracks[track_id] = PersonTrack(track_id)
                                    
                                track = active_tracks[track_id]
                                track.unseen_count = 0
                                
                                # Store raw frames and updates
                                track.bbox_history.append([cx, cy, w, h])
                                track.raw_keypoints_history.append(person_kpts)
                                
                                # Extract per-keypoint confidences for CA-EMA
                                person_kpt_confs = keypoint_confs[i] if keypoint_confs is not None else None
                                
                                # Run Fall Detection state update & feature scoring
                                FallDetector.detect(track, width, height, keypoint_confs=person_kpt_confs, fps=fps)
                                
                                # Track fall activity for this frame
                                if track.current_state == FallStateMachine.CONFIRMED:
                                    video_has_confirmed_fall = True
                                    frame_has_fall_activity = True
                                    if fall_frame_idx is None:
                                        fall_frame_idx = video_frames_count
                                        fall_timestamp = video_frames_count / (fps + 1e-6)
                                elif track.current_state in [FallStateMachine.LYING, FallStateMachine.IMPACT]:
                                    frame_has_fall_activity = True
                                    
                                # Render visual overlays
                                if CONFIG["SHOW_LIVE_PREVIEW"] or CONFIG["SAVE_ANNOTATED_VIDEOS"]:
                                    # Build immobility info string for overlay
                                    imm_info = ""
                                    if track.current_state == FallStateMachine.CONFIRMED:
                                        imm_info = f"Ground: {track.immobility_seconds:.1f}s ({track.immobility_status})"
                                    
                                    draw_person_overlay(
                                        frame,
                                        [cx, cy, w, h],
                                        track.keypoints_history[-1],
                                        track_id,
                                        track.current_state,
                                        track.fall_confidence,
                                        track.last_vy,
                                        track.last_torso_angle,
                                        track.state_machine.consecutive_lying_frames,
                                        near_fall_count=track.state_machine.near_fall_count,
                                        severity=track.fall_severity,
                                        immobility_info=imm_info
                                    )
                    
                    # Count fall-activity frames
                    if frame_has_fall_activity:
                        fall_state_frame_count += 1
                                    
                    # Remove stale/unseen tracks
                    for tid in list(active_tracks.keys()):
                        if active_tracks[tid].unseen_count > 60:
                            del active_tracks[tid]
                            
                    if CONFIG["SHOW_LIVE_PREVIEW"]:
                        cv2.imshow("Nexus Redesigned Fall Benchmark", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            early_stop = True
                            break
                            
                    if out_writer:
                        out_writer.write(frame)
                        
                cap.release()
                if out_writer:
                    out_writer.release()
                    
                if temp_mp4_path and os.path.exists(temp_mp4_path):
                    try: os.remove(temp_mp4_path)
                    except Exception: pass
                    
                # Verdict
                prediction = "FALLEN" if video_has_confirmed_fall else "NORMAL"
                
                # Collect USP metrics across all tracks
                total_near_falls = sum(t.state_machine.near_fall_count for t in active_tracks.values())
                fall_severity = "N/A"
                fall_severity_score = 0.0
                immobility_seconds = 0.0
                immobility_status = "N/A"
                
                # Collect reason and max confidence
                reasons_str = "No fall detected"
                max_fall_conf = 0.0
                if video_has_confirmed_fall:
                    for track in active_tracks.values():
                        if track.current_state == FallStateMachine.CONFIRMED or track.fall_severity != "N/A":
                            reasons_str = track.reason
                            max_fall_conf = max(max_fall_conf, track.fall_confidence)
                            fall_severity = track.fall_severity
                            fall_severity_score = track.fall_severity_score
                            immobility_seconds = track.immobility_seconds
                            immobility_status = track.immobility_status
                            break
                else:
                    if active_tracks:
                        max_conf_track = max(active_tracks.values(), key=lambda t: t.fall_confidence)
                        max_fall_conf = max_conf_track.fall_confidence
                        reasons_str = f"Max Conf: {max_fall_conf*100:.1f}% ({max_conf_track.reason})"
                
                # Build verdict message
                if prediction == "FALLEN":
                    verdict_msg = f"FALL DETECTED at frame {fall_frame_idx} ({fall_timestamp:.2f}s) | Severity: {fall_severity}"
                    det_lbl = f"{CLR_RED}{verdict_msg}{CLR_RESET}"
                else:
                    verdict_msg = "No fall detected"
                    det_lbl = f"{CLR_GREEN}{verdict_msg}{CLR_RESET}"
                
                # USP summary line
                usp_summary = ""
                if total_near_falls > 0:
                    usp_summary += f" | ⚡ Near-Falls: {total_near_falls}"
                if immobility_seconds > 0:
                    usp_summary += f" | Ground Time: {immobility_seconds:.1f}s ({immobility_status})"
                    
                fall_pct = (fall_state_frame_count / (video_frames_count + 1e-6)) * 100
                print(f"    └─ Detection: {det_lbl} | Fall Frames: {fall_state_frame_count}/{video_frames_count} ({fall_pct:.1f}%){usp_summary}")
                print(f"       {reasons_str}\n")
                
                model_detections.append({
                    "video_path": video_path,
                    "video_name": video_name,
                    "environment": environment,
                    "prediction": prediction,
                    "fall_frame_idx": fall_frame_idx,
                    "fall_timestamp": fall_timestamp,
                    "inference_time": video_inference_time,
                    "frames_count": video_frames_count,
                    "fall_state_frames": fall_state_frame_count,
                    "max_fall_confidence": max_fall_conf,
                    "verdict_msg": verdict_msg,
                    "reason": reasons_str,
                    # USP fields
                    "near_fall_count": total_near_falls,
                    "fall_severity": fall_severity,
                    "fall_severity_score": fall_severity_score,
                    "immobility_seconds": immobility_seconds,
                    "immobility_status": immobility_status,
                })
                
                if early_stop:
                    break
                    
            all_detections[model_name] = model_detections
            
            # Print intermediate detection summary
            total_inf = sum(d["inference_time"] for d in model_detections)
            total_fr = sum(d["frames_count"] for d in model_detections)
            model_fps = total_fr / (total_inf + 1e-6)
            falls_found = sum(1 for d in model_detections if d["prediction"] == "FALLEN")
            print(f"\n  📊 {model_name.upper()} Summary: Detected falls in {falls_found}/{len(model_detections)} videos | Avg Speed: {model_fps:.1f} FPS")
            
        if CONFIG["SHOW_LIVE_PREVIEW"]:
            cv2.destroyAllWindows()
            for _ in range(4):
                cv2.waitKey(1)
        
        # ════════════════════════════════════════════════════════════
        # PHASE 2: CONSENSUS GROUND TRUTH (Majority Vote)
        # ════════════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print(f"{CLR_BOLD}{CLR_CYAN}   PHASE 2: AUTO GROUND TRUTH — Cross-Model Consensus (Majority Vote){CLR_RESET}")
        print("=" * 80)
        
        num_videos = len(video_dataset)
        num_models = len(all_detections)
        model_names_ordered = list(all_detections.keys())
        consensus_ground_truth = []
        
        # Print detection matrix header
        header = f"\n  {'Video':<25} |"
        for mn in model_names_ordered:
            header += f" {mn:<18} |"
        header += f" {'CONSENSUS':<10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        for vid_idx in range(num_videos):
            video_name = os.path.basename(video_dataset[vid_idx][0])
            fall_votes = 0
            
            row = f"  {video_name:<25} |"
            
            for mn in model_names_ordered:
                detections = all_detections[mn]
                if vid_idx < len(detections):
                    det = detections[vid_idx]
                    pred = det["prediction"]
                    ff = det["fall_state_frames"]
                    
                    if pred == "FALLEN":
                        fall_votes += 1
                        cell = f"FALL ({ff}f)"
                        row += f" {CLR_RED}{cell:<18}{CLR_RESET} |"
                    elif pred == "SKIPPED":
                        cell = "SKIP"
                        row += f" {CLR_YELLOW}{cell:<18}{CLR_RESET} |"
                    else:
                        cell = f"NORMAL ({ff}f)"
                        row += f" {CLR_GREEN}{cell:<18}{CLR_RESET} |"
                else:
                    row += f" {'N/A':<18} |"
            
            # Majority vote: more than half must agree
            if fall_votes > num_models / 2:
                consensus = "FALLEN"
                row += f" {CLR_RED}{consensus}{CLR_RESET}"
            else:
                consensus = "NORMAL"
                row += f" {CLR_GREEN}{consensus}{CLR_RESET}"
                
            consensus_ground_truth.append(consensus)
            print(row)
        
        total_fallen = sum(1 for c in consensus_ground_truth if c == "FALLEN")
        total_normal = sum(1 for c in consensus_ground_truth if c == "NORMAL")
        print(f"\n  Consensus Result: {CLR_RED}{total_fallen} FALL{CLR_RESET} videos + {CLR_GREEN}{total_normal} NORMAL{CLR_RESET} videos")
        print(f"  (Ground truth determined by {num_models}-model majority vote)")
        
        # ════════════════════════════════════════════════════════════
        # PHASE 3: BENCHMARK METRICS (vs Consensus)
        # ════════════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print(f"{CLR_BOLD}{CLR_CYAN}   PHASE 3: BENCHMARK METRICS (Each Model vs Consensus Ground Truth){CLR_RESET}")
        print("=" * 80)
        
        benchmark_results = {}
        
        for model_name in model_names_ordered:
            detections = all_detections[model_name]
            tracker = PerformanceTracker(model_name)
            
            for vid_idx, det in enumerate(detections):
                if det["prediction"] == "SKIPPED":
                    continue
                gt = consensus_ground_truth[vid_idx]
                tracker.update_metrics(det["prediction"], gt, det["environment"])
                tracker.add_latency(det["inference_time"], det["frames_count"], det["environment"])
                tracker.video_details.append({
                    "video_name": det["video_name"],
                    "ground_truth": gt,
                    "prediction": det["prediction"],
                    "verdict_msg": det["verdict_msg"],
                    "environment": det["environment"],
                })
            
            summary = tracker.get_results_summary()
            benchmark_results[model_name] = summary
            
            # Print per-model metrics card
            g_res = summary["GLOBAL"]
            tp, tn, fp, fn = g_res['matrix']
            print(f"\n  {'─'*60}")
            print(f"  {CLR_BOLD}{model_name.upper()}{CLR_RESET}")
            print(f"  Recall: {g_res['recall']:.2f}% | Precision: {g_res['precision']:.2f}% | F1: {CLR_GREEN}{g_res['f1']:.2f}%{CLR_RESET} | Speed: {g_res['fps']:.1f} FPS")
            print(f"  Confusion Matrix: [ TP(Correct Falls): {tp} | TN(Correct Normals): {tn} | FP(False Alarms): {fp} | FN(Missed Falls): {fn} ]")
            
            # Per-video verdict details
            for vid_idx, det in enumerate(detections):
                if det["prediction"] == "SKIPPED":
                    continue
                gt = consensus_ground_truth[vid_idx]
                if det["prediction"] == gt:
                    status = f"{CLR_GREEN}✅ CORRECT{CLR_RESET}"
                elif gt == "FALLEN":
                    status = f"{CLR_RED}❌ MISSED{CLR_RESET}"
                else:
                    status = f"{CLR_YELLOW}❌ FALSE ALARM{CLR_RESET}"
                gt_lbl = f"{CLR_CYAN}FALL{CLR_RESET}" if gt == "FALLEN" else "NORMAL"
                print(f"    {det['video_name']:<25} | {status} | GT: {gt_lbl} | {det['verdict_msg']}")
            
            # Write per-model report file
            model_report_path = f"benchmark_report_{model_name}.txt"
            with open(model_report_path, "w") as f_model:
                f_model.write("=" * 80 + "\n")
                f_model.write(" " * 12 + f"BENCHMARK REPORT: {model_name.upper()} PERFORMANCE\n")
                f_model.write("=" * 80 + "\n")
                f_model.write(f"Generated on   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_model.write(f"Model Name     : {model_name.upper()}\n")
                f_model.write(f"Dataset Source : {dataset_src_path}\n")
                f_model.write(f"Total Videos   : {len(tracker.video_details)}\n")
                f_model.write(f"Hardware Accel : {DEVICE}\n")
                f_model.write(f"Ground Truth   : Auto-consensus ({num_models}-model majority vote)\n")
                f_model.write("=" * 80 + "\n\n")
                
                f_model.write("1. PERFORMANCE METRICS (ENVIRONMENT BREAKDOWN)\n")
                f_model.write("=" * 80 + "\n")
                f_model.write(f"{'Environment':<18} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<9} | {'Speed (FPS)':<11}\n")
                f_model.write("-" * 80 + "\n")
                f_model.write(f"{'GLOBAL':<18} | {g_res['recall']:>6.2f}% | {g_res['precision']:>8.2f}% | {g_res['f1']:>7.2f}% | {g_res['fps']:>9.1f} FPS\n")
                for env_name, env_res in summary.items():
                    if env_name == "GLOBAL": continue
                    f_model.write(f"{env_name:<18} | {env_res['recall']:>6.2f}% | {env_res['precision']:>8.2f}% | {env_res['f1']:>7.2f}% | {env_res['fps']:>9.1f} FPS\n")
                f_model.write("-" * 80 + "\n\n")
                
                f_model.write("2. CONFUSION MATRIX\n")
                f_model.write("-" * 40 + "\n")
                for env_name, env_res in summary.items():
                    tp2, tn2, fp2, fn2 = env_res['matrix']
                    f_model.write(f" -> {env_name:<18} : [ TP: {tp2:<3} | TN: {tn2:<3} | FP: {fp2:<3} | FN: {fn2:<3} ]\n")
                
                f_model.write("\n3. DETAILED VIDEO-LEVEL VERDICTS\n")
                f_model.write("-" * 100 + "\n")
                for vid_idx, det in enumerate(detections):
                    if det["prediction"] == "SKIPPED":
                        f_model.write(f"⊘ {det['video_name']:<25} | SKIPPED\n")
                        continue
                    gt = consensus_ground_truth[vid_idx]
                    match = "✅" if det["prediction"] == gt else "❌"
                    f_model.write(f"{match} {det['video_name']:<25} | Detected: {det['prediction']:<7} | Consensus GT: {gt:<7} | {det['verdict_msg']}\n")
                    f_model.write(f"   Fall Frames: {det['fall_state_frames']}/{det['frames_count']} | Max Conf: {det['max_fall_confidence']*100:.1f}% | Reason: {det['reason']}\n")
                
                f_model.write("\n" + "=" * 80 + "\n")
                f_model.write(" " * 24 + "END OF MODEL EVALUATION LOGS\n")
                f_model.write("=" * 80 + "\n")
            
            print(f"  Report: '{model_report_path}'")
        
        # ── Final Scoreboard ────────────────────────────────────────
        best_accuracy_model = None
        best_accuracy_score = -1.0
        best_speed_model = None
        best_speed_score = -1.0
        best_balanced_model = None
        best_balanced_score = -1.0
        
        for model_name, env_results in benchmark_results.items():
            g_res = env_results["GLOBAL"]
            f1 = g_res["f1"]
            fps_val = g_res["fps"]
            
            if f1 > best_accuracy_score:
                best_accuracy_score = f1
                best_accuracy_model = model_name
                
            if fps_val > best_speed_score:
                best_speed_score = fps_val
                best_speed_model = model_name
                
            norm_fps_score = min(100.0, (fps_val / 120.0) * 100.0)
            balanced_score = (f1 * 0.70) + (norm_fps_score * 0.30)
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_balanced_model = model_name
        
        # ── Write Final Consolidated Report ─────────────────────────
        with open(report_path, "a") as f:
            # Detection Matrix
            f.write("=" * 100 + "\n")
            f.write("                    DETECTION MATRIX (All Models × All Videos)\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Video':<25} |")
            for mn in model_names_ordered:
                f.write(f" {mn:<18} |")
            f.write(f" {'CONSENSUS':<10}\n")
            f.write("-" * 100 + "\n")
            
            for vid_idx in range(num_videos):
                video_name = os.path.basename(video_dataset[vid_idx][0])
                f.write(f"{video_name:<25} |")
                for mn in model_names_ordered:
                    detections = all_detections[mn]
                    if vid_idx < len(detections):
                        det = detections[vid_idx]
                        ff = det["fall_state_frames"]
                        cell = f"{det['prediction']} ({ff}f)"
                        f.write(f" {cell:<18} |")
                    else:
                        f.write(f" {'N/A':<18} |")
                f.write(f" {consensus_ground_truth[vid_idx]}\n")
            f.write("-" * 100 + "\n\n")
            
            # Final metrics table
            f.write("=" * 100 + "\n")
            f.write("                     FINAL PERFORMANCE METRICS (vs Consensus Ground Truth)\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Model Name':<18} | {'Environment':<18} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<9} | {'Speed (FPS)':<11}\n")
            f.write("-" * 100 + "\n")
            for model_name, env_results in benchmark_results.items():
                g_res = env_results["GLOBAL"]
                f.write(f"{model_name:<18} | {'GLOBAL':<18} | {g_res['recall']:>6.2f}% | {g_res['precision']:>8.2f}% | {g_res['f1']:>7.2f}% | {g_res['fps']:>9.1f} FPS\n")
                for env_name, env_res in env_results.items():
                    if env_name == "GLOBAL": continue
                    f.write(f"{'':<18} | {env_name:<18} | {env_res['recall']:>6.2f}% | {env_res['precision']:>8.2f}% | {env_res['f1']:>7.2f}% | {env_res['fps']:>9.1f} FPS\n")
                f.write("-" * 100 + "\n")
            
            # Confusion matrices
            f.write("\nConfusion Matrices (vs Consensus Ground Truth):\n")
            for model_name, env_results in benchmark_results.items():
                f.write(f"\nModel: {model_name.upper()}\n")
                for env_name, env_res in env_results.items():
                    tp2, tn2, fp2, fn2 = env_res['matrix']
                    f.write(f" -> {env_name:<18} : [ TP: {tp2:<3} | TN: {tn2:<3} | FP: {fp2:<3} | FN: {fn2:<3} ]\n")
            
            # Scoreboard
            f.write("\n" + "=" * 80 + "\n")
            f.write(" " * 20 + "FINAL MULTI-MODEL COMPETITIVE SCOREBOARD\n")
            f.write("=" * 80 + "\n")
            f.write(f"🏆 ACCURACY CHAMPION  : {best_accuracy_model.upper() if best_accuracy_model else 'N/A'} ({best_accuracy_score:.2f}% F1-Score)\n")
            f.write(f"⚡ SPEED CHAMPION     : {best_speed_model.upper() if best_speed_model else 'N/A'} ({best_speed_score:.1f} FPS)\n")
            f.write(f"⚖️ BALANCED CHAMPION  : {best_balanced_model.upper() if best_balanced_model else 'N/A'} (Best balanced for edge safety)\n")
            f.write("=" * 80 + "\n")
        
        # ── Print Final Console Scoreboard ──────────────────────────
        print("\n" + "="*100)
        print(f"{CLR_BOLD}{CLR_CYAN}                     FINAL BENCHMARK RESULTS (vs Consensus Ground Truth){CLR_RESET}")
        print("="*100)
        print(f"  {'Model Name':<18} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<9} | {'Speed (FPS)':<11}")
        print("  " + "-"*70)
        for model_name, env_results in benchmark_results.items():
            g_res = env_results["GLOBAL"]
            print(f"  {CLR_BOLD}{model_name:<18}{CLR_RESET} | {g_res['recall']:>6.2f}% | {g_res['precision']:>8.2f}% | {CLR_GREEN}{g_res['f1']:>7.2f}%{CLR_RESET} | {g_res['fps']:>9.1f} FPS")
        print("  " + "-"*70)
            
        print(f"\n{'#' * 80}")
        print(f"🥇 {CLR_BOLD}ACCURACY CHAMPION{CLR_RESET}  : {CLR_GREEN}{best_accuracy_model.upper() if best_accuracy_model else 'N/A'}{CLR_RESET} ({best_accuracy_score:.2f}% F1-Score)")
        print(f"⚡ {CLR_BOLD}SPEED CHAMPION{CLR_RESET}     : {CLR_YELLOW}{best_speed_model.upper() if best_speed_model else 'N/A'}{CLR_RESET} ({best_speed_score:.1f} FPS)")
        print(f"⚖️ {CLR_BOLD}BALANCED CHAMPION{CLR_RESET}  : {CLR_CYAN}{best_balanced_model.upper() if best_balanced_model else 'N/A'}{CLR_RESET} (Best for real-time safety)")
        print(f"{'#' * 80}")
        print(f"\n[INFO] Comprehensive report written to: '{report_path}'")
        print("#" * 80 + "\n")

        # ── Graphical Confusion Matrix ──────────────────────────────
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import seaborn as sns  # type: ignore
            
            num_models_plot = len(benchmark_results)
            if num_models_plot > 0:
                fig, axes = plt.subplots(1, num_models_plot, figsize=(5 * num_models_plot, 5))
                if num_models_plot == 1:
                    axes = [axes]
                    
                for idx, (model_name, env_results) in enumerate(benchmark_results.items()):
                    tp2, tn2, fp2, fn2 = env_results["GLOBAL"]['matrix']
                    
                    matrix = np.array([[tn2, fp2],
                                       [fn2, tp2]])
                    
                    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                                xticklabels=["Normal", "Fall"],
                                yticklabels=["Normal", "Fall"],
                                cbar=False, annot_kws={"size": 16})
                    axes[idx].set_title(f"{model_name.upper()} (Global)", fontsize=14)
                    axes[idx].set_xlabel("Predicted", fontsize=12)
                    axes[idx].set_ylabel("Consensus GT", fontsize=12)
                    
                plt.suptitle("Confusion Matrices (vs Consensus Ground Truth)", fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                plot_path = "confusion_matrices.png"
                plt.savefig(plot_path)
                print(f"[INFO] Graphical confusion matrices saved to: '{plot_path}'")
                
                plt.show(block=False)
                plt.pause(3)
        except ImportError:
            print("[INFO] 'matplotlib' or 'seaborn' not installed. Skipping graphical confusion matrix.")
            print("       Install them via: pip install matplotlib seaborn")

if __name__ == "__main__":
    BenchmarkRunner.run()