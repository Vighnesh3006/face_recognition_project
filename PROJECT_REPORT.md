# GestureOS — Face-Authenticated Real-Time Gesture Control System
### Project Report

---

> **Authors:** Vighnesh Ghorpade, Aryan, Atharv
> **Platform:** Windows 10 / 11 · Python 3.10
> **Entry Point:** `python main.py`

---

# Chapter 1: Introduction

## 1.1 Background

Human-Computer Interaction (HCI) has evolved dramatically over the past two decades — from keyboards and mice, to touchscreens, to voice assistants. The next frontier is **gesture-based control**: the ability to operate a computer using natural hand movements captured by a standard webcam, with no physical contact required.

Gesture control is not a new concept. Research laboratories and high-end commercial products have explored it for years. However, most implementations either require expensive depth-sensing hardware (Microsoft Kinect, Intel RealSense), are limited to a single pre-defined user, or lack the security layer needed for real-world deployment. A system that works on commodity hardware, recognises multiple users, and enforces identity verification before accepting any command has remained largely absent from the open-source ecosystem.

## 1.2 What This Project Is

**GestureOS** is a Python-based desktop application that allows a registered, face-authenticated user to control their Windows PC entirely through hand gestures captured by a standard USB or built-in webcam. No special hardware is required beyond the camera already present in most laptops.

The system supports 19 distinct gestures mapped to system-level actions including media playback, volume control, screen brightness, navigation, scrolling, screen recording, window management, and — with appropriate safeguards — system shutdown.

## 1.3 Why It Matters

- **Accessibility:** Users with limited mobility who cannot comfortably use a keyboard or mouse can interact with their PC through natural hand movements.
- **Hygiene:** In medical, laboratory, or food-preparation environments, touchless control eliminates contamination risk.
- **Convenience:** Controlling media playback or volume from across the room without reaching for a device.
- **Security:** Every gesture command is gated behind continuous face authentication — an unrecognised face cannot trigger any action.

## 1.4 Document Structure

This report is organised into 13 chapters covering the project's motivation, technical design, implementation details, testing methodology, results, and future directions.

---

# Chapter 2: Objective and Scope

## 2.1 Primary Objective

To design and implement a real-time, face-authenticated, gesture-based PC control system that:

1. Detects and recognises hand gestures from a live webcam feed using computer vision and machine learning.
2. Maps recognised gestures to meaningful Windows system actions.
3. Restricts all gesture commands to registered, face-verified users only.
4. Provides a polished graphical interface for user management, gesture configuration, and session monitoring.
5. Runs entirely on commodity hardware (CPU-only, standard webcam) without any cloud dependency.

## 2.2 Specific Goals

| Goal | Metric |
|---|---|
| Gesture recognition accuracy | ≥ 90% on trained gestures |
| Face authentication accuracy | ≥ 95% under normal lighting |
| System latency (gesture → action) | < 500 ms |
| Supported gestures | 19 distinct gestures |
| Supported users | Unlimited (multi-user) |
| Hardware requirement | Standard webcam, CPU only |

## 2.3 Scope

**In scope:**
- Real-time hand gesture detection using MediaPipe
- Per-user SVM gesture classifier with rule-based fallback
- Multi-angle face registration and continuous re-authentication
- 19 gesture-to-action mappings (fully configurable)
- GUI for user management, gesture mapping, settings, and logs
- Screen recording triggered by gesture
- Safety mechanisms for destructive actions (shutdown/restart)
- Activity logging with timestamps

**Out of scope:**
- Mobile or web deployment
- Voice commands
- Mouse cursor control via hand position
- Cross-platform support (Linux/macOS)
- Cloud-based face recognition

---

# Chapter 3: Problem Statement

## 3.1 The Core Problem

Standard PC input devices — keyboard and mouse — require physical contact, fine motor control, and proximity to the device. This creates barriers for:

- Users with motor disabilities or repetitive strain injuries
- Scenarios requiring touchless interaction (medical, industrial, food-prep)
- Situations where the user is physically away from the desk (presentations, media control)

Existing gesture control solutions either require expensive specialised hardware, work only for a single pre-configured user, or lack any identity verification — meaning anyone who walks in front of the camera can control the PC.

## 3.2 Specific Technical Challenges

**Challenge 1 — Accurate finger counting across hand orientations**
Simple y-coordinate comparisons (tip.y < mcp.y) fail when the hand is tilted, rotated, or held at an angle. A scale-invariant, orientation-robust counting method is needed.

**Challenge 2 — Distinguishing similar gestures**
Fist vs. 4-fingers, peace sign vs. 2-fingers, and OK sign vs. other shapes are visually similar. Pure rule-based detection produces frequent misclassifications.

**Challenge 3 — Preventing accidental action triggers**
A momentary flash of 10 fingers should not shut down the PC. Destructive actions need deliberate, sustained confirmation.

**Challenge 4 — Continuous identity verification without blocking gesture detection**
Face recognition is computationally expensive (~200–500 ms per frame). Running it synchronously would drop the gesture detection frame rate to unusable levels.

**Challenge 5 — Lighting and distance variability**
HOG-based face detection and MediaPipe hand detection both degrade under poor lighting. A preprocessing step is needed to normalise lighting conditions.

**Challenge 6 — Multi-user support with per-user models**
Different users have different hand sizes, skin tones, and gesture styles. A single global model performs poorly across users.

## 3.3 Problem Statement (Formal)

*Design and implement a CPU-only, webcam-based gesture control system for Windows that: (a) accurately counts fingers and classifies hand gestures across varied orientations and lighting conditions; (b) maps gestures to system actions with configurable hold-to-confirm safety; (c) restricts all commands to face-authenticated registered users via non-blocking continuous re-authentication; and (d) provides a complete graphical interface for multi-user management and configuration.*

---

# Chapter 4: Literature Survey

## 4.1 Hand Gesture Recognition — Overview

Hand gesture recognition has been studied extensively in computer vision. Approaches broadly fall into three categories:

### 4.1.1 Depth-Sensor Based Methods
Systems like Microsoft Kinect and Intel RealSense use structured light or time-of-flight sensors to capture depth maps. These provide highly accurate 3D hand models but require specialised hardware costing $100–$500+. Works by Keskin et al. (2012) and Tang et al. (2014) demonstrated real-time hand pose estimation using depth data, achieving >95% accuracy. Not applicable to commodity webcam setups.

### 4.1.2 Colour/Skin Segmentation Based Methods
Early webcam-based systems (Bradski, 1998 — CAMShift; Jones & Rehg, 1999 — skin colour histograms) segmented the hand by skin colour in HSV space. These methods are fast but highly sensitive to lighting changes and fail for users with darker skin tones or when the background contains skin-coloured objects.

### 4.1.3 Landmark-Based Methods (Current Approach)
MediaPipe Hands (Zhang et al., 2020 — Google Research) uses a two-stage pipeline: a palm detector followed by a hand landmark model that predicts 21 3D keypoints per hand. This approach is lighting-robust, works on CPU in real time, and is skin-tone agnostic. It forms the foundation of this project.

## 4.2 Face Recognition

**Eigenfaces (Turk & Pentland, 1991)** — PCA-based approach, fast but sensitive to lighting and pose.

**Fisherfaces (Belhumeur et al., 1997)** — LDA-based, better class separation but still limited by pose variation.

**Local Binary Patterns (Ahonen et al., 2006)** — Texture-based, more robust to lighting but still struggles with large pose changes.

**Deep Learning (DeepFace, FaceNet, ArcFace)** — CNN-based embeddings achieve near-human accuracy but require GPU and large training datasets.

**dlib HOG + ResNet (King, 2009–2017)** — The `face_recognition` library used in this project wraps dlib's HOG face detector and a ResNet-based 128-dimensional face embedding model. It achieves 99.38% accuracy on the Labeled Faces in the Wild (LFW) benchmark and runs on CPU in ~200–400 ms per frame. This is the practical sweet spot for a CPU-only desktop application.

## 4.3 Machine Learning for Gesture Classification

**Support Vector Machines (Cortes & Vapnik, 1995)** — SVMs with RBF kernels are well-suited for high-dimensional, relatively small datasets (hundreds to thousands of samples). They generalise well and are fast at inference time. This project uses an SVM pipeline (StandardScaler → SVC with RBF kernel, C=10, gamma='scale') trained on 126-dimensional hand landmark feature vectors.

**Temporal Models (LSTM, TCN)** — For dynamic gestures (swipes, circles), temporal models capture motion over time. This project focuses on static gestures, making frame-level classification sufficient.

## 4.4 Related Systems

| System | Approach | Limitation vs. This Project |
|---|---|---|
| HandsFree (2019) | MediaPipe + rule-based | No face auth, single user |
| GestureRecognizer (TensorFlow) | CNN on image patches | Requires GPU, no system integration |
| Windows Speech Recognition | Voice, not gesture | Different modality |
| Leap Motion Controller | Depth sensor | Requires $80 hardware |
| AirSig | Accelerometer | Requires wearable device |

## 4.5 Key Insights from Literature

1. Landmark-based methods outperform colour segmentation for robustness.
2. CLAHE preprocessing significantly improves HOG detection under variable lighting (Pizer et al., 1987).
3. Per-user models consistently outperform global models for gesture recognition due to individual variation in hand size and gesture style.
4. Hold-to-confirm (dwell time) is the standard UX pattern for preventing accidental triggers in gesture interfaces (Wobbrock et al., 2009).

---

# Chapter 5: System Specification

## 5.1 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Processor | Intel Core i5 (8th gen) / AMD Ryzen 5 | Intel Core i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| Camera | 720p USB/built-in webcam | 1080p webcam |
| Storage | 500 MB free | 1 GB free |
| OS | Windows 10 (64-bit) | Windows 11 (64-bit) |
| GPU | Not required | Not required |

## 5.2 Software Requirements

| Software | Version | Purpose |
|---|---|---|
| Python | 3.10.x | Runtime |
| opencv-python | 4.8.1.78 | Camera capture, frame processing, UI overlay |
| mediapipe | 0.10.9 | Hand landmark detection (21 points per hand) |
| face-recognition | 1.3.0 | Face encoding, comparison, authentication |
| dlib | 19.24.2 | Backend for face_recognition (HOG + ResNet) |
| cmake | 3.27.7 | Build dependency for dlib |
| numpy | 1.24.3 | Numerical operations on landmark arrays |
| scikit-learn | 1.3.2 | SVM classifier, StandardScaler, cross-validation |
| pyautogui | 0.9.54 | Keyboard/hotkey simulation, scroll |
| Pillow | 10.1.0 | Image handling in GUI camera preview |
| screen-brightness-control | 0.22.0 | Monitor brightness adjustment |
| pywin32 | 305 | Windows API (scroll, window targeting) |
| tkinter | Built-in | GUI framework |

## 5.3 Performance Characteristics

| Metric | Value |
|---|---|
| Gesture detection frame rate | 25 FPS (gesture worker thread) |
| Camera capture frame rate | 30 FPS |
| Face re-authentication interval | Every 2 seconds (background thread) |
| Initial authentication timeout | 30 seconds |
| Face absence timeout | 4 seconds |
| Gesture hold-to-confirm | 0.4 seconds (standard), 5 seconds (shutdown) |
| Action cooldown | 1.0 second |
| Gesture stability window | 3 frames |
| ML model inference time | ~2–5 ms (SVM on CPU) |
| Face recognition time | ~200–400 ms (HOG + ResNet on CPU) |

## 5.4 Data Storage

| File | Format | Contents |
|---|---|---|
| `system_config.json` | JSON | User profiles, gesture mappings, settings |
| `faces/<user>_encoding.pkl` | Pickle | List of 128-dim face encoding numpy arrays |
| `faces/<user>_model.pkl` | Pickle | Trained SVM pipeline + class labels + metadata |
| `faces/<user>_gestures.csv` | CSV | Raw training data: label + 126 landmark features |
| `activity_log.txt` | Plain text | Timestamped event log |
| `recordings/*.mp4` | MP4/AVI | Screen recordings |

---

# Chapter 6: Block Diagram / Description

## 6.1 High-Level System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER                                        │
│                    (sits in front of webcam)                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  Physical input: face + hand gestures
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      WEBCAM (640×480 @ 30fps)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  Raw BGR frames
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                               │
│   • cv2.flip(frame, 1)  — mirror for natural interaction            │
│   • CLAHE on LAB L-channel — lighting normalisation                 │
│   • cv2.resize to 480×360 — reduce MediaPipe compute               │
└──────────┬────────────────────────────┬────────────────────────────┘
           │                            │
           ▼                            ▼
┌──────────────────────┐    ┌───────────────────────────────────────┐
│  FACE AUTHENTICATION │    │       HAND DETECTION                  │
│                      │    │                                       │
│  dlib HOG detector   │    │  MediaPipe Hands                      │
│  → 128-dim ResNet    │    │  model_complexity=1                   │
│    encoding          │    │  max_num_hands=2                      │
│  → Compare vs stored │    │  → 21 landmarks × (x,y,z) per hand   │
│    encodings         │    │                                       │
│  → Runs every 2s     │    │  Runs at 25 FPS in worker thread      │
│    in background     │    │                                       │
└──────────┬───────────┘    └──────────────┬────────────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐    ┌───────────────────────────────────────┐
│  AUTH DECISION       │    │    GESTURE DETECTION                  │
│                      │    │                                       │
│  ✅ Authorised       │    │  1. ML path: SVM classifier           │
│  ❌ Unauthorised     │    │     126-feature vector → label + conf │
│  ⏱ 4s timeout       │    │                                       │
│    → stop session    │    │  2. Rule-based fallback:              │
└──────────┬───────────┘    │     palm-scale finger counting        │
           │                │     + shape detectors                 │
           │                └──────────────┬────────────────────────┘
           │                               │
           └──────────────┬────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STABILITY & SAFETY LAYER                         │
│                                                                     │
│  • Gesture history buffer (3 frames) — must be stable              │
│  • Hold-to-confirm timer (0.4s standard, 5s for shutdown)          │
│  • Confidence threshold (≥ 0.75 to fire)                           │
│  • Cooldown (1.0s between actions)                                  │
│  • Ghost frames (3 frames) — bridge momentary hand loss            │
│  • Danger hold timer — independent 5s timer for shutdown           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  Confirmed gesture + auth status
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ACTION EXECUTION                                 │
│                                                                     │
│  pyautogui.press()     — key presses (volume, media, escape)       │
│  pyautogui.hotkey()    — combinations (Win+Tab, Win+L, Win+D)      │
│  pyautogui.scroll()    — scroll wheel                              │
│  win32api.SendMessage  — targeted scroll to focused window         │
│  sbc.set_brightness()  — monitor brightness                        │
│  ScreenRecorder        — start/stop screen recording               │
│  subprocess.run()      — shutdown /s /t 0 (after 3s countdown)    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ACTIVITY LOGGER                                  │
│   Writes timestamped events to activity_log.txt                    │
└─────────────────────────────────────────────────────────────────────┘
```

## 6.2 Data Flow Description

**Step 1 — Frame capture:** The main thread reads frames from the webcam at 30 FPS and writes them to a shared state dictionary protected by a threading lock.

**Step 2 — Gesture worker thread:** A dedicated thread reads frames at 25 FPS, applies CLAHE preprocessing, runs MediaPipe, extracts landmarks, and runs gesture detection.

**Step 3 — Auth worker thread:** Every 2 seconds, a separate background thread reads the current frame, applies CLAHE, runs HOG face detection, computes a 128-dim encoding, and compares it against stored encodings. It updates `last_seen_time` on success.

**Step 4 — Display thread (main thread):** Reads the annotated frame from shared state and calls `draw_ui()` to overlay the auth status, finger count dots, gesture label, hold progress bars, and cooldown bar before displaying via `cv2.imshow()`.

---

# Chapter 7: System Architecture

## 7.1 Module Architecture

```
gesture_control_project/
│
├── main.py                   ← Entry point
│   └── check_dependencies()
│   └── launches gui_app.main()
│
├── gui_app.py                ← Presentation layer
│   └── GestureControlGUI     (Tkinter, 5-page sidebar app)
│       ├── Control page      (start/stop session, recording)
│       ├── Users page        (register, delete, view users)
│       ├── Gestures page     (remap gesture → action)
│       ├── Settings page     (tune all parameters)
│       └── Logs page         (view/clear activity log)
│
├── gesture_control.py        ← Core engine
│   └── GestureController
│       ├── _fingers_up()     (palm-scale finger counting)
│       ├── _ml_detect()      (SVM inference)
│       ├── _rule_detect()    (rule-based fallback)
│       ├── detect_gesture()  (unified entry point)
│       ├── execute_action()  (action dispatch)
│       ├── draw_ui()         (OpenCV overlay)
│       └── start()           (main loop: Phase 1 auth + Phase 2 gesture)
│
├── face_auth.py              ← Identity layer
│   └── FaceAuthenticator
│       ├── register_multi_angle()   (CNN registration)
│       ├── _authenticate_frame()    (HOG + CLAHE live auth)
│       ├── continuous_authenticate() (non-blocking wrapper)
│       └── _bg_auth()               (background worker)
│
├── config_manager.py         ← Configuration layer
│   └── ConfigManager
│       ├── User CRUD
│       ├── Gesture mapping CRUD
│       └── Settings CRUD
│
├── gesture_collector.py      ← Training data tool
│   └── GestureCollectorApp   (standalone Tkinter app)
│
├── gesture_trainer.py        ← ML training
│   └── GestureTrainer
│       ├── train()           (SVM pipeline training)
│       └── load_model()      (static loader)
│
├── logger.py                 ← Logging layer
│   └── ActivityLogger        (thread-safe file logger)
│
└── screen_recorder.py        ← Recording layer
    └── ScreenRecorder        (background screen capture)
```

## 7.2 Threading Architecture

The system uses four concurrent threads to keep the UI responsive and gesture detection smooth:

```
Main Thread (Tkinter event loop)
│
├── GUI rendering + user interaction
└── Camera display loop (cv2.imshow at 30 FPS)

Gesture Worker Thread (daemon)
│
├── Reads frames from shared state at 25 FPS
├── CLAHE preprocessing
├── MediaPipe hand detection
├── ML/rule gesture detection
├── Hold-to-confirm logic
└── execute_action() calls

Auth Background Thread (daemon, spawned every 2s)
│
├── HOG face detection on current frame
├── 128-dim encoding comparison
└── Updates last_seen_time / _auth_result

Screen Recorder Thread (daemon, on demand)
│
└── pyautogui.screenshot() at 10 FPS → VideoWriter
```

**Thread safety:** All shared state (frame buffer, auth result, gesture state) is protected by a single `threading.Lock`. Auth state uses a dedicated `_auth_lock`. The gesture worker never blocks the display thread.

## 7.3 State Machine — Session Lifecycle

```
[IDLE]
  │  User clicks "Start Session"
  ▼
[PHASE 1: AUTHENTICATION]
  │  Spawns auth workers every 0.35s
  │  Displays "SCANNING FOR REGISTERED USER..."
  │  Timeout: 30 seconds
  │
  ├── Face recognised → [PHASE 2: GESTURE CONTROL]
  └── Timeout / Q pressed → [IDLE]

[PHASE 2: GESTURE CONTROL]
  │  Gesture worker thread running
  │  Auth background thread running every 2s
  │
  ├── Q pressed → [CLEANUP] → [IDLE]
  ├── auth_ok = False (face absent 4s) → [CLEANUP] → [IDLE]
  └── Shutdown gesture confirmed → [SHUTDOWN COUNTDOWN]
        │  3-second cancellable countdown
        ├── Hand dropped → [PHASE 2: GESTURE CONTROL]
        └── Countdown complete → subprocess shutdown

[CLEANUP]
  │  cam.release()
  │  cv2.destroyAllWindows()
  └── log_session_end()
```

---

# Chapter 8: System Design

## 8.1 Gesture Detection Design

### 8.1.1 Feature Representation

Each hand is represented as a 63-dimensional vector: 21 landmarks × 3 coordinates (x, y, z). For two-hand gestures, both hands are concatenated to form a 126-dimensional vector.

**Normalisation:**
```
coords -= coords[0]                    # translate wrist to origin
scale = ||coords[9]|| + ε             # scale by wrist→middle-MCP distance
coords /= scale                        # scale-invariant representation
```

This makes the feature vector independent of hand size and camera distance.

### 8.1.2 Finger Extension Detection (Rule-Based)

The core innovation in finger counting is the **palm-scale normalised, 3-cue majority vote** approach:

```
palm_scale = ||lm[9] - lm[0]||   # wrist → middle-MCP Euclidean distance

For each finger (index, middle, ring, pinky):
  cue1 = tip.y < dip.y - palm_scale × 0.04   # tip above DIP (tight)
  cue2 = tip.y < pip.y - palm_scale × 0.02   # tip above PIP (medium)
  cue3 = tip.y < mcp.y - palm_scale × 0.10   # tip above MCP (loose)
  extended = (cue1 + cue2 + cue3) ≥ 2        # majority vote

For thumb:
  lateral_vec = normalise(lm[17] - lm[5])    # index→pinky direction
  thumb_proj = -(tip_delta · lateral_vec)     # projection away from pinky
  extended = thumb_proj > palm_scale × 0.08
```

**Why this works better than simple y-comparison:**
- `palm_scale` is a 2D Euclidean distance — it never collapses to zero when the hand is tilted horizontally (unlike `abs(dip.y - mcp.y)` which approaches zero for horizontal hands).
- Three independent cues with majority voting means a single noisy landmark doesn't flip the count.
- The thumb uses a lateral projection rather than a y-comparison, making it correct for both left and right hands at any tilt angle.

### 8.1.3 ML Classification Pipeline

```
Training data: faces/<user>_gestures.csv
  └── 150 samples × 19 gestures = 2,850 rows
  └── Each row: [label, f1, f2, ..., f126]

Pipeline:
  StandardScaler → SVC(kernel='rbf', C=10, gamma='scale', probability=True)

Inference:
  feats = extract_features(hand_landmarks)   # 126-dim vector
  proba = pipeline.predict_proba([feats])    # probability per class
  label = classes[argmax(proba)]
  conf  = max(proba)
```

**Confidence thresholds:**
- `conf ≥ 0.92` → INSTANT path: fires after 2 stable frames (~80ms)
- `conf ≥ 0.75` → NORMAL path: fires after 3 stable frames + 0.4s hold
- `conf < 0.75` → IGNORED: too uncertain, reset hold timer

### 8.1.4 Gesture Firing Pipeline

```
Frame N:
  detect_gesture() → (label, confidence)
  _update_history(label)          # append to 3-frame history
  _stable_gesture()               # all 3 frames same? → stable label

  if stable and confidence ≥ threshold:
    _update_hold(stable)          # start/continue hold timer
    if hold_time ≥ HOLD_DURATION: # 0.4s
      execute_action(label)
```

## 8.2 Face Authentication Design

### 8.2.1 Registration

```
5 camera frames (different angles) → CNN face_locations()
→ face_encodings(num_jitters=5)   # high-quality, runs once
→ store list of 128-dim vectors in faces/<user>_encoding.pkl
```

### 8.2.2 Live Authentication

```
Every 2 seconds (background thread):
  frame → CLAHE preprocessing
        → cv2.resize(×0.6)
        → quality gate (brightness ≥ 30, sharpness ≥ 50)
        → HOG face_locations()
        → face_encodings(num_jitters=2)
        → face_distance(known_encodings, live_encoding)
        → dist ≤ tolerance (0.5) → match
        → confidence = (1 - dist) × 100
        → update last_seen_time
```

### 8.2.3 Timeout Mechanism

```
Every frame (main gesture loop):
  if last_seen_time > 0 and (now - last_seen_time) > 4.0:
    _auth_result = False
    → gesture worker exits
    → session ends
```

## 8.3 Safety Design for Destructive Actions

```
Shutdown/Restart require ALL of:
  1. Gesture = count_10 (both palms open)
  2. ML confidence ≥ 75%
  3. _danger_hold_start timer ≥ 5.0 seconds
     (independent timer, never resets on cooldown cycles)
  4. execute_action() called → spawns _shutdown_countdown()
  5. 3-second countdown with visual overlay
  6. Drop hands at any point → _shutdown_cancelled = True → abort
```

## 8.4 Configuration Design

All runtime parameters are stored in `system_config.json` and managed by `ConfigManager`. Hot-path settings (`cooldown`, `gesture_stability`, `face_tolerance`) are cached as instance attributes to avoid JSON dict lookups every frame.

```json
{
  "settings": {
    "cooldown": 1.0,
    "gesture_stability": 3,
    "face_tolerance": 0.5,
    "auth_check_frequency": 2,
    "detection_confidence": 0.7,
    "tracking_confidence": 0.7
  }
}
```

---

# Chapter 9: Implementation

## 9.1 Complete Gesture Mapping

| Gesture | Hand Shape / Position | Default Action | Mechanism |
|---|---|---|---|
| `fist` | All fingers closed, thumb tucked | Play / Pause | Rule: `_is_fist()` |
| `count_1` | Index finger only | Volume Up | Rule: finger count = 1 |
| `count_2` | Index + middle | Volume Down | Rule: finger count = 2 |
| `count_3` | Index + middle + ring | Mute | Rule: finger count = 3 |
| `count_4` | All 4 fingers, no thumb | Play / Pause | Rule: finger count = 4 |
| `count_5_center` | Open palm, center frame | Screenshot | Rule: 5 fingers + palm zone |
| `count_5_left` | Open palm, left edge | Navigate Back | Rule: 5 fingers + palm zone |
| `count_5_right` | Open palm, right edge | Navigate Forward | Rule: 5 fingers + palm zone |
| `count_5_up` | Open palm, top frame | Scroll Up | Rule: 5 fingers + palm zone |
| `count_5_down` | Open palm, bottom frame | Scroll Down | Rule: 5 fingers + palm zone |
| `thumbs_up` | Thumb up, others curled | Brightness Up | Rule: `_is_thumbs_up()` |
| `thumbs_down` | Thumb down, others curled | Brightness Down | Rule: `_is_thumbs_down()` |
| `ok_sign` | Thumb+index circle, others up | Screen Record | Rule: `_is_ok()` |
| `peace_sign` | Index+middle spread in V | Screenshot | Rule: `_is_peace()` |
| `count_6` | 6 fingers (both hands) | Escape | ML + Rule |
| `count_7` | 7 fingers (both hands) | Task View (Win+Tab) | ML + Rule |
| `count_8` | 8 fingers (both hands) | Lock Screen (Win+L) | ML + Rule |
| `count_9` | 9 fingers (both hands) | Show Desktop (Win+D) | ML + Rule |
| `count_10` | Both palms open (10 fingers) | Shutdown PC (5s hold) | ML + Rule + Danger guard |

## 9.2 Key Implementation Details

### 9.2.1 CLAHE Preprocessing

Applied to both face authentication and hand detection frames:

```python
def _enhance_frame(self, frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

CLAHE (Contrast Limited Adaptive Histogram Equalisation) operates on the L (luminance) channel of the LAB colour space, leaving colour information unchanged. `clipLimit=2.0` prevents over-amplification of noise.

### 9.2.2 Temporal Smoothing (Ghost Frames)

When the hand briefly disappears from the frame (e.g., fast movement), the system bridges up to 3 frames:

```python
if results.multi_hand_landmarks:
    self._frames_no_hand = 0
    # ... normal detection
else:
    self._frames_no_hand += 1
    if self._frames_no_hand <= GHOST_FRAMES:   # 3
        gesture    = self._last_gesture
        confidence = self._last_confidence * 0.7   # decay
        self._update_history(gesture)
    else:
        gesture = "none"
        self.gesture_history.clear()
        self._update_hold(None)
```

### 9.2.3 Danger Hold Timer

The shutdown timer is independent of the per-fire hold timer:

```python
# In _update_hold():
if stable != self._danger_gesture:
    self._danger_gesture    = stable
    self._danger_hold_start = now   # starts once, never resets on cooldown

# In execute_action():
if action in _DANGER_ACTIONS:
    held = time.time() - self._danger_hold_start
    if held < 5.0:
        print(f"⏳ hold for {5.0 - held:.1f}s more...")
        return   # not yet
    # proceed to countdown
```

### 9.2.4 Non-Blocking Face Re-Authentication

```python
def continuous_authenticate(self, frame):
    # 1. Timeout check — runs every frame, O(1)
    if last_seen > 0 and now - last_seen > 4.0:
        self._auth_result = False
        return False

    # 2. Spawn background thread every 2s
    if now - self.last_auth_time >= 2.0 and not self._auth_running:
        self.last_auth_time = now
        self._auth_running = True
        threading.Thread(target=self._bg_auth,
                         args=(frame.copy(),), daemon=True).start()

    # 3. Return last known result immediately — never blocks
    return self._auth_result
```

### 9.2.5 Scroll Implementation

Scroll uses both Win32 API (for the focused window) and pyautogui (fallback):

```python
hwnd  = win32gui.WindowFromPoint((cx, cy))
delta = 120 if v == "scroll_up" else -120
for _ in range(3):
    win32api.SendMessage(hwnd, win32con.WM_MOUSEWHEEL,
                         win32api.MAKELONG(0, delta),
                         win32api.MAKELONG(cx, cy))
pyautogui.scroll(3 if v == "scroll_up" else -3)
```

## 9.3 Training Workflow

```
1. python gesture_collector.py
   → Select user → Select gesture → Start Collecting
   → 150 samples per gesture × 19 gestures = 2,850 rows
   → Saved to faces/<user>_gestures.csv

2. Click "Train Model" in gesture_collector.py
   → GestureTrainer.train() called
   → StandardScaler + SVC(RBF, C=10) pipeline
   → 5-fold cross-validation accuracy reported
   → Model saved to faces/<user>_model.pkl
   → system_config.json updated with model path

3. Next session start:
   → GestureController._probe_available_models()
   → Model loaded automatically
   → ML detection active from first frame
```

---

# Chapter 10: Testing

## 10.1 Unit Testing — Finger Detection

A dedicated test file `_test_fingers.py` validates the `_fingers_up()` method with 26 test cases covering:

- Right hand: 0–5 fingers, thumb only, pinky only, ring+pinky
- Left hand: 0–5 fingers, thumb only
- Two-hand totals: 5+1, 5+2, 5+3, 5+4, 5+5, 4+2, 4+3, 3+3, 4+1, 3+4

**Test methodology:** Synthetic `FakeLM` objects simulate MediaPipe landmark positions for known finger configurations. Extended fingers have tips placed clearly above their MCPs; curled fingers have tips placed below their MCPs.

**Result:** 26/26 tests pass.

```
── Right-hand tests ─────────────────────────────────────
  PASS   0 fingers — fist: 0
  PASS   1 finger  — index: 1
  PASS   2 fingers — index+middle: 2
  PASS   3 fingers — i+m+r: 3
  PASS   4 fingers — no thumb: 4
  PASS   5 fingers — all: 5
  PASS   thumb only: 1
  PASS   pinky only: 1
  PASS   ring+pinky: 2

── Left-hand tests ──────────────────────────────────────
  PASS   0 fingers — fist: 0
  PASS   1 finger  — index: 1
  ...   (all pass)

── Two-hand totals ──────────────────────────────────────
  PASS   5+1 = 6
  PASS   5+2 = 7
  ...   (all pass)

  Results: 26 passed, 0 failed
```

## 10.2 Integration Testing

### 10.2.1 Face Authentication Tests

| Test Case | Expected | Result |
|---|---|---|
| Registered user, good lighting | Authenticated in < 5s | ✅ Pass |
| Registered user, dim lighting | Authenticated (CLAHE helps) | ✅ Pass |
| Unregistered face | Not authenticated, session blocked | ✅ Pass |
| Face leaves frame for 3s | Warning overlay shown | ✅ Pass |
| Face leaves frame for 4s | Session terminates | ✅ Pass |
| Face returns before 4s | Session continues | ✅ Pass |
| Blurry frame (motion blur) | Frame skipped, no false failure | ✅ Pass |

### 10.2.2 Gesture Detection Tests

| Test Case | Expected | Result |
|---|---|---|
| Raise 1 finger | `count_1` → Volume Up | ✅ Pass |
| Raise 7 fingers (both hands) | `count_7` → Task View | ✅ Pass |
| Fist | `fist` → Play/Pause | ✅ Pass |
| Thumbs up | `thumbs_up` → Brightness Up | ✅ Pass |
| Open palm, move left | `count_5_left` → Navigate Back | ✅ Pass |
| Flash 10 fingers briefly (< 5s) | No shutdown triggered | ✅ Pass |
| Hold 10 fingers for 5s | Shutdown countdown starts | ✅ Pass |
| Drop hands during countdown | Shutdown cancelled | ✅ Pass |

### 10.2.3 Safety Tests

| Test Case | Expected | Result |
|---|---|---|
| Shutdown with confidence < 75% | Blocked, message printed | ✅ Pass |
| Shutdown held < 5 seconds | Blocked, remaining time printed | ✅ Pass |
| Shutdown held ≥ 5 seconds | Countdown starts | ✅ Pass |
| Gesture during cooldown | Action suppressed | ✅ Pass |
| Gesture by unrecognised face | Action blocked, logged | ✅ Pass |

## 10.3 Performance Testing

| Metric | Measured Value |
|---|---|
| Gesture worker FPS | 24–25 FPS |
| Display FPS | 28–30 FPS |
| Auth thread latency | 220–380 ms |
| ML inference time (SVM) | 2–4 ms |
| Gesture → action latency | 400–600 ms (hold-to-confirm) |
| Memory usage (steady state) | ~350–450 MB |
| CPU usage (gesture loop) | 25–40% (single core) |

## 10.4 Cross-User Testing

Three users (Vighnesh, Aryan, Atharv) were registered and tested:

| User | Face Auth Rate | Gesture Accuracy (ML) |
|---|---|---|
| Vighnesh (trained model) | 97% | 93% |
| Aryan (no model, rule-based) | 95% | 82% (rule-based) |
| Atharv (no model, rule-based) | 94% | 80% (rule-based) |

The significant accuracy gap between ML and rule-based detection for Aryan and Atharv confirms the value of per-user model training.

---

# Chapter 11: Results and Analysis

## 11.1 System Output Screens

### Screen 1 — Main GUI (Control Page)
```
┌─────────────────────────────────────────────────────────────────────┐
│ ✋ GestureOS  v2.0  │  🎮 Control Center                            │
│─────────────────────│  Start a session and monitor gesture activity  │
│ 🎮 Control    ●     │─────────────────────────────────────────────── │
│ 👤 Users            │  ● Ready to start                              │
│ ✋ Gestures         │  Not authenticated                             │
│ ⚙️  Settings        │                                                │
│ 📋 Logs             │  ┌─ Session Control ──────────────────────┐   │
│                     │  │  ▶  Start Session                      │   │
│                     │  │  ⏹  Stop Session  (disabled)           │   │
│ Face-Authenticated  │  └────────────────────────────────────────┘   │
│ Real-Time·Hands-Free│                                                │
│                     │  CURRENT USER: —   STATUS: Ready   CONF: —    │
└─────────────────────┴────────────────────────────────────────────────┘
```

### Screen 2 — Authentication Phase (Camera Window)
```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   SCANNING FOR REGISTERED USER...                    │
│   Closest match: 72%                                 │
│                                                      │
│         [face bounding box drawn in red/green]       │
│                                                      │
│                                                      │
│   Timeout in 24s  |  Press Q to cancel              │
└──────────────────────────────────────────────────────┘
```

### Screen 3 — Active Session (Camera Window)
```
┌──────────────────────────────────────────────────────┐
│ ┌──────────────────────────────┐    ● ● ● ● ● ○ ○ 5 │
│ │ USER: vighnesh               │                     │
│ │ STATUS: AUTHORIZED  [ML]     │                     │
│ │ Face conf: 87.3%             │                     │
│ │ ████████████████░░░░░░░░░░░  │                     │
│ └──────────────────────────────┘                     │
│  3 FINGERS  →  Mute  [82%]                           │
│  Hold... ████████████░░░░░░░░░░                      │
│                                                      │
│   [hand skeleton drawn by MediaPipe]                 │
│                                                      │
│                              Cooldown 0.6s           │
│                                          Q = Quit    │
└──────────────────────────────────────────────────────┘
```

### Screen 4 — Face Absence Warning
```
┌──────────────────────────────────────────────────────┐
│ ┌──────────────────────────────────────────────────┐ │
│ │ USER: vighnesh                                   │ │
│ │ FACE NOT DETECTED                                │ │
│ │ Stopping in 2.3s — look at camera               │ │
│ │ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  │ │
│ └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Screen 5 — Shutdown Countdown
```
┌──────────────────────────────────────────────────────┐
│                                                      │
│                                                      │
│          !! SHUTDOWN COUNTDOWN !!                    │
│                                                      │
│             DROP HAND TO CANCEL                      │
│                                                      │
│                                                      │
│  SHUTDOWN 1.4s  ████████████████████████░░░░░░░░░   │
└──────────────────────────────────────────────────────┘
```

### Screen 6 — Users Page
```
┌─────────────────────────────────────────────────────────────────────┐
│ 👤 User Management                                                  │
│─────────────────────────────────────────────────────────────────────│
│ Registered Users  [3]  │  Register New User          [Multi-Angle] │
│─────────────────────── │  FULL NAME / USERNAME                     │
│   vighnesh             │  ┌─────────────────────────────────────┐  │
│   Aryan                │  │                                     │  │
│   Atharv               │  └─────────────────────────────────────┘  │
│                        │                                            │
│  User    : vighnesh    │  [Camera Preview 440×290]                  │
│  Created : 2026-03-18  │                                            │
│  Last    : 2026-05-30  │  Progress: ● ● ● ● ●                      │
│  Sessions: 84          │  ✅ All 5 angles captured                  │
│                        │                                            │
│  🗑 Delete User  ↻     │  ✅ Register User   📁 Upload Photo        │
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 7 — Activity Log
```
┌─────────────────────────────────────────────────────────────────────┐
│ 📋 Activity Logs                                                    │
│─────────────────────────────────────────────────────────────────────│
│ ↻ Refresh   🗑 Clear                              142 entries       │
│─────────────────────────────────────────────────────────────────────│
│ 2026-06-01 13:26:31 [vighnesh] [AUTH_SUCCESS] Face authentication   │
│ 2026-06-01 13:26:31 [vighnesh] [SESSION_START] Gesture control      │
│ 2026-06-01 13:26:34 [vighnesh] [GESTURE] 'count_3' → 'mute'        │
│ 2026-06-01 13:26:37 [vighnesh] [GESTURE] 'count_1' → 'volumeup'    │
│ 2026-06-01 13:26:41 [vighnesh] [GESTURE] 'thumbs_up' → 'brightness'│
│ 2026-06-01 13:30:45 [vighnesh] [AUTH_FAILURE] Face absent 4.1s     │
│ 2026-06-01 13:30:45 [vighnesh] [SESSION_END] Session ended          │
└─────────────────────────────────────────────────────────────────────┘
```

## 11.2 Accuracy Analysis

### Gesture Recognition Accuracy (Vighnesh — ML model)

| Gesture | Accuracy |
|---|---|
| count_1 through count_4 | 95–98% |
| count_5 (all zones) | 91–94% |
| thumbs_up / thumbs_down | 96–98% |
| ok_sign | 89% |
| peace_sign | 92% |
| fist | 94% |
| count_6 through count_10 | 88–93% |
| **Overall** | **~93%** |

### Face Authentication Accuracy

| Condition | Detection Rate |
|---|---|
| Good lighting, frontal | 97–99% |
| Dim lighting (CLAHE active) | 91–94% |
| Side angle (< 30°) | 88–92% |
| Glasses | 85–90% |
| Motion blur (skipped by quality gate) | N/A (frame skipped) |

## 11.3 Latency Analysis

The end-to-end latency from gesture formation to action execution:

```
Gesture formed by user
  → MediaPipe detection:        ~10–20 ms
  → Feature extraction:          ~1–2 ms
  → SVM inference:               ~2–4 ms
  → Stability buffer (3 frames): ~120 ms at 25 FPS
  → Hold-to-confirm:             400 ms
  ─────────────────────────────────────────
  Total:                         ~535–546 ms
```

This is perceptible but acceptable for the use case. The 0.4s hold is intentional — it prevents accidental triggers.

---

# Chapter 12: Applications and Future Scope

## 12.1 Current Applications

### 12.1.1 Accessibility
Users with motor disabilities, repetitive strain injuries, or conditions affecting fine motor control (Parkinson's, arthritis) can control their PC through gross hand movements rather than precise keyboard/mouse input. The face authentication ensures the system only responds to the intended user.

### 12.1.2 Presentation Control
A presenter can advance slides, adjust volume, or take screenshots without touching a keyboard or clicker. The palm-left/right gestures map naturally to "previous/next slide" navigation.

### 12.1.3 Media Control
Controlling music or video playback from across the room — volume up/down, play/pause, mute — without needing to reach for a device.

### 12.1.4 Hygiene-Critical Environments
In medical examination rooms, laboratory settings, or food preparation areas, touchless PC control eliminates the need to touch shared input devices between tasks.

### 12.1.5 Smart Home / Kiosk Integration
The system can be extended to control smart home devices or serve as a touchless kiosk interface in public spaces.

## 12.2 Future Scope

### 12.2.1 Dynamic Gesture Recognition
The current system recognises only static gestures (hand shapes held for 0.4s). Adding temporal models (LSTM or Temporal Convolutional Networks) would enable dynamic gestures — swipes, circles, waves — dramatically expanding the gesture vocabulary.

### 12.2.2 Mouse Cursor Control
Mapping the index fingertip position to cursor movement would enable full touchless mouse control. This requires smoothing (Kalman filter or exponential moving average) to prevent jitter.

### 12.2.3 Per-User Gesture Profiles
Currently all users share the same gesture-to-action mappings. Future versions could store per-user mappings so each registered user has their own personalised configuration that loads automatically on authentication.

### 12.2.4 Voice + Gesture Fusion
Combining gesture recognition with voice commands (via `speech_recognition` + `pyttsx3`) would create a multimodal interface — gestures for frequent actions, voice for complex commands.

### 12.2.5 Custom Action Scripts
Allowing users to map gestures to arbitrary shell commands or Python scripts would make the system extensible without code changes — e.g., "3 fingers = open Spotify", "thumbs up = run backup script".

### 12.2.6 Cross-Platform Support
Replacing `pywin32` scroll calls and `screen-brightness-control` with cross-platform equivalents would enable Linux and macOS support.

### 12.2.7 Improved Face Authentication
- **Anti-spoofing:** Detect printed photos or screen replays using liveness detection (blink detection, depth estimation).
- **Low-light enhancement:** Replace CLAHE with a learned low-light enhancement model for better performance in very dark conditions.
- **Faster recognition:** Replace dlib HOG with a lightweight CNN face detector (e.g., YuNet in OpenCV) for faster detection at the same accuracy.

### 12.2.8 Web Dashboard
A Flask/FastAPI web interface for remote log viewing, user management, and gesture configuration — useful for IT administrators managing multiple kiosk deployments.

### 12.2.9 Gesture Feedback
Audio feedback (short beep or TTS announcement) when a gesture is recognised would improve usability, especially for users who cannot see the camera overlay.

---

# Chapter 13: Conclusion

## 13.1 Summary

This project successfully demonstrates that a fully functional, face-authenticated, real-time gesture control system can be built using only commodity hardware (a standard webcam and a CPU) and open-source Python libraries.

The system achieves its primary objectives:

- **19 distinct gestures** are reliably detected and mapped to system actions.
- **Face authentication** restricts all commands to registered users, with continuous re-verification and a 4-second absence timeout.
- **Per-user SVM models** achieve ~93% gesture recognition accuracy for trained users.
- **Safety mechanisms** — hold-to-confirm, danger hold timer, confidence thresholds, and cancellable countdowns — prevent accidental or malicious triggering of destructive actions.
- **A complete GUI** provides user management, gesture remapping, settings tuning, and activity log viewing without any command-line interaction.

## 13.2 Key Technical Contributions

1. **Palm-scale normalised finger counting** with 3-cue majority voting — robust to hand tilt, rotation, and camera distance. Validated with 26 unit tests (26/26 pass).

2. **Non-blocking continuous face re-authentication** — background thread runs every 2 seconds, never blocking the 25 FPS gesture loop. Wall-clock timeout (4 seconds) replaces the unreliable failure-count approach.

3. **Independent danger hold timer** — the shutdown gesture requires a dedicated 5-second continuous hold timer that is never reset by the per-fire cooldown cycle, fixing a fundamental bug in the original design.

4. **CLAHE preprocessing for both face and hand detection** — lighting normalisation applied consistently across the pipeline, improving detection rates in variable lighting conditions.

5. **Dual-path gesture detection** — ML (SVM) for trained gestures with confidence-based firing, rule-based fallback for untrained users or low-confidence frames.

## 13.3 Lessons Learned

- **Threading discipline is critical.** The gesture loop, auth loop, and display loop must be carefully isolated with locks to prevent race conditions and frame drops.
- **Scale matters for HOG.** The original 0.4× scale was too aggressive — faces at normal desk distance became too small for reliable HOG detection. 0.6× significantly improved detection rates.
- **Per-user models are worth the effort.** The 10–13% accuracy gap between ML and rule-based detection for untrained users is significant enough to make training a practical necessity for reliable use.
- **Safety UX requires deliberate design.** The original shutdown implementation had a fundamental timer bug that prevented it from ever firing. Proper safety mechanisms require careful reasoning about timer lifecycles.

## 13.4 Final Remarks

GestureOS demonstrates that the gap between research-grade gesture control systems and practical, deployable applications can be bridged with careful engineering. The system is production-ready for personal use and provides a solid foundation for the future enhancements described in Chapter 12.

The combination of face authentication, per-user ML models, robust finger counting, and comprehensive safety mechanisms makes this system meaningfully more capable and secure than existing open-source gesture control tools.

---

## Appendix A: File Structure

```
gesture_control_project/
├── main.py                    Entry point — dependency check + launch GUI
├── gui_app.py                 Tkinter GUI (5-page sidebar application)
├── gesture_control.py         Core engine — detection, actions, CV loop
├── face_auth.py               Face recognition — registration + live auth
├── config_manager.py          JSON config management
├── gesture_collector.py       Training data collection tool
├── gesture_trainer.py         SVM model training
├── logger.py                  Thread-safe activity logger
├── screen_recorder.py         Background screen recorder
├── _test_fingers.py           Unit tests for finger detection
├── system_config.json         Runtime configuration
├── activity_log.txt           Activity log
├── requirements.txt           Python dependencies
├── faces/
│   ├── vighnesh_encoding.pkl  Face encodings (128-dim vectors)
│   ├── Aryan_encoding.pkl
│   ├── Atharv_encoding.pkl
│   ├── vighnesh_model.pkl     Trained SVM gesture model
│   └── vighnesh_gestures.csv  Training data (126 features + label)
└── recordings/                Screen recordings (MP4)
```

## Appendix B: How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py

# Collect gesture training data (optional, improves accuracy)
python gesture_collector.py

# Train model from command line (optional)
python gesture_trainer.py --user vighnesh

# Run finger detection unit tests
python _test_fingers.py
```

## Appendix C: Gesture Quick Reference

```
☝  1 finger     → Volume Up          ✌  2 fingers    → Volume Down
🤟  3 fingers    → Mute               ✊  4 fingers    → Play/Pause
✊  Fist         → Play/Pause         👍  Thumbs Up   → Brightness Up
👎  Thumbs Down  → Brightness Down    👌  OK Sign     → Screen Record
✌  Peace Sign   → Screenshot

🖐  Palm CENTER  → Screenshot         👋  Palm LEFT   → Navigate Back
👋  Palm RIGHT   → Navigate Forward   👋  Palm UP     → Scroll Up
👋  Palm DOWN    → Scroll Down

✋  6 fingers    → Escape             ✋  7 fingers   → Task View
✋  8 fingers    → Lock Screen        ✋  9 fingers   → Show Desktop
🙌  10 fingers   → Shutdown (5s hold)
```

---
*End of Report*
