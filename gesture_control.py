"""
Gesture Controller
==================
Features:

1. ML classifier (per-user SVM) — replaces if/elif threshold rules.
   Falls back to rule-based detection if no trained model exists.

2. CLAHE preprocessing — normalises lighting before MediaPipe inference.
   Trained in varied lighting → works across lighting conditions.

3. Confidence-based instant firing — high-confidence gestures fire in
   ~80 ms; lower-confidence gestures use the stability + hold window.

4. Temporal smoothing — bridges up to 3 frames of lost hand detection
   so single-frame glitches don't reset the gesture state.

5. Per-user model loading — when face auth switches users, the matching
   gesture model is loaded automatically.

6. model_complexity=1 for better landmark accuracy on CPU.

7. Palm-scale normalised finger detection — uses wrist→middle-MCP
   Euclidean distance as scale reference (never collapses to zero).
   Each finger uses 3 independent cues with majority voting (2/3),
   making counts robust to partial bends and tilted hands.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import glob
import pickle as _pkl
import screen_brightness_control as sbc
import win32api
import win32con
import win32gui
from screen_recorder import ScreenRecorder
from gesture_trainer import GestureTrainer

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ── Tuning constants ───────────────────────────────────────────────────────────
HOLD_DURATION       = 0.4    # seconds for hold-to-confirm (reduced from 0.6)
INSTANT_CONFIDENCE  = 0.92   # above this → fire immediately (2-frame stable)
NORMAL_CONFIDENCE   = 0.75   # above this → use 3-frame stability window
# Below NORMAL_CONFIDENCE → ignore (too uncertain)

GHOST_FRAMES        = 3      # frames to hold last gesture when hand disappears

# Directional palm gestures bypass hold — position IS the confirmation
_INSTANT_GESTURES = frozenset({
    "count_5_left", "count_5_right", "count_5_up",
    "count_5_down",
})


# ── Feature extraction (must match gesture_collector.py) ──────────────────────

def _extract_features(hand_landmarks_list):
    """
    Normalised feature vector from detected hands.

    Single hand  → 63 values  (21 landmarks × 3 coords)
    Two hands    → 126 values (both hands concatenated, left hand first)

    For two-hand gestures (counts 6–10) both hands are encoded.
    For single-hand gestures only the first hand is used (padded to 126
    with zeros so the vector length is always consistent at 126).

    Normalisation: subtract each hand's own wrist, scale by palm size.
    This makes features lighting- and hand-size-independent.
    """
    if not hand_landmarks_list:
        return None

    def _encode_hand(lm_list):
        coords = np.array([[p.x, p.y, p.z] for p in lm_list],
                          dtype=np.float32)
        coords -= coords[0]                          # wrist to origin
        scale   = np.linalg.norm(coords[9]) + 1e-6  # scale by palm size
        coords /= scale
        return coords.flatten()                      # (63,)

    hand0 = _encode_hand(hand_landmarks_list[0].landmark)

    if len(hand_landmarks_list) >= 2:
        hand1 = _encode_hand(hand_landmarks_list[1].landmark)
    else:
        hand1 = np.zeros(63, dtype=np.float32)       # pad for single hand

    return np.concatenate([hand0, hand1])             # (126,)


# ── Main controller ────────────────────────────────────────────────────────────

class GestureController:

    def __init__(self, config_manager, authenticator, logger, recorder=None):
        self.config_manager = config_manager
        self.authenticator  = authenticator
        self.logger         = logger
        self.recorder = recorder if recorder is not None else ScreenRecorder(logger)

        # model_complexity=1: better landmark accuracy, still runs on CPU
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=config_manager.get_setting("detection_confidence"),
            min_tracking_confidence=config_manager.get_setting("tracking_confidence"),
        )

        # CLAHE for lighting normalisation
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Per-user ML model (loaded on auth)
        self._gesture_model  = None   # loaded sklearn Pipeline
        self._model_username = None   # which user the model belongs to
        self._model_classes  = []     # list of gesture label strings

        # Probe: try loading any available model at startup so we're ready
        self._probe_available_models()

        self.last_action_time = 0
        self.gesture_history  = []
        self.last_auth_status = True

        # Thread control
        self._stop_event = threading.Event()
        self._lock  = None
        self._state = None

        # Hold-to-confirm state
        self.hold_gesture    = None
        self.hold_start_time = 0
        self.gesture_fired   = False

        # Temporal smoothing — ghost frames
        self._last_gesture       = "none"
        self._frames_no_hand     = 0
        self._last_confidence    = 0.0

        # Shutdown countdown state
        self._shutdown_active    = False
        self._shutdown_cancelled = False

        # Danger-gesture hold timer — tracks how long a danger gesture has been
        # held continuously. Independent of hold_start_time (which resets on
        # every fire/cooldown cycle). Only resets when the gesture changes or
        # the hand drops.
        self._danger_gesture      = None   # which danger gesture is being held
        self._danger_hold_start   = 0.0    # when it was first seen

        # Palm zone thresholds (normalised 0–1)
        self.left_threshold   = 0.38
        self.right_threshold  = 0.62
        self.top_threshold    = 0.35
        self.bottom_threshold = 0.65

    def _probe_available_models(self):
        """
        Scan for any trained models at startup and report what's available.
        Pre-loads the first found model so detection works immediately
        even before face auth completes.
        """
        print("\n── Gesture Model Probe ──────────────────────────────")
        found = []

        for username in self.config_manager.get_all_users():
            data = GestureTrainer.load_model(username, self.config_manager)
            if data:
                found.append((username, data))
                print(f"  ✅ {username}: model found "
                      f"({len(data['classes'])} gestures, "
                      f"CV acc: {data.get('cv_acc', 0):.1f}%)")
            else:
                print(f"  ⚠️  {username}: no model — will use rule-based")

        # Also scan faces/ directory for any _model.pkl not in config
        for path in glob.glob("faces/*_model.pkl"):
            try:
                with open(path, 'rb') as f:
                    data = _pkl.load(f)
                uname = data.get("username", path)
                if not any(u == uname for u, _ in found):
                    found.append((uname, data))
                    print(f"  ✅ {uname}: model found at {path}")
            except Exception:
                pass

        if found:
            uname, data = found[0]
            self._gesture_model  = data["model"]
            self._model_classes  = data["classes"]
            self._model_username = "__preloaded__"
            # Warn if model uses old 63-feature format
            try:
                n_feat = data["model"].named_steps["scaler"].n_features_in_
                if n_feat == 63:
                    print(f"\n  ⚠️  Model uses old 63-feature format (single hand only).")
                    print(f"     Two-hand gestures (6–10 fingers) will use rule-based detection.")
                    print(f"     Re-collect data and retrain for full accuracy:")
                    print(f"     python gesture_collector.py")
                else:
                    print(f"\n  Pre-loaded model from '{uname}' — ML active from session start")
            except Exception:
                print(f"\n  Pre-loaded model from '{uname}' — ML active from session start")
        else:
            print("  No trained models found. Run gesture_collector.py to train.")
        print("─────────────────────────────────────────────────────\n")

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model_for_user(self, username: str):
        """
        Load the trained gesture model for `username`.
        Falls back to any available model if the user has none trained yet.
        This ensures ML detection works even before per-user training.
        """
        if username == self._model_username:
            return   # already loaded for this user

        # 1. Try the user's own model first
        data = GestureTrainer.load_model(username, self.config_manager)

        # 2. If not found, try any other user's model as a shared fallback
        if data is None:
            for other_user in self.config_manager.get_all_users():
                if other_user == username:
                    continue
                data = GestureTrainer.load_model(other_user, self.config_manager)
                if data:
                    print(f"ℹ️  No model for '{username}' — "
                          f"using '{other_user}' model as fallback")
                    break

        # 3. Also try scanning faces/ directory directly for any _model.pkl
        if data is None:
            for path in glob.glob("faces/*_model.pkl"):
                try:
                    with open(path, 'rb') as f:
                        data = _pkl.load(f)
                    print(f"ℹ️  Loaded fallback model from {path}")
                    break
                except Exception:
                    continue

        if data:
            self._gesture_model  = data["model"]
            self._model_classes  = data["classes"]
            self._model_username = username
            print(f"✅ Gesture model active for session as '{username}' "
                  f"({len(self._model_classes)} gestures, "
                  f"CV acc: {data.get('cv_acc', 0):.1f}%)")
        else:
            self._gesture_model  = None
            self._model_classes  = []
            self._model_username = username
            print(f"⚠️  No trained model found anywhere — using rule-based detection")
            print(f"   Run: python gesture_collector.py  to train a model")

    # ── Cooldown / stability ───────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()
        if self._lock is not None and self._state is not None:
            with self._lock:
                self._state["stop"] = True

    def _cooldown_active(self):
        # Use cached _cooldown — avoids dict lookup every frame
        return (time.time() - self.last_action_time
                < self.config_manager._cooldown)

    def _update_history(self, gesture):
        # Use cached _stability — avoids dict lookup every frame
        size = self.config_manager._stability
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > size:
            self.gesture_history.pop(0)

    def _stable_gesture(self):
        size = self.config_manager._stability
        if len(self.gesture_history) < size:
            return None
        return (self.gesture_history[0]
                if len(set(self.gesture_history)) == 1 else None)

    # ── Hold-to-confirm ────────────────────────────────────────────────────────

    def _update_hold(self, stable):
        now = time.time()
        if not stable or stable == "none":
            self.hold_gesture, self.hold_start_time, self.gesture_fired = None, 0, False
            if self._shutdown_active:
                self._shutdown_cancelled = True
            # Reset danger timer — hand dropped
            self._danger_gesture    = None
            self._danger_hold_start = 0.0
            return None
        if stable != self.hold_gesture:
            # New gesture — reset hold timer
            self.hold_gesture, self.hold_start_time, self.gesture_fired = stable, now, False
            # Reset danger timer if gesture changed
            if stable != self._danger_gesture:
                self._danger_gesture    = stable
                self._danger_hold_start = now
            return None
        if self.gesture_fired:
            # Already fired — allow re-fire once cooldown has expired
            # This enables continuous repeated actions while holding a gesture
            if not self._cooldown_active():
                self.gesture_fired   = False
                self.hold_start_time = now   # restart hold timer
            return None
        if now - self.hold_start_time >= HOLD_DURATION:
            self.gesture_fired = True
            return stable
        return None

    def _hold_progress(self):
        if not self.hold_gesture or self.hold_gesture == "none":
            return 0.0
        if self.gesture_fired:
            return 1.0
        return min((time.time() - self.hold_start_time) / HOLD_DURATION, 1.0)

    # ── CLAHE preprocessing ────────────────────────────────────────────────────

    def _enhance_frame(self, frame):
        """Apply CLAHE to the L channel of LAB — normalises lighting."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ── ML-based gesture detection ─────────────────────────────────────────────

    def _ml_detect(self, hand_landmarks_list):
        """
        Run the SVM classifier on the current landmarks.
        Returns (gesture_label, confidence) or (None, 0) if model unavailable.

        Handles both old 63-feature models and new 126-feature models.
        If the model expects 63 features, only the first hand is used.
        """
        if self._gesture_model is None:
            return None, 0.0
        feats = _extract_features(hand_landmarks_list)   # always 126
        if feats is None:
            return None, 0.0
        try:
            # Detect what the model expects
            expected = self._gesture_model.named_steps["scaler"].n_features_in_
            if expected == 63:
                feats = feats[:63]   # legacy model — use first hand only
            elif expected != 126:
                return None, 0.0    # unknown model format

            proba = self._gesture_model.predict_proba([feats])[0]
            idx   = int(np.argmax(proba))
            conf  = float(proba[idx])
            label = self._model_classes[idx]
            return label, conf
        except Exception as e:
            print(f"ML detect error: {e}")
            return None, 0.0

    # ── Rule-based fallback detection ─────────────────────────────────────────

    def _fingers_up(self, lm, handedness_label="Right"):
        """
        Robust per-finger extension check using palm-size normalisation.

        Design:
        - Scale reference: wrist → middle-MCP (lm[9]) Euclidean distance.
          This is stable regardless of hand tilt, rotation, or camera distance.
          It never collapses to zero the way a single-axis segment length can.
        - Each of the four fingers uses THREE independent cues (tip vs DIP,
          tip vs PIP, tip vs MCP) and requires at least 2/3 to agree.
          This makes the decision robust to partial bends and noisy landmarks.
        - Thumb uses a 2D projection onto the hand's lateral axis so it works
          correctly for both left and right hands at any tilt angle.
        """
        count = 0

        # ── Palm-size scale ────────────────────────────────────────────
        # Euclidean distance from wrist (0) to middle-finger MCP (9).
        # Stable across all hand orientations and camera distances.
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        palm_scale = (dx*dx + dy*dy) ** 0.5 + 1e-6

        # ── Thumb ──────────────────────────────────────────────────────
        # Build the hand's lateral unit vector (index-MCP → pinky-MCP).
        # The thumb tip should be clearly displaced laterally from the
        # thumb IP joint in the direction away from the other fingers.
        lat_x = lm[17].x - lm[5].x
        lat_y = lm[17].y - lm[5].y
        lat_len = (lat_x*lat_x + lat_y*lat_y) ** 0.5 + 1e-6
        lat_x /= lat_len
        lat_y /= lat_len

        # For both hands, the thumb extends AWAY from the pinky side
        # (i.e., opposite to the index→pinky lateral vector).
        # Negate the projection for both handedness labels.
        tip_dx = lm[4].x - lm[3].x   # tip relative to IP joint
        tip_dy = lm[4].y - lm[3].y
        thumb_proj = -(tip_dx * lat_x + tip_dy * lat_y)

        # Threshold: 8% of palm scale — generous enough for partial extension
        if thumb_proj > palm_scale * 0.08:
            count += 1

        # ── Four fingers ───────────────────────────────────────────────
        # Landmarks:
        #   tip  8  12  16  20
        #   dip  7  11  15  19
        #   pip  6  10  14  18
        #   mcp  5   9  13  17
        tips = [8,  12, 16, 20]
        dips = [7,  11, 15, 19]
        pips = [6,  10, 14, 18]
        mcps = [5,   9, 13, 17]

        for tip, dip, pip, mcp in zip(tips, dips, pips, mcps):
            # Cue 1: tip clearly above DIP  (tight extension check)
            cue1 = lm[tip].y < lm[dip].y - palm_scale * 0.04

            # Cue 2: tip clearly above PIP  (medium extension check)
            cue2 = lm[tip].y < lm[pip].y - palm_scale * 0.02

            # Cue 3: tip clearly above MCP by a meaningful margin
            cue3 = lm[tip].y < lm[mcp].y - palm_scale * 0.10

            # Require at least 2 of 3 cues — majority vote
            if (cue1 + cue2 + cue3) >= 2:
                count += 1

        return count

    def _is_fist(self, lm):
        """All four fingers curled and thumb tucked — uses palm-scale threshold."""
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        palm_scale = (dx*dx + dy*dy) ** 0.5 + 1e-6

        tips = [8,  12, 16, 20]
        dips = [7,  11, 15, 19]
        pips = [6,  10, 14, 18]
        mcps = [5,   9, 13, 17]
        fingers_curled = 0
        for tip, dip, pip, mcp in zip(tips, dips, pips, mcps):
            # Curled: tip is NOT clearly above DIP AND NOT clearly above PIP
            cue1 = lm[tip].y < lm[dip].y - palm_scale * 0.04
            cue2 = lm[tip].y < lm[pip].y - palm_scale * 0.02
            cue3 = lm[tip].y < lm[mcp].y
            if (cue1 + cue2 + cue3) < 2:   # not extended → curled
                fingers_curled += 1
        # Thumb tucked: tip close to index MCP (relative to palm scale)
        tdx = lm[4].x - lm[5].x
        tdy = lm[4].y - lm[5].y
        thumb_tucked = (tdx*tdx + tdy*tdy) ** 0.5 < palm_scale * 0.5
        return fingers_curled >= 4 and thumb_tucked

    def _is_thumbs_up(self, lm):
        thumb_up = lm[4].y < lm[3].y - 0.04 and lm[4].y < lm[2].y - 0.04
        curled = sum(1 for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17])
                     if lm[tip].y > lm[mcp].y)
        return thumb_up and curled >= 3

    def _is_thumbs_down(self, lm):
        thumb_down = lm[4].y > lm[3].y + 0.04 and lm[4].y > lm[2].y + 0.04
        curled = sum(1 for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17])
                     if lm[tip].y > lm[mcp].y)
        return thumb_down and curled >= 3

    def _is_ok(self, lm):
        dx, dy = lm[4].x - lm[8].x, lm[4].y - lm[8].y
        return ((dx*dx + dy*dy)**0.5 < 0.07
                and lm[12].y < lm[10].y - 0.02
                and lm[16].y < lm[14].y - 0.02
                and lm[20].y < lm[18].y - 0.02)

    def _is_peace(self, lm):
        return (lm[8].y  < lm[6].y  - 0.04
                and lm[12].y < lm[10].y - 0.04
                and lm[16].y > lm[14].y
                and lm[20].y > lm[18].y
                and abs(lm[8].x - lm[12].x) > 0.06)

    def _is_open_palm(self, lm):
        """All four fingers clearly extended — uses palm-scale threshold."""
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        palm_scale = (dx*dx + dy*dy) ** 0.5 + 1e-6

        tips = [8,  12, 16, 20]
        dips = [7,  11, 15, 19]
        pips = [6,  10, 14, 18]
        mcps = [5,   9, 13, 17]
        extended = 0
        for tip, dip, pip, mcp in zip(tips, dips, pips, mcps):
            cue1 = lm[tip].y < lm[dip].y - palm_scale * 0.04
            cue2 = lm[tip].y < lm[pip].y - palm_scale * 0.02
            cue3 = lm[tip].y < lm[mcp].y - palm_scale * 0.10  # consistent with _fingers_up
            if (cue1 + cue2 + cue3) >= 2:
                extended += 1
        return extended >= 4

    def _palm_zone(self, lm):
        px, py = lm[9].x, lm[9].y
        if px < self.left_threshold:   return "left"
        if px > self.right_threshold:  return "right"
        if py < self.top_threshold:    return "up"
        if py > self.bottom_threshold: return "down"
        return "center"

    def _total_fingers(self, hand_landmarks_list, handedness_list=None):
        total = 0
        for i, hl in enumerate(hand_landmarks_list):
            label = "Right"
            if handedness_list and i < len(handedness_list):
                raw_label = handedness_list[i].classification[0].label
                # MediaPipe reports handedness from the camera's perspective.
                # Because we flip the frame (cv2.flip, 1), Left↔Right are swapped
                # relative to what the user sees. Invert to get the correct label.
                label = "Left" if raw_label == "Right" else "Right"
            total += self._fingers_up(hl.landmark, label)
        return total

    def _rule_detect(self, hand_landmarks_list, handedness_list=None):
        """Rule-based fallback — same logic as before."""
        if not hand_landmarks_list:
            return "none"
        for hl in hand_landmarks_list:
            lm = hl.landmark
            if self._is_thumbs_up(lm):   return "thumbs_up"
            if self._is_thumbs_down(lm): return "thumbs_down"
            if self._is_ok(lm):          return "ok_sign"
            if self._is_peace(lm):       return "peace_sign"
            if self._is_fist(lm):        return "fist"
        total = self._total_fingers(hand_landmarks_list, handedness_list)
        if total == 5:
            for hl in hand_landmarks_list:
                lm = hl.landmark
                if self._is_open_palm(lm):
                    z = self._palm_zone(lm)
                    return f"count_5_{z}"   # includes "count_5_center"
            return "none"                   # 5 fingers but no open palm
        count_map = {1:"count_1",2:"count_2",3:"count_3",4:"count_4",
                     6:"count_6",7:"count_7",8:"count_8",9:"count_9",10:"count_10"}
        return count_map.get(total, "none")

    # ── Unified detection entry point ──────────────────────────────────────────

    def detect_gesture(self, hand_landmarks_list, handedness_list=None):
        """
        Returns (gesture_label, confidence).
        Uses ML model if available, falls back to rule-based.
        """
        if not hand_landmarks_list:
            return "none", 0.0

        # Try ML model first
        ml_label, ml_conf = self._ml_detect(hand_landmarks_list)
        if ml_label is not None and ml_conf >= NORMAL_CONFIDENCE:
            return ml_label, ml_conf

        # Rule-based fallback (or ML below confidence threshold)
        rule_label = self._rule_detect(hand_landmarks_list, handedness_list)
        # Rule-based gets a fixed "confidence" of 0.80 so it uses the
        # normal stability window, not instant firing
        return rule_label, 0.80

    # ── Shutdown / restart countdowns ─────────────────────────────────────────

    def _shutdown_countdown(self):
        self._shutdown_active   = True
        self._shutdown_cancelled = False
        print("⚠️  Shutdown in 3 seconds... drop hand to cancel")
        for _ in range(3):
            time.sleep(1)
            if self._shutdown_cancelled:
                print("✅ Shutdown cancelled")
                self._shutdown_active = False
                return
        if not self._shutdown_cancelled:
            self.logger.log("SHUTDOWN", "System shutdown triggered by gesture",
                            self.authenticator.current_user)
            import subprocess
            subprocess.run(["shutdown", "/s", "/t", "0"], check=False)
        self._shutdown_active = False

    def _restart_countdown(self):
        self._shutdown_active    = True
        self._shutdown_cancelled = False
        print("⚠️  Restart in 3 seconds... drop hand to cancel")
        for _ in range(3):
            time.sleep(1)
            if self._shutdown_cancelled:
                print("✅ Restart cancelled")
                self._shutdown_active = False
                return
        if not self._shutdown_cancelled:
            self.logger.log("RESTART", "System restart triggered by gesture",
                            self.authenticator.current_user)
            import subprocess
            subprocess.run(["shutdown", "/r", "/t", "0"], check=False)
        self._shutdown_active = False

    # ── Action execution ───────────────────────────────────────────────────────

    # Actions that require minimum confidence + sustained hold before firing.
    # Shutdown/restart need a deliberate 5-second hold so accidental flashes
    # of 10 fingers never trigger them.
    _DANGER_ACTIONS = frozenset({"shutdown", "restart"})
    _DANGER_CONFIDENCE = 0.75   # must be 75%+ confident
    _DANGER_HOLD       = 5.0    # must hold for 5 full seconds

    def execute_action(self, gesture, is_authorized, confidence=0.0):
        if not gesture or gesture == "none":
            return
        if not is_authorized:
            self.logger.log_blocked_action(gesture, "Unauthorized")
            return
        if self._cooldown_active():
            return

        action = self.config_manager.get_gesture_mapping(gesture)
        cmd    = self.config_manager.get_action_command(action) if action else None
        if not cmd:
            return

        # Extra guard for dangerous actions — require minimum confidence and a
        # sustained 5-second hold. Uses _danger_hold_start which tracks
        # continuous hold time independently of the per-fire hold_start_time
        # (which resets every cooldown cycle and would never reach 5s).
        if action in self._DANGER_ACTIONS:
            if confidence < self._DANGER_CONFIDENCE:
                print(f"⚠️  '{action}' blocked — confidence {confidence*100:.0f}% "
                      f"< required {self._DANGER_CONFIDENCE*100:.0f}%")
                return
            held = time.time() - self._danger_hold_start if self._danger_hold_start > 0 else 0
            if held < self._DANGER_HOLD:
                remaining = self._DANGER_HOLD - held
                print(f"⏳  '{action}' — hold for {remaining:.1f}s more...")
                return

        try:
            t, v = cmd["type"], cmd["value"]
            if t == "key":
                pyautogui.press(v)
            elif t == "hotkey":
                pyautogui.hotkey(*v)
            elif t == "custom":
                if v == "swipe_left":
                    pyautogui.hotkey("alt", "left")
                    time.sleep(0.05)
                    pyautogui.press("left")
                elif v == "swipe_right":
                    pyautogui.hotkey("alt", "right")
                    time.sleep(0.05)
                    pyautogui.press("right")
                elif v == "brightness_up":
                    sbc.set_brightness(min(sbc.get_brightness()[0] + 10, 100))
                elif v == "brightness_down":
                    sbc.set_brightness(max(sbc.get_brightness()[0] - 10, 0))
                elif v == "screen_record":
                    state = self.recorder.toggle(self.authenticator.current_user)
                    print(f"🎥 Screen recording {state}: {self.recorder.current_file or ''}")
                elif v == "shutdown":
                    threading.Thread(target=self._shutdown_countdown,
                                     daemon=True).start()
                elif v == "restart":
                    threading.Thread(target=self._restart_countdown,
                                     daemon=True).start()
                elif v in ("scroll_up", "scroll_down"):
                    screen_w, screen_h = pyautogui.size()
                    cx, cy = screen_w // 2, screen_h // 2
                    hwnd   = win32gui.WindowFromPoint((cx, cy))
                    delta  = 120 if v == "scroll_up" else -120
                    wparam = win32api.MAKELONG(0, delta)
                    lparam = win32api.MAKELONG(cx, cy)
                    for _ in range(3):
                        win32api.SendMessage(hwnd, win32con.WM_MOUSEWHEEL,
                                             wparam, lparam)
                    pyautogui.scroll(3 if v == "scroll_up" else -3)

            self.logger.log_gesture_action(gesture, action,
                                           self.authenticator.current_user)
            self.last_action_time = time.time()
        except Exception as e:
            print(f"Action error: {e}")

    # ── UI overlay ─────────────────────────────────────────────────────────────

    GESTURE_LABELS = {
        "fist":          "FIST  ✊",
        "count_1":       "1 FINGER  ☝",
        "count_2":       "2 FINGERS  ✌",
        "count_3":       "3 FINGERS",
        "count_4":       "4 FINGERS  ✊",
        "count_5_center":"PALM CENTER  🖐",
        "count_5_left":  "PALM LEFT  👈",
        "count_5_right": "PALM RIGHT  👉",
        "count_5_up":    "PALM UP  👆",
        "count_5_down":  "PALM DOWN  👇",
        "count_6":       "6 FINGERS",
        "count_7":       "7 FINGERS",
        "count_8":       "8 FINGERS",
        "count_9":       "9 FINGERS",
        "count_10":      "10 FINGERS  🤲",
        "thumbs_up":     "THUMBS UP  👍",
        "thumbs_down":   "THUMBS DOWN  👎",
        "ok_sign":       "OK SIGN  👌",
        "peace_sign":    "PEACE SIGN  ✌",
    }

    ACTION_LABELS = {
        "playpause":      "Play / Pause",
        "volumeup":       "Volume Up",
        "volumedown":     "Volume Down",
        "mute":           "Mute",
        "task_view":      "Task View",
        "screenshot":     "Screenshot",
        "brightness_up":  "Brightness Up",
        "brightness_down":"Brightness Down",
        "swipe_left":     "Prev / Back",
        "swipe_right":    "Next / Forward",
        "scroll_up":      "Scroll Up",
        "scroll_down":    "Scroll Down",
        "screen_record":  "Screen Record ON/OFF",
        "escape":         "Escape",
        "shutdown":       "Shutdown PC (5s)",
        "restart":        "Restart PC (3s)",
        "prev_track":     "Previous Track",
        "next_track":     "Next Track",
        "lock_screen":    "Lock Screen",
        "desktop":        "Show Desktop",
        "close_window":   "Close Window",
    }

    def draw_ui(self, img, gesture, fingers, is_authorized, confidence=0.0):
        h, w = img.shape[:2]
        conf = self.authenticator.last_confidence

        # ── Auth status box ────────────────────────────────────────────
        if is_authorized:
            # Check how long since face was last seen
            now = time.time()
            with self.authenticator._auth_lock:
                last_seen = self.authenticator.last_seen_time
            absent = (now - last_seen) if last_seen > 0 else 0
            timeout = self.authenticator.FACE_TIMEOUT

            if absent > 1.0:
                # Face missing — show warning with countdown
                remaining = max(timeout - absent, 0)
                ratio = remaining / timeout
                bc = (0, 165, 255)   # orange
                cv2.rectangle(img, (5, 5), (310, 115), bc, 2)
                cv2.rectangle(img, (6, 6), (309, 114), (0, 0, 0), -1)
                cv2.putText(img, f"USER: {self.authenticator.current_user}",
                            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 1)
                cv2.putText(img, "FACE NOT DETECTED",
                            (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)
                cv2.putText(img, f"Stopping in {remaining:.1f}s — look at camera",
                            (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bc, 1)
                # Countdown bar
                bw = int(280 * ratio)
                cv2.rectangle(img, (14, 84), (294, 92), (40, 40, 40), -1)
                bar_color = (0, 200, 255) if ratio > 0.5 else (0, 100, 255)
                cv2.rectangle(img, (14, 84), (14 + bw, 92), bar_color, -1)
            else:
                bc = (34, 197, 94)
                cv2.rectangle(img, (5, 5), (310, 105), bc, 2)
                cv2.rectangle(img, (6, 6), (309, 104), (0, 0, 0), -1)
                cv2.putText(img, f"USER: {self.authenticator.current_user}",
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bc, 2)
                mode = "ML" if self._gesture_model else "RULES"
                cv2.putText(img, f"STATUS: AUTHORIZED  [{mode}]",
                            (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bc, 1)
                cv2.putText(img, f"Face conf: {conf:.1f}%",
                            (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 100), 1)
                bw = int(280 * min(conf / 100, 1.0))
                cv2.rectangle(img, (14, 88), (294, 96), (40, 40, 40), -1)
                cv2.rectangle(img, (14, 88), (14 + bw, 96), bc, -1)
        else:
            bc = (60, 60, 220)
            cv2.rectangle(img, (5, 5), (310, 80), bc, 2)
            cv2.rectangle(img, (6, 6), (309, 79), (0, 0, 0), -1)
            cv2.putText(img, "UNAUTHORIZED",
                        (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, bc, 2)
            cv2.putText(img, "GESTURES BLOCKED",
                        (14, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 1)

        # ── Finger count dots ──────────────────────────────────────────
        dot_gap = 20
        dot_start_x = w - 10 - (10 * dot_gap)
        dot_y = 20
        for i in range(10):
            cx_d = dot_start_x + i * dot_gap
            color = (100, 220, 255) if i < fingers else (50, 50, 70)
            cv2.circle(img, (cx_d, dot_y), 8, color, -1 if i < fingers else 1)
        cv2.putText(img, f"{fingers}", (dot_start_x - 28, dot_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # ── Gesture + action + confidence ─────────────────────────────
        if gesture and gesture != "none":
            action_key    = self.config_manager.get_gesture_mapping(gesture)
            gesture_label = self.GESTURE_LABELS.get(
                gesture, gesture.replace("_", " ").upper())
            action_label  = self.ACTION_LABELS.get(
                action_key,
                action_key.replace("_", " ").upper() if action_key else "—")
            conf_str = f" [{confidence*100:.0f}%]" if confidence > 0 else ""
            if is_authorized:
                text = f"  {gesture_label}  →  {action_label}{conf_str}  "
                (tw, _), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (8, 110), (14 + tw, 132), (20, 20, 40), -1)
                cv2.rectangle(img, (8, 110), (14 + tw, 132), (100, 200, 255), 1)
                cv2.putText(img, text, (10, 127),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)
            else:
                cv2.putText(img, f"BLOCKED: {gesture_label}",
                            (10, 127), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (60, 60, 220), 1)

        # ── Hold progress bar ──────────────────────────────────────────
        progress = self._hold_progress()
        if 0 < progress < 1.0:
            bx, by, bw2, bh = 10, 142, 220, 10
            cv2.rectangle(img, (bx, by), (bx + bw2, by + bh), (30, 30, 50), -1)
            fc = (0, 200, 255) if progress < 0.8 else (0, 255, 150)
            cv2.rectangle(img, (bx, by),
                          (bx + int(bw2 * progress), by + bh), fc, -1)
            cv2.rectangle(img, (bx, by), (bx + bw2, by + bh), (80, 80, 100), 1)
            cv2.putText(img, "Hold...", (bx + bw2 + 6, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1)

        # ── Danger hold progress bar (shutdown / restart) ──────────────
        if (self._danger_gesture is not None
                and self._danger_hold_start > 0
                and not self._shutdown_active):
            held    = time.time() - self._danger_hold_start
            dratio  = min(held / self._DANGER_HOLD, 1.0)
            remaining = max(self._DANGER_HOLD - held, 0)
            bx, by, bw2, bh = 10, 158, 220, 10
            cv2.rectangle(img, (bx, by), (bx + bw2, by + bh), (30, 10, 10), -1)
            fc = (0, 80, 255) if dratio < 0.6 else (0, 140, 255) if dratio < 0.9 else (0, 0, 255)
            cv2.rectangle(img, (bx, by),
                          (bx + int(bw2 * dratio), by + bh), fc, -1)
            cv2.rectangle(img, (bx, by), (bx + bw2, by + bh), (100, 40, 40), 1)
            cv2.putText(img, f"SHUTDOWN {remaining:.1f}s",
                        (bx + bw2 + 6, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1)

        # ── Cooldown bar ───────────────────────────────────────────────
        if self._cooldown_active():
            cooldown  = self.config_manager.get_setting("cooldown")
            remaining = cooldown - (time.time() - self.last_action_time)
            ratio = remaining / cooldown
            bx, by, bw2, bh = 10, h - 30, 160, 8
            cv2.rectangle(img, (bx, by), (bx + bw2, by + bh), (30, 30, 50), -1)
            cv2.rectangle(img, (bx, by),
                          (bx + int(bw2 * ratio), by + bh), (0, 165, 255), -1)
            cv2.putText(img, f"Cooldown {remaining:.1f}s",
                        (bx + bw2 + 6, by + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # ── Shutdown overlay ───────────────────────────────────────────
        if self._shutdown_active:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 120), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "!! SHUTDOWN COUNTDOWN !!",
                        (w//2 - 200, h//2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3)
            cv2.putText(img, "DROP HAND TO CANCEL",
                        (w//2 - 165, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        # ── Recording indicator ────────────────────────────────────────
        if self.recorder.is_recording():
            cv2.circle(img, (w - 18, 18), 7, (0, 0, 255), -1)
            cv2.putText(img, "REC", (w - 52, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 60, 255), 1)

        cv2.putText(img, "Q = Quit", (w - 90, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        return img

    # ── Main loop ──────────────────────────────────────────────────────────────

    def start(self):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("❌ Cannot open camera")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        try:
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        WINDOW = "Gesture Control System"
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

        self._stop_event.clear()
        self._lock  = threading.Lock()
        _lock = self._lock
        self._state = _state = {
            "result":          (False, None, 0.0),
            "face_locs":       [],
            "auth_running":    False,
            "frame":           None,
            "frame_annotated": None,
            "gesture":         "none",
            "fingers":         0,
            "confidence":      0.0,
            "stop":            False,
        }

        SCALE     = self.authenticator.SCALE
        scale_inv = 1.0 / SCALE   # float — used to scale face box coords back to full frame
        AUTH_TIMEOUT = 30
        AUTH_SPAWN   = 0.35

        def _auth_worker(frame_copy):
            try:
                success, username, confidence, locs = \
                    self.authenticator._authenticate_frame(
                        frame_copy, return_locations=True)
                with _lock:
                    _state["result"]       = (success, username, confidence)
                    _state["face_locs"]    = locs
                    _state["auth_running"] = False
            except Exception as e:
                print(f"Auth worker: {e}")
                with _lock:
                    _state["auth_running"] = False

        # ── Gesture worker ─────────────────────────────────────────────
        def _gesture_worker():
            FRAME_INTERVAL = 1 / 25

            while True:
                if self._stop_event.is_set():
                    break
                t0 = time.time()

                with _lock:
                    if _state["stop"]:
                        break
                    raw = _state["frame"]
                if raw is None:
                    time.sleep(0.01)
                    continue

                img = raw.copy()

                # ── Per-user model loading ─────────────────────────────
                current_user = self.authenticator.current_user
                if current_user and current_user != self._model_username:
                    self._load_model_for_user(current_user)

                # ── Continuous re-auth ─────────────────────────────────
                auth_ok = self.authenticator.continuous_authenticate(img)
                if not auth_ok:
                    print("❌ Re-authentication failed. Exiting.")
                    with _lock:
                        _state["stop"] = True
                    break
                self.last_auth_status = auth_ok

                # ── CLAHE preprocessing ────────────────────────────────
                # Skip CLAHE when no hand has been seen recently — saves CPU
                if self._frames_no_hand < 10:
                    enhanced = self._enhance_frame(img)
                else:
                    enhanced = img   # no hand present, skip enhancement

                proc   = cv2.resize(enhanced, (480, 360))
                imgRGB = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                results = self.hands.process(imgRGB)

                gesture    = "none"
                confidence = 0.0
                fingers    = 0

                if results.multi_hand_landmarks:
                    self._frames_no_hand = 0   # reset ghost counter

                    handedness = (results.multi_handedness
                                  if hasattr(results, 'multi_handedness')
                                  else None)

                    gesture, confidence = self.detect_gesture(
                        results.multi_hand_landmarks, handedness)
                    fingers = self._total_fingers(
                        results.multi_hand_landmarks, handedness)
                    self._last_gesture    = gesture
                    self._last_confidence = confidence

                    for hl in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(img, hl,
                                               mp_hands.HAND_CONNECTIONS)

                    self._update_history(gesture)

                    # ── Confidence-based firing ────────────────────────
                    if gesture in _INSTANT_GESTURES:
                        # Directional palm: fire on stability, no hold
                        stable = self._stable_gesture()
                        if stable and stable in _INSTANT_GESTURES:
                            self.execute_action(stable, self.last_auth_status,
                                                confidence)
                        self._update_hold(None)

                    elif confidence >= INSTANT_CONFIDENCE:
                        # Very high confidence: fire after just 2 stable frames.
                        # history was already updated above — don't double-add.
                        stable = self._stable_gesture()
                        if stable and stable == gesture:
                            fire = self._update_hold(stable)
                            if fire:
                                self.execute_action(fire, self.last_auth_status,
                                                    confidence)

                    elif confidence >= NORMAL_CONFIDENCE:
                        # Normal confidence: standard stability + hold
                        fire = self._update_hold(self._stable_gesture())
                        if fire and fire != "none":
                            self.execute_action(fire, self.last_auth_status,
                                                confidence)

                    else:
                        # Below threshold: ignore, reset hold
                        self._update_hold(None)

                else:
                    # ── Temporal smoothing (ghost frames) ──────────────
                    self._frames_no_hand += 1
                    if self._frames_no_hand <= GHOST_FRAMES:
                        # Bridge the gap — hold last gesture, keep history warm
                        gesture    = self._last_gesture
                        confidence = self._last_confidence * 0.7
                        self._update_history(gesture)   # keep history stable
                    else:
                        gesture    = "none"
                        confidence = 0.0
                        self.gesture_history.clear()    # reset cleanly
                        self._update_hold(None)

                with _lock:
                    _state["frame_annotated"] = img
                    _state["gesture"]         = gesture
                    _state["fingers"]         = fingers
                    _state["confidence"]      = confidence

                elapsed = time.time() - t0
                sleep_t = FRAME_INTERVAL - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        try:
            # ── Phase 1: Authentication ────────────────────────────────
            start_time = time.time()
            last_spawn = 0

            while True:
                if self._stop_event.is_set():
                    break
                ret, img = cam.read()
                if not ret:
                    continue
                img = cv2.flip(img, 1)
                now = time.time()

                if now - start_time > AUTH_TIMEOUT:
                    print("❌ Authentication timed out.")
                    self.logger.log_auth_failure("Timeout")
                    return

                with _lock:
                    running = _state["auth_running"]
                if not running and (now - last_spawn) >= AUTH_SPAWN:
                    last_spawn = now
                    with _lock:
                        _state["auth_running"] = True
                    threading.Thread(target=_auth_worker,
                                     args=(img.copy(),), daemon=True).start()

                with _lock:
                    success, username, confidence = _state["result"]
                    face_locs = list(_state["face_locs"])

                h, w = img.shape[:2]
                for (top, right, bottom, left) in face_locs:
                    color = (0, 255, 0) if success else (0, 0, 255)
                    cv2.rectangle(img,
                                  (int(left * scale_inv), int(top * scale_inv)),
                                  (int(right * scale_inv), int(bottom * scale_inv)),
                                  color, 2)

                if success:
                    self.authenticator.current_user    = username
                    self.authenticator.last_confidence = confidence
                    self.authenticator._auth_result    = True
                    self.authenticator.consecutive_failures = 0
                    self.authenticator.last_seen_time  = time.time()  # start the absence timer fresh
                    self.config_manager.set_current_user(username)
                    self.logger.log_auth_success(username)
                    self.last_auth_status = True
                    self._load_model_for_user(username)   # load model on auth

                    cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 6)
                    cv2.putText(img, f"WELCOME  {username}",
                                (30, h//2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
                    cv2.putText(img, f"Confidence: {confidence:.1f}%",
                                (30, h//2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    model_status = ("ML model loaded" if self._gesture_model
                                    else "No model — using rules")
                    cv2.putText(img, model_status,
                                (30, h//2 + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                    cv2.imshow(WINDOW, img)
                    cv2.waitKey(800)
                    self.logger.log_session_start(username)
                    break

                remaining = max(0, AUTH_TIMEOUT - (now - start_time))
                overlay = img.copy()
                cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                cv2.putText(img, "SCANNING FOR REGISTERED USER...",
                            (30, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 255), 2)
                if confidence > 0:
                    cv2.putText(img, f"Closest match: {confidence:.0f}%",
                                (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 140, 255), 2)
                cv2.putText(img,
                            f"Timeout in {remaining:.0f}s  |  Press Q to cancel",
                            (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)
                cv2.imshow(WINDOW, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            # ── Phase 2: Gesture loop ──────────────────────────────────
            proc_thread = threading.Thread(target=_gesture_worker, daemon=True)
            proc_thread.start()

            while True:
                ret, raw = cam.read()
                if not ret:
                    continue
                raw = cv2.flip(raw, 1)

                with _lock:
                    _state["frame"] = raw
                    stop       = _state["stop"]
                    annotated  = _state["frame_annotated"]
                    gesture    = _state["gesture"]
                    fingers    = _state["fingers"]
                    confidence = _state["confidence"]

                if stop:
                    break

                base = annotated if annotated is not None else raw
                show = self.draw_ui(base.copy(), gesture, fingers,
                                    self.last_auth_status, confidence)
                cv2.imshow(WINDOW, show)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    with _lock:
                        _state["stop"] = True
                    break

        except Exception as e:
            print(f"Error: {e}")
            self.logger.log_session_end(self.authenticator.current_user,
                                        f"Error: {e}")
        finally:
            with _lock:
                _state["stop"] = True
            cam.release()
            cv2.destroyAllWindows()
            self.logger.log_session_end(self.authenticator.current_user,
                                        "Normal exit")
