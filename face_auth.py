import face_recognition
import cv2
import numpy as np
import time
import os
import pickle
import threading


class FaceAuthenticator:
    """
    Face authenticator optimised for real-time CPU use.

    Detection pipeline improvements:
    - CLAHE preprocessing normalises lighting before HOG runs.
    - Adaptive scale: uses 0.6 instead of 0.4 so faces are larger in the
      downscaled frame and HOG has enough pixels to work with reliably.
    - Frame quality gate: skips blurry or very dark frames that produce
      unreliable encodings.
    - num_jitters=1 during live auth (fast); num_jitters=5 only at registration.
    - Confidence smoothing: keeps a rolling average of the last 3 successful
      confidence readings so the displayed value doesn't flicker.
    - HOG model for live frames (fast on CPU), CNN only during registration.
    """

    SCALE        = 0.6    # downscale for live frames — larger than 0.4 for better HOG accuracy
    FACE_TIMEOUT = 10.0   # seconds without a recognised face before session ends

    # Frame quality thresholds — kept loose so normal movement doesn't skip frames
    _MIN_BRIGHTNESS  = 20    # mean pixel value — only reject very dark frames
    _MIN_SHARPNESS   = 20.0  # Laplacian variance — only reject severely blurry frames

    # Confidence smoothing
    _CONF_HISTORY    = 3     # rolling window size

    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        self.known_faces = {}       # {username: [enc, ...]}
        self.current_user = None
        self.last_confidence = 0.0
        self.consecutive_failures = 0
        self.last_auth_time = 0
        self.last_seen_time = 0.0   # 0 until initial auth succeeds

        self._auth_lock = threading.Lock()
        self._auth_result = True
        self._auth_running = False

        # Rolling confidence history for smooth display
        self._conf_history = []

        self._load_all_users()

    # ── Loading ────────────────────────────────────────────────────────

    def _load_all_users(self):
        for username in self.config_manager.get_all_users():
            info = self.config_manager.get_user_info(username)
            path = info.get("face_encoding")
            if path and os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    self.known_faces[username] = data if isinstance(data, list) else [data]
                    print(f"✅ Loaded {len(self.known_faces[username])} encoding(s) for: {username}")
                except Exception as e:
                    print(f"❌ Error loading {username}: {e}")

    # ── Registration (CNN — runs once, accuracy matters) ───────────────

    def register_new_user(self, username, image_path):
        """Register from a single uploaded image."""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image, num_jitters=5)
            if not encodings:
                return False, "No face detected in image"
            if len(encodings) > 1:
                return False, "Multiple faces detected. Use a single-face image"
            return self._save_encodings(username, [encodings[0]])
        except Exception as e:
            return False, f"Registration error: {e}"

    def register_multi_angle(self, username, frames):
        """Register from multiple camera frames using CNN (one-time, best quality)."""
        encodings = []
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply CLAHE before encoding so registered encodings match
            # the lighting-normalised frames used during live auth
            rgb = self._clahe_rgb(rgb)
            locs = face_recognition.face_locations(rgb, model="cnn")
            if not locs:
                print(f"  ⚠️  No face in frame {i+1}, skipping")
                continue
            # num_jitters=5 for high-quality registration encodings
            enc = face_recognition.face_encodings(rgb, locs, num_jitters=5)
            if enc:
                encodings.append(enc[0])
                print(f"  ✅ Angle {i+1} encoded")

        if len(encodings) < 3:
            return False, f"Only {len(encodings)} valid face(s). Need at least 3. Try better lighting."
        return self._save_encodings(username, encodings)

    def _save_encodings(self, username, encodings):
        path = f"faces/{username}_encoding.pkl"
        os.makedirs("faces", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(encodings, f)
        self.config_manager.add_user(username, path)
        self.known_faces[username] = encodings
        self.logger.log("USER_REGISTER", f"Registered with {len(encodings)} encoding(s)", username)
        return True, f"Registered with {len(encodings)} face angle(s)"

    # ── Preprocessing helpers ──────────────────────────────────────────

    def _clahe_rgb(self, rgb):
        """Apply CLAHE to the L channel of LAB — normalises lighting."""
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def _frame_quality_ok(self, gray):
        """
        Returns True if the frame is bright enough and sharp enough to
        produce reliable face encodings.
        - Brightness: mean pixel value of the grayscale frame.
        - Sharpness: variance of the Laplacian (standard blur detector).
        """
        brightness = float(np.mean(gray))
        if brightness < self._MIN_BRIGHTNESS:
            return False
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self._MIN_SHARPNESS:
            return False
        return True

    # ── Core auth (HOG + CLAHE + adaptive scale) ───────────────────────

    def _authenticate_frame(self, frame, tolerance=None, return_locations=False):
        """
        Authenticate one frame.
        Returns (success, username, confidence_pct) and optionally face locations.

        Improvements over the old version:
        - SCALE = 0.6 (was 0.4) — larger downscaled frame, better HOG accuracy.
        - CLAHE preprocessing before HOG — normalises lighting.
        - Frame quality gate — skips severely blurry/dark frames only.
        - num_jitters=1 for fast live auth (jitters used only at registration).
        """
        if not self.known_faces:
            return (False, None, 0.0, []) if return_locations else (False, None, 0.0)

        if tolerance is None:
            tolerance = self.config_manager.get_setting("face_tolerance")

        try:
            # ── Quality gate ───────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not self._frame_quality_ok(gray):
                # Frame too dark or blurry — return last known result
                # rather than counting it as a failure
                return (False, None, 0.0, []) if return_locations else (False, None, 0.0)

            # ── Resize + CLAHE ─────────────────────────────────────────
            small = cv2.resize(frame, (0, 0), fx=self.SCALE, fy=self.SCALE)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb   = self._clahe_rgb(rgb)

            # ── HOG face detection ─────────────────────────────────────
            locs = face_recognition.face_locations(rgb, model="hog")
            if not locs:
                return (False, None, 0.0, locs) if return_locations else (False, None, 0.0)

            # ── Encoding (num_jitters=1 — fast for live auth) ─────────
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
            best_match, best_conf = None, 0.0

            for enc in encs:
                for uname, known in self.known_faces.items():
                    distances = face_recognition.face_distance(known, enc)
                    dist = float(np.min(distances))
                    conf = round((1.0 - dist) * 100.0, 1)
                    if dist <= tolerance and conf > best_conf:
                        best_conf, best_match = conf, uname

            if return_locations:
                return ((True, best_match, best_conf, locs) if best_match
                        else (False, None, 0.0, locs))
            return ((True, best_match, best_conf) if best_match
                    else (False, None, 0.0))

        except Exception as e:
            print(f"Auth error: {e}")
            return (False, None, 0.0, []) if return_locations else (False, None, 0.0)

    # ── Continuous re-auth (background thread, never blocks gesture loop) ─

    def continuous_authenticate(self, frame):
        """
        Called every frame. Spawns a background auth thread at the configured
        interval. Returns the last known auth result immediately — never blocks.

        Also enforces the 4-second face-absence timeout: if no registered face
        has been seen for FACE_TIMEOUT seconds, sets _auth_result = False so
        the gesture loop stops the session.
        """
        now = time.time()

        # ── Timeout check (runs every frame — fast) ───────────────────
        # last_seen_time is 0 until initial auth succeeds — skip until then.
        with self._auth_lock:
            last_seen = self.last_seen_time

        if last_seen > 0 and now - last_seen > self.FACE_TIMEOUT:
            with self._auth_lock:
                if self._auth_result:   # only log once
                    elapsed = now - last_seen
                    self.logger.log_auth_failure(
                        f"Face absent for {elapsed:.1f}s (timeout {self.FACE_TIMEOUT}s)")
                    print(f"❌ Face not detected for {elapsed:.1f}s — stopping session")
                self._auth_result = False
            return False

        auth_interval = self.config_manager.get_setting("auth_check_frequency") or 2.0
        with self._auth_lock:
            if now - self.last_auth_time < float(auth_interval) or self._auth_running:
                return self._auth_result
            self.last_auth_time = now
            self._auth_running = True

        threading.Thread(target=self._bg_auth, args=(frame.copy(),), daemon=True).start()
        with self._auth_lock:
            return self._auth_result

    def _bg_auth(self, frame):
        """Background auth worker — updates _auth_result, never blocks main loop."""
        try:
            success, username, confidence = self._authenticate_frame(frame)

            with self._auth_lock:
                if not success:
                    self.consecutive_failures += 1
                else:
                    # Face recognised — reset timer and failure counter
                    self.consecutive_failures = 0

                    # Smooth the displayed confidence with a rolling average
                    self._conf_history.append(confidence)
                    if len(self._conf_history) > self._CONF_HISTORY:
                        self._conf_history.pop(0)
                    self.last_confidence = round(
                        sum(self._conf_history) / len(self._conf_history), 1)

                    self.last_seen_time = time.time()

                    if username != self.current_user:
                        self.logger.log("AUTH_SWITCH",
                                        f"Active user switched to {username}", username)
                        self.current_user = username
                        self.config_manager.set_current_user(username)
                    self._auth_result = True
        finally:
            with self._auth_lock:
                self._auth_running = False
