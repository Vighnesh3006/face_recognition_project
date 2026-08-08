import cv2
import numpy as np
import pyautogui
import threading
import time
import os
from datetime import datetime


class ScreenRecorder:
    """
    Records the screen to an AVI file in a background thread.
    Start/stop via gesture or GUI button.
    Saves to recordings/ folder with timestamp filename.
    """

    FPS        = 10          # capture rate — higher = smoother but larger file
    OUTPUT_DIR = "recordings"
    EXT        = ".mp4"

    def __init__(self, logger=None):
        self.logger   = logger
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()
        self.current_file = None
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def is_recording(self):
        with self._lock:
            return self._running

    def toggle(self, user=None):
        """Start if stopped, stop if running. Returns new state string."""
        if self.is_recording():
            self.stop(user)
            return "stopped"
        else:
            self.start(user)
            return "started"

    def start(self, user=None):
        if self.is_recording():
            return
        with self._lock:
            self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        if self.logger:
            self.logger.log("SCREEN_RECORD", "Recording started", user)

    def stop(self, user=None):
        with self._lock:
            self._running = False
        # Don't join — let the recording thread finish on its own.
        # Joining with a 3s timeout would block the GUI/gesture thread.
        if self.logger:
            self.logger.log("SCREEN_RECORD",
                            f"Recording stopped: {self.current_file}", user)

    # ── Recording loop ─────────────────────────────────────────────────

    def _record_loop(self):
        sw, sh = pyautogui.size()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(self.OUTPUT_DIR, f"recording_{timestamp}.mp4")
        self.current_file = filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, self.FPS, (sw, sh))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            filename = os.path.join(self.OUTPUT_DIR, f"recording_{timestamp}.avi")
            self.current_file = filename
            writer = cv2.VideoWriter(filename, fourcc, self.FPS, (sw, sh))

        interval = 1.0 / self.FPS

        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                t0 = time.time()

                # Capture screen
                img = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if not writer.isOpened():
                    break
                writer.write(frame)

                # Sleep remainder of interval
                elapsed = time.time() - t0
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
        finally:
            writer.release()
