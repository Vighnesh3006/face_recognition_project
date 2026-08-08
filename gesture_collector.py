"""
Gesture Data Collector
======================
Records MediaPipe hand landmark samples for each gesture and saves them
to  faces/<username>_gestures.csv  for training.

Usage:
    python gesture_collector.py

Steps:
  1. Select a registered user from the dropdown
  2. Select a gesture from the list
  3. Click  Start Collecting
  4. Perform the gesture in front of the camera — samples auto-capture
  5. Repeat for every gesture
  6. Click  Train Model  when done — trains and saves the model instantly
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from config_manager import ConfigManager
from gesture_trainer import GestureTrainer

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 150   # samples to collect per gesture
COLLECT_FPS         = 10    # how many samples per second to capture
DATA_DIR            = "faces"

GESTURES = [
    "count_1", "count_2", "count_3", "count_4",
    "fist",
    "count_5_center", "count_5_left", "count_5_right",
    "count_5_up", "count_5_down",
    "thumbs_up", "thumbs_down",
    "ok_sign", "peace_sign",
    "count_6", "count_7", "count_8", "count_9", "count_10",
]

GESTURE_INSTRUCTIONS = {
    "count_1":       "☝️  Raise only your INDEX finger",
    "count_2":       "✌️  Raise INDEX + MIDDLE (close together)",
    "count_3":       "🤟  Raise INDEX + MIDDLE + RING",
    "count_4":       "✊  Raise all 4 fingers, thumb tucked",
    "fist":          "✊  Close ALL fingers into a tight fist, thumb tucked",
    "count_5_center":"🖐  Open palm, hand in CENTER of frame",
    "count_5_left":  "👈  Open palm, move hand to LEFT edge",
    "count_5_right": "👉  Open palm, move hand to RIGHT edge",
    "count_5_up":    "👆  Open palm, raise hand to TOP of frame",
    "count_5_down":  "👇  Open palm, lower hand to BOTTOM of frame",
    "thumbs_up":     "👍  Thumbs UP, all other fingers curled",
    "thumbs_down":   "👎  Thumbs DOWN, all other fingers curled",
    "ok_sign":       "👌  Thumb + index form circle, others extended",
    "peace_sign":    "✌️  Index + middle spread apart in V shape",
    "count_6":       "✋  6 fingers (use both hands: 1+5 or 2+4)",
    "count_7":       "✋  7 fingers (both hands: 2+5 or 3+4)",
    "count_8":       "✋  8 fingers (both hands: 3+5 or 4+4)",
    "count_9":       "✋  9 fingers (both hands: 4+5)",
    "count_10":      "🙌  Both hands fully open — 10 fingers",
}

# ── Colours ────────────────────────────────────────────────────────────────────
BG     = "#0f0f1a"
BG2    = "#1a1a2e"
BG3    = "#16213e"
ACCENT = "#7c3aed"
GREEN  = "#10b981"
RED    = "#ef4444"
YELLOW = "#f59e0b"
TEXT   = "#f1f5f9"
TDIM   = "#64748b"


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(hand_landmarks_list, handedness_list=None):
    """
    Extract a normalised 126-value feature vector from hand landmarks.
    Two hands: both encoded and concatenated (126 values).
    One hand:  encoded + zero-padded to 126.
    Normalisation: subtract each hand's wrist, scale by palm size.
    This makes features lighting- and hand-size-independent.
    """
    if not hand_landmarks_list:
        return None

    def _encode(lm_list):
        coords = np.array([[p.x, p.y, p.z] for p in lm_list],
                          dtype=np.float32)
        coords -= coords[0]
        scale   = np.linalg.norm(coords[9]) + 1e-6
        coords /= scale
        return coords.flatten()   # (63,)

    hand0 = _encode(hand_landmarks_list[0].landmark)
    hand1 = (_encode(hand_landmarks_list[1].landmark)
             if len(hand_landmarks_list) >= 2
             else np.zeros(63, dtype=np.float32))

    return np.concatenate([hand0, hand1])   # (126,)


# ── Main collector app ─────────────────────────────────────────────────────────

class GestureCollectorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Data Collector")
        self.root.geometry("1100x720")
        self.root.configure(bg=BG)

        self.config = ConfigManager()
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.camera        = None
        self.cam_active    = False
        self.collecting    = False
        self.sample_count  = 0
        self.current_frame = None
        self._clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._build_ui()
        self._refresh_users()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=ACCENT, height=56)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="✋  Gesture Data Collector",
                 bg=ACCENT, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(
                     side='left', padx=20, pady=12)
        tk.Label(hdr, text=f"Collect {SAMPLES_PER_GESTURE} samples per gesture",
                 bg=ACCENT, fg="#ddd6fe", font=("Segoe UI", 9)).pack(
                     side='right', padx=20)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=20, pady=16)

        # ── LEFT panel ────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG2, width=300)
        left.pack(side='left', fill='y', padx=(0, 14))
        left.pack_propagate(False)
        li = tk.Frame(left, bg=BG2)
        li.pack(fill='both', expand=True, padx=14, pady=14)

        # User selector
        tk.Label(li, text="SELECT USER", bg=BG2, fg=TDIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor='w')
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(li, textvariable=self.user_var,
                                       state='readonly', font=("Segoe UI", 10))
        self.user_combo.pack(fill='x', pady=(4, 12))

        # Gesture list
        tk.Label(li, text="SELECT GESTURE", bg=BG2, fg=TDIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor='w')
        lf = tk.Frame(li, bg=BG3)
        lf.pack(fill='both', expand=True, pady=(4, 0))
        sb = ttk.Scrollbar(lf)
        sb.pack(side='right', fill='y')
        self.gesture_listbox = tk.Listbox(
            lf, yscrollcommand=sb.set,
            bg=BG3, fg=TEXT, selectbackground=ACCENT,
            font=("Segoe UI", 10), borderwidth=0,
            highlightthickness=0, activestyle='none')
        self.gesture_listbox.pack(fill='both', expand=True, padx=2, pady=2)
        sb.config(command=self.gesture_listbox.yview)
        for g in GESTURES:
            self.gesture_listbox.insert(tk.END, f"  {g}")
        self.gesture_listbox.bind('<<ListboxSelect>>', self._on_gesture_select)

        # Progress per gesture
        self.progress_label = tk.Label(li, text="",
                                       bg=BG2, fg=YELLOW,
                                       font=("Segoe UI", 9, "bold"))
        self.progress_label.pack(anchor='w', pady=(8, 0))

        # Buttons
        tk.Frame(li, bg="#334155", height=1).pack(fill='x', pady=10)
        self.collect_btn = tk.Button(
            li, text="▶  Start Collecting",
            bg=GREEN, fg=TEXT, font=("Segoe UI", 10, "bold"),
            relief='flat', cursor='hand2',
            command=self.toggle_collect)
        self.collect_btn.pack(fill='x', pady=(0, 6))

        self.train_btn = tk.Button(
            li, text="🧠  Train Model",
            bg=ACCENT, fg=TEXT, font=("Segoe UI", 10, "bold"),
            relief='flat', cursor='hand2',
            command=self.train_model)
        self.train_btn.pack(fill='x')

        # ── RIGHT panel ───────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side='right', fill='both', expand=True)

        # Camera feed
        cam_border = tk.Frame(right, bg="#334155", padx=2, pady=2)
        cam_border.pack(pady=(0, 10))
        cam_container = tk.Frame(cam_border, bg=BG3, width=640, height=400)
        cam_container.pack()
        cam_container.pack_propagate(False)
        self.cam_label = tk.Label(
            cam_container,
            text="📷\n\nClick  Start Camera  to begin",
            bg=BG3, fg=TDIM, font=("Segoe UI", 11))
        self.cam_label.place(relwidth=1, relheight=1)

        # Instruction label
        self.instruction_label = tk.Label(
            right, text="Select a gesture to see instructions",
            bg=BG, fg=TDIM, font=("Segoe UI", 11),
            wraplength=640, justify='center')
        self.instruction_label.pack(pady=(0, 8))

        # Progress bar
        prog_frame = tk.Frame(right, bg=BG)
        prog_frame.pack(fill='x', pady=(0, 8))
        tk.Label(prog_frame, text="Progress:", bg=BG, fg=TDIM,
                 font=("Segoe UI", 9)).pack(side='left', padx=(0, 8))
        self.prog_bar_canvas = tk.Canvas(prog_frame, height=16,
                                         bg=BG3, highlightthickness=0)
        self.prog_bar_canvas.pack(side='left', fill='x', expand=True)
        self.prog_bar_canvas.configure(width=400)
        self.prog_count_label = tk.Label(prog_frame, text="0 / 0",
                                         bg=BG, fg=TEXT,
                                         font=("Segoe UI", 9, "bold"))
        self.prog_count_label.pack(side='left', padx=(8, 0))

        # Camera controls
        cf = tk.Frame(right, bg=BG)
        cf.pack()
        self.start_cam_btn = tk.Button(
            cf, text="📷  Start Camera",
            bg=ACCENT, fg=TEXT, font=("Segoe UI", 10, "bold"),
            relief='flat', cursor='hand2', padx=14, pady=6,
            command=self.start_camera)
        self.start_cam_btn.pack(side='left', padx=(0, 8))
        self.stop_cam_btn = tk.Button(
            cf, text="⏹  Stop Camera",
            bg=RED, fg=TEXT, font=("Segoe UI", 10, "bold"),
            relief='flat', cursor='hand2', padx=14, pady=6,
            state='disabled', command=self.stop_camera)
        self.stop_cam_btn.pack(side='left')

        # Status bar
        self.status_label = tk.Label(
            self.root, text="Ready — select a user and gesture to begin",
            bg=BG3, fg=TDIM, font=("Segoe UI", 9),
            anchor='w', padx=12, pady=6)
        self.status_label.pack(fill='x', side='bottom')

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _refresh_users(self):
        users = self.config.get_all_users()
        self.user_combo['values'] = users
        if users:
            self.user_combo.set(users[0])

    def _on_gesture_select(self, _=None):
        sel = self.gesture_listbox.curselection()
        if not sel:
            return
        gesture = self.gesture_listbox.get(sel[0]).strip()
        instr = GESTURE_INSTRUCTIONS.get(gesture, gesture)
        self.instruction_label.config(text=instr, fg=TEXT)
        self._update_progress_display(gesture)

    def _csv_path(self, username):
        return os.path.join(DATA_DIR, f"{username}_gestures.csv")

    def _count_samples(self, username, gesture):
        path = self._csv_path(username)
        if not os.path.exists(path):
            return 0
        count = 0
        try:
            with open(path, 'r') as f:
                for row in csv.reader(f):
                    if row and row[0] == gesture:
                        count += 1
        except Exception:
            pass
        return count

    def _update_progress_display(self, gesture=None):
        username = self.user_var.get()
        if not username or not gesture:
            return
        n = self._count_samples(username, gesture)
        pct = min(n / SAMPLES_PER_GESTURE, 1.0)
        self.prog_count_label.config(text=f"{n} / {SAMPLES_PER_GESTURE}")
        self.prog_bar_canvas.delete("all")
        w = self.prog_bar_canvas.winfo_width() or 400
        color = GREEN if pct >= 1.0 else (YELLOW if pct > 0.5 else ACCENT)
        self.prog_bar_canvas.create_rectangle(0, 0, int(w * pct), 16,
                                              fill=color, outline="")
        status = "✅ Complete" if pct >= 1.0 else f"{int(pct*100)}%"
        self.progress_label.config(
            text=f"{gesture}: {n}/{SAMPLES_PER_GESTURE}  {status}",
            fg=GREEN if pct >= 1.0 else YELLOW)

    def _set_status(self, msg, color=None):
        self.status_label.config(text=msg, fg=color or TDIM)

    # ── Camera ─────────────────────────────────────────────────────────────────

    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam_active = True
        self.start_cam_btn.config(state='disabled')
        self.stop_cam_btn.config(state='normal')
        self._update_feed()

    def stop_camera(self):
        self.cam_active = False
        self.collecting = False
        if self.camera:
            self.camera.release()
        self.start_cam_btn.config(state='normal')
        self.stop_cam_btn.config(state='disabled')
        self.collect_btn.config(text="▶  Start Collecting", bg=GREEN)
        self.cam_label.configure(image='',
                                 text="📷\n\nClick  Start Camera  to begin")

    def _update_feed(self):
        if not self.cam_active or not self.camera:
            return
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()

            # CLAHE enhancement for display
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            # Run MediaPipe for live preview
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hl, self.mp_hands.HAND_CONNECTIONS)

            # Collecting indicator
            if self.collecting:
                cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)
                cv2.putText(frame, "RECORDING", (36, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(
                image=Image.fromarray(frame_rgb).resize((640, 400)))
            self.cam_label.imgtk = img
            self.cam_label.configure(image=img, text='')

        self.root.after(33, self._update_feed)   # ~30 fps display

    # ── Collection ─────────────────────────────────────────────────────────────

    def toggle_collect(self):
        if not self.cam_active:
            messagebox.showwarning("Warning", "Start the camera first")
            return
        sel = self.gesture_listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "Select a gesture first")
            return
        username = self.user_var.get()
        if not username:
            messagebox.showwarning("Warning", "Select a user first")
            return

        if self.collecting:
            self.collecting = False
            self.collect_btn.config(text="▶  Start Collecting", bg=GREEN)
            self._set_status("Collection paused")
        else:
            gesture = self.gesture_listbox.get(sel[0]).strip()
            existing = self._count_samples(username, gesture)
            if existing >= SAMPLES_PER_GESTURE:
                if not messagebox.askyesno(
                        "Already complete",
                        f"Already have {existing} samples for '{gesture}'.\n"
                        "Collect more anyway?"):
                    return
            self.collecting = True
            self.collect_btn.config(text="⏸  Pause", bg=YELLOW)
            self._set_status(f"Collecting '{gesture}' for {username}...",
                             color=GREEN)
            threading.Thread(
                target=self._collect_loop,
                args=(username, gesture),
                daemon=True).start()

    def _collect_loop(self, username, gesture):
        path = self._csv_path(username)
        os.makedirs(DATA_DIR, exist_ok=True)
        interval = 1.0 / COLLECT_FPS
        collected = 0

        while self.collecting:
            t0 = time.time()
            frame = self.current_frame
            if frame is None:
                time.sleep(0.05)
                continue

            # CLAHE normalisation before feature extraction
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                feats = extract_features(
                    results.multi_hand_landmarks,
                    results.multi_handedness)
                if feats is not None:
                    row = [gesture] + feats.tolist()
                    with open(path, 'a', newline='') as f:
                        csv.writer(f).writerow(row)
                    collected += 1
                    total = self._count_samples(username, gesture)
                    self.root.after(0, lambda t=total: self._on_sample_collected(
                        username, gesture, t))
                    if total >= SAMPLES_PER_GESTURE:
                        self.collecting = False
                        self.root.after(0, self._on_gesture_complete, gesture)
                        break

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _on_sample_collected(self, username, gesture, total):
        self._update_progress_display(gesture)
        self._set_status(
            f"Collecting '{gesture}': {total}/{SAMPLES_PER_GESTURE} samples",
            color=GREEN)

    def _on_gesture_complete(self, gesture):
        self.collect_btn.config(text="▶  Start Collecting", bg=GREEN)
        self._set_status(
            f"✅ '{gesture}' complete — select next gesture", color=GREEN)
        messagebox.showinfo("Complete",
                            f"✅ {SAMPLES_PER_GESTURE} samples collected for '{gesture}'!")

    # ── Training ───────────────────────────────────────────────────────────────

    def train_model(self):
        username = self.user_var.get()
        if not username:
            messagebox.showwarning("Warning", "Select a user first")
            return
        path = self._csv_path(username)
        if not os.path.exists(path):
            messagebox.showerror("Error",
                                 f"No data found for {username}.\n"
                                 "Collect samples first.")
            return

        self._set_status("Training model...", color=YELLOW)
        self.train_btn.config(state='disabled')

        def _run():
            trainer = GestureTrainer()
            success, msg = trainer.train(username, path)
            self.root.after(0, lambda: self._on_train_done(success, msg, username))

        threading.Thread(target=_run, daemon=True).start()

    def _on_train_done(self, success, msg, username):
        self.train_btn.config(state='normal')
        if success:
            self._set_status(f"✅ Model trained for {username}: {msg}",
                             color=GREEN)
            messagebox.showinfo("Model Trained",
                                f"✅ {username}'s gesture model is ready!\n\n{msg}")
        else:
            self._set_status(f"❌ Training failed: {msg}", color=RED)
            messagebox.showerror("Training Failed", msg)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GestureCollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
