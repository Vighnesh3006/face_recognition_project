"""
Gesture Control System — Modern GUI
Sidebar navigation · Glassmorphism cards · Animated status indicators
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import os
from config_manager import ConfigManager
from logger import ActivityLogger
from face_auth import FaceAuthenticator
from gesture_control import GestureController
from screen_recorder import ScreenRecorder

# ── Design tokens ──────────────────────────────────────────────────────────────
BG          = "#0f0f1a"   # deepest background
BG2         = "#1a1a2e"   # card background
BG3         = "#16213e"   # input / list background
BG4         = "#0d1117"   # sidebar
SURFACE     = "#1e1e35"   # elevated surface
ACCENT      = "#7c3aed"   # primary purple
ACCENT2     = "#a855f7"   # lighter purple
ACCENT_GLOW = "#6d28d9"   # darker purple for depth
CYAN        = "#06b6d4"   # info / highlight
GREEN       = "#10b981"   # success
GREEN2      = "#34d399"   # success light
RED         = "#ef4444"   # danger
RED2        = "#f87171"   # danger light
YELLOW      = "#f59e0b"   # warning
YELLOW2     = "#fbbf24"   # warning light
TEXT        = "#f1f5f9"   # primary text
TEXT_DIM    = "#64748b"   # muted text
TEXT_MID    = "#94a3b8"   # secondary text
BORDER      = "#1e293b"   # subtle border
BORDER2     = "#334155"   # visible border
SIDEBAR_W   = 200

GESTURE_DISPLAY = {
    "fist":          "✊  Fist",
    "count_1":       "☝️   1 Finger",
    "count_2":       "✌️   2 Fingers",
    "count_3":       "🤟  3 Fingers",
    "count_4":       "✊  4 Fingers",
    "count_5_center":"🖐  Palm Center",
    "count_5_left":  "👋  Palm Left",
    "count_5_right": "👋  Palm Right",
    "count_5_up":    "👋  Palm Top",
    "count_5_down":  "👋  Palm Bottom",
    "count_6":       "✋  6 Fingers",
    "count_7":       "✋  7 Fingers",
    "count_8":       "✋  8 Fingers",
    "count_9":       "✋  9 Fingers",
    "count_10":      "✋  10 Fingers",
    "peace_sign":    "🤞  Peace Sign",
    "ok_sign":       "👌  OK Sign",
    "thumbs_up":     "👍  Thumbs Up",
    "thumbs_down":   "👎  Thumbs Down",
}


# ── Custom widget helpers ──────────────────────────────────────────────────────

def _rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Draw a rounded rectangle on a canvas."""
    pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
           x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


class GlowButton(tk.Frame):
    """
    Flat pill button with hover glow effect.
    Uses tk.Frame as the outer widget; canvas is created inside.
    Width/height are NOT passed to Frame.__init__ to avoid Tk 8.6
    internal widget-name corruption on Python 3.10.
    """
    def __init__(self, parent, text, command=None, bg_color=ACCENT,
                 hover_color=ACCENT2, fg=TEXT, width=160, height=38,
                 font=("Segoe UI", 10, "bold"), radius=10):
        # Resolve parent bg safely (ttk widgets don't support cget('bg'))
        try:
            pbg = parent.cget('bg')
        except Exception:
            pbg = BG2

        # Do NOT pass width/height to Frame — it corrupts _w on Tk 8.6/Py3.10
        super().__init__(parent, bg=pbg, cursor="hand2")
        # Set fixed size after init
        self.configure(width=width, height=height)
        self.pack_propagate(False)

        self._bg       = bg_color
        self._hov      = hover_color
        self._fg       = fg
        self._cmd      = command
        self._txt      = text
        self._font     = font
        self._r        = radius
        self._btn_w    = width   # avoid shadowing tk.Frame._w
        self._btn_h    = height
        self._disabled = False

        # Internal canvas — created AFTER Frame is fully initialised
        self._cv = tk.Canvas(self, bg=pbg, highlightthickness=0,
                             cursor="hand2")
        self._cv.configure(width=width, height=height)
        self._cv.place(x=0, y=0, relwidth=1, relheight=1)

        self._draw(bg_color)

        for w in (self, self._cv):
            w.bind("<Enter>",    self._on_enter)
            w.bind("<Leave>",    self._on_leave)
            w.bind("<Button-1>", self._on_click)

    def _draw(self, color):
        self._cv.delete("all")
        _rounded_rect(self._cv, 2, 2, self._btn_w - 2, self._btn_h - 2,
                      self._r, fill=color, outline="")
        self._cv.create_text(
            self._btn_w // 2, self._btn_h // 2,
            text=self._txt,
            fill=self._fg if not self._disabled else TEXT_DIM,
            font=self._font)

    def _on_enter(self, _):
        if not self._disabled:
            self._draw(self._hov)

    def _on_leave(self, _):
        if not self._disabled:
            self._draw(self._bg)

    def _on_click(self, _):
        if not self._disabled and self._cmd:
            self._cmd()

    def config_state(self, state):
        self._disabled = (state == 'disabled')
        self._draw(BG3 if self._disabled else self._bg)

    def config_text(self, text):
        self._txt = text
        self._draw(self._bg if not self._disabled else BG3)

    def config_colors(self, bg_color, hover_color):
        self._bg  = bg_color
        self._hov = hover_color
        self._draw(BG3 if self._disabled else bg_color)



class PillBadge(tk.Canvas):
    """Small coloured pill label."""
    def __init__(self, parent, text, color=ACCENT):
        pill_w = max(len(text) * 7 + 16, 40)
        try:
            parent_bg = parent.cget('bg')
        except Exception:
            parent_bg = BG2
        super().__init__(parent, highlightthickness=0, bg=parent_bg)
        self.configure(width=pill_w, height=20)
        _rounded_rect(self, 1, 1, pill_w - 1, 19, 9, fill=color, outline="")
        self.create_text(pill_w // 2, 10, text=text, fill=TEXT,
                         font=("Segoe UI", 8, "bold"))


class SidebarBtn(tk.Frame):
    """Sidebar navigation item with icon + label + active indicator."""
    def __init__(self, parent, icon, label, command, **kw):
        super().__init__(parent, bg=BG4, cursor="hand2", **kw)
        self._cmd    = command
        self._active = False
        self._icon_lbl = tk.Label(self, text=icon, bg=BG4, fg=TEXT_MID,
                                  font=("Segoe UI", 14), width=3)
        self._icon_lbl.pack(side='left', padx=(12, 4), pady=12)
        self._text_lbl = tk.Label(self, text=label, bg=BG4, fg=TEXT_MID,
                                  font=("Segoe UI", 10, "bold"), anchor='w')
        self._text_lbl.pack(side='left', fill='x', expand=True)
        self._bar = tk.Frame(self, bg=BG4, width=4)
        self._bar.pack(side='right', fill='y')
        for w in (self, self._icon_lbl, self._text_lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>",    self._hover)
            w.bind("<Leave>",    self._leave)

    def _click(self, _):  self._cmd()
    def _hover(self, _):
        if not self._active:
            for w in (self, self._icon_lbl, self._text_lbl):
                w.config(bg=SURFACE)
            self._bar.config(bg=SURFACE)

    def _leave(self, _):
        if not self._active:
            for w in (self, self._icon_lbl, self._text_lbl):
                w.config(bg=BG4)
            self._bar.config(bg=BG4)

    def set_active(self, active):
        self._active = active
        bg   = SURFACE if active else BG4
        fg   = TEXT    if active else TEXT_MID
        bar  = ACCENT  if active else BG4
        for w in (self, self._icon_lbl, self._text_lbl):
            w.config(bg=bg)
        self._icon_lbl.config(fg=ACCENT2 if active else TEXT_MID)
        self._text_lbl.config(fg=fg)
        self._bar.config(bg=bar)


# ── ttk style setup ────────────────────────────────────────────────────────────

def _apply_styles(root):
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".",
                 background=BG, foreground=TEXT, font=("Segoe UI", 10),
                 troughcolor=BG3, borderwidth=0, relief="flat")
    s.configure("TFrame",      background=BG)
    s.configure("TLabel",      background=BG, foreground=TEXT)
    s.configure("TScrollbar",  background=BG3, troughcolor=BG2, arrowcolor=TEXT_DIM,
                borderwidth=0, relief="flat")
    s.configure("TCombobox",   fieldbackground=BG3, foreground=TEXT,
                selectbackground=ACCENT, borderwidth=1, relief="flat",
                arrowcolor=TEXT_MID)
    s.map("TCombobox",
          fieldbackground=[("readonly", BG3)],
          foreground=[("readonly", TEXT)])
    s.configure("TEntry",      fieldbackground=BG3, foreground=TEXT,
                insertcolor=TEXT, borderwidth=0, relief="flat")
    s.configure("TCheckbutton", background=BG2, foreground=TEXT)
    s.map("TCheckbutton",       background=[("active", BG2)])


# ── Main application ───────────────────────────────────────────────────────────

class GestureControlGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Control System")
        self.root.geometry("1340x840")
        self.root.minsize(1100, 700)
        self.root.resizable(True, True)
        _apply_styles(root)

        self.config_manager = ConfigManager()
        self.logger = ActivityLogger(
            enabled=self.config_manager.get_setting("enable_logging"))
        self.authenticator = FaceAuthenticator(
            self.config_manager, self.logger)
        self.recorder   = ScreenRecorder(self.logger)
        self.controller = None

        self.camera_active  = False
        self.camera         = None
        self.current_frame  = None
        self.captured_angles = []
        self.required_angles = 5

        self._pages      = {}
        self._nav_btns   = {}
        self._active_page = None

        self._build_shell()
        self._build_pages()
        self._show_page("control")

        self.refresh_user_list()
        self.refresh_gesture_mappings()

    # ── Shell (sidebar + content area) ────────────────────────────────────────

    def _build_shell(self):
        # ── Sidebar ──────────────────────────────────────────────────
        self.sidebar = tk.Frame(self.root, bg=BG4, width=SIDEBAR_W)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        # Logo block
        logo_block = tk.Frame(self.sidebar, bg=BG4, height=80)
        logo_block.pack(fill='x')
        logo_block.pack_propagate(False)

        logo_cv = tk.Canvas(logo_block, width=40, height=40,
                            bg=BG4, highlightthickness=0)
        logo_cv.create_oval(2, 2, 38, 38, fill=ACCENT, outline=ACCENT2, width=2)
        logo_cv.create_text(20, 20, text="✋", font=("Segoe UI", 16))
        logo_cv.place(x=16, y=20)

        tk.Label(logo_block, text="GestureOS", bg=BG4, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).place(x=64, y=22)
        tk.Label(logo_block, text="v2.0", bg=BG4, fg=TEXT_DIM,
                 font=("Segoe UI", 8)).place(x=66, y=44)

        # Divider
        tk.Frame(self.sidebar, bg=BORDER2, height=1).pack(fill='x', padx=12)

        # Nav items
        nav_items = [
            ("🎮", "Control",  "control"),
            ("👤", "Users",    "users"),
            ("✋", "Gestures", "gestures"),
            ("⚙️", "Settings", "settings"),
            ("📋", "Logs",     "logs"),
        ]
        tk.Label(self.sidebar, text="NAVIGATION", bg=BG4, fg=TEXT_DIM,
                 font=("Segoe UI", 7, "bold")).pack(anchor='w', padx=16, pady=(14, 4))

        for icon, label, key in nav_items:
            btn = SidebarBtn(self.sidebar, icon, label,
                             command=lambda k=key: self._show_page(k))
            btn.pack(fill='x', padx=6, pady=1)
            self._nav_btns[key] = btn

        # Bottom info
        tk.Frame(self.sidebar, bg=BORDER2, height=1).pack(
            fill='x', padx=12, side='bottom', pady=(0, 8))
        tk.Label(self.sidebar, text="Face-Authenticated\nReal-Time · Hands-Free",
                 bg=BG4, fg=TEXT_DIM, font=("Segoe UI", 7),
                 justify='center').pack(side='bottom', pady=4)

        # ── Content area ─────────────────────────────────────────────
        self.content = tk.Frame(self.root, bg=BG)
        self.content.pack(side='left', fill='both', expand=True)

    def _show_page(self, key):
        if self._active_page:
            self._pages[self._active_page].pack_forget()
            self._nav_btns[self._active_page].set_active(False)
        self._pages[key].pack(fill='both', expand=True)
        self._nav_btns[key].set_active(True)
        self._active_page = key

    def _build_pages(self):
        for key in ("control", "users", "gestures", "settings", "logs"):
            f = tk.Frame(self.content, bg=BG)
            self._pages[key] = f
        self._build_control_page()
        self._build_users_page()
        self._build_gestures_page()
        self._build_settings_page()
        self._build_logs_page()

    # ── Shared layout helpers ──────────────────────────────────────────────────

    def _page_header(self, parent, title, subtitle, icon=""):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        row = tk.Frame(hdr, bg=BG)
        row.pack(fill='x')
        if icon:
            tk.Label(row, text=icon, bg=BG, fg=ACCENT2,
                     font=("Segoe UI", 20)).pack(side='left', padx=(0, 10))
        tk.Label(row, text=title, bg=BG, fg=TEXT,
                 font=("Segoe UI", 18, "bold")).pack(side='left', anchor='s')
        tk.Label(hdr, text=subtitle, bg=BG, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor='w', pady=(2, 0))
        tk.Frame(hdr, bg=BORDER2, height=1).pack(fill='x', pady=(12, 0))
        return hdr

    def _card(self, parent, title=None, accent_bar=False, **kw):
        """Glassmorphism-style card with optional left accent bar."""
        outer = tk.Frame(parent, bg=BG2, **kw)
        if accent_bar:
            bar = tk.Frame(outer, bg=ACCENT, width=3)
            bar.pack(side='left', fill='y')
        inner = tk.Frame(outer, bg=BG2)
        inner.pack(fill='both', expand=True, padx=14, pady=12)
        if title:
            title_row = tk.Frame(inner, bg=BG2)
            title_row.pack(fill='x', pady=(0, 10))
            tk.Frame(title_row, bg=ACCENT2, width=3, height=16).pack(
                side='left', padx=(0, 8))
            tk.Label(title_row, text=title, bg=BG2, fg=TEXT,
                     font=("Segoe UI", 10, "bold")).pack(side='left')
        return inner

    def _sep(self, parent, color=BORDER2):
        tk.Frame(parent, bg=color, height=1).pack(fill='x', pady=8)

    def _stat_chip(self, parent, label, value, color=ACCENT2):
        """Small stat display chip."""
        f = tk.Frame(parent, bg=BG3)
        f.pack(side='left', padx=(0, 8), pady=2)
        tk.Label(f, text=label, bg=BG3, fg=TEXT_DIM,
                 font=("Segoe UI", 7, "bold")).pack(padx=10, pady=(6, 0))
        tk.Label(f, text=value, bg=BG3, fg=color,
                 font=("Segoe UI", 11, "bold")).pack(padx=10, pady=(0, 6))
        return f

    def _input_field(self, parent, label, var=None, width=28):
        """Labelled entry with underline style."""
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill='x', pady=(0, 10))
        tk.Label(f, text=label, bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor='w')
        entry_frame = tk.Frame(f, bg=BG3)
        entry_frame.pack(fill='x', pady=(3, 0))
        e = tk.Entry(entry_frame, textvariable=var, width=width,
                     bg=BG3, fg=TEXT, insertbackground=ACCENT2,
                     relief='flat', font=("Segoe UI", 10),
                     highlightthickness=1, highlightbackground=BORDER2,
                     highlightcolor=ACCENT)
        e.pack(fill='x', padx=1, pady=1)
        return e

    # ── Control Page ──────────────────────────────────────────────────────────

    def _build_control_page(self):
        p = self._pages["control"]
        self._page_header(p, "Control Center", "Start a session and monitor gesture activity", "🎮")

        body = tk.Frame(p, bg=BG)
        body.pack(fill='both', expand=True, padx=28, pady=16)

        # ── LEFT column ───────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side='left', fill='y', padx=(0, 16))

        # Status card
        status_card = tk.Frame(left, bg=BG2)
        status_card.pack(fill='x', pady=(0, 12))
        status_inner = tk.Frame(status_card, bg=BG2)
        status_inner.pack(fill='x', padx=16, pady=14)

        dot_row = tk.Frame(status_inner, bg=BG2)
        dot_row.pack(fill='x')
        self.status_dot_canvas = tk.Canvas(dot_row, width=14, height=14,
                                           bg=BG2, highlightthickness=0)
        self.status_dot_canvas.pack(side='left', padx=(0, 8))
        self._status_oval = self.status_dot_canvas.create_oval(
            2, 2, 12, 12, fill=TEXT_DIM, outline="")
        self.control_status = tk.Label(dot_row, text="Ready to start",
                                       bg=BG2, fg=TEXT,
                                       font=("Segoe UI", 11, "bold"))
        self.control_status.pack(side='left')

        self.current_user_label = tk.Label(status_inner, text="Not authenticated",
                                           bg=BG2, fg=TEXT_DIM,
                                           font=("Segoe UI", 9))
        self.current_user_label.pack(anchor='w', pady=(4, 0))

        # Session buttons
        btn_card = self._card(left, "Session Control")
        btn_card.master.pack(fill='x', pady=(0, 12))

        self.start_btn = GlowButton(btn_card, "▶   Start Session",
                                    command=self.start_system,
                                    bg_color=GREEN, hover_color=GREEN2,
                                    width=220, height=42,
                                    font=("Segoe UI", 10, "bold"))
        self.start_btn.pack(pady=(0, 8))

        self.stop_btn = GlowButton(btn_card, "⏹   Stop Session",
                                   command=self.stop_system,
                                   bg_color=BG3, hover_color=RED,
                                   width=220, height=42,
                                   font=("Segoe UI", 10, "bold"))
        self.stop_btn.config_state('disabled')
        self.stop_btn.pack()

        # Recording card
        rec_card = self._card(left, "Screen Recording")
        rec_card.master.pack(fill='x', pady=(0, 12))

        self.record_btn = GlowButton(rec_card, "⏺   Start Recording",
                                     command=self.toggle_recording,
                                     bg_color=ACCENT, hover_color=ACCENT2,
                                     width=220, height=38,
                                     font=("Segoe UI", 10, "bold"))
        self.record_btn.pack(pady=(0, 6))
        self.record_status = tk.Label(rec_card, text="● Not recording",
                                      bg=BG2, fg=TEXT_DIM,
                                      font=("Segoe UI", 8))
        self.record_status.pack(anchor='w')

        # How it works
        how_card = self._card(left, "How It Works")
        how_card.master.pack(fill='x')
        steps = [
            ("1", "Click  Start Session"),
            ("2", "Look at the camera"),
            ("3", "Wait for face auth  (~2s)"),
            ("4", "Use gestures freely"),
            ("5", "Press  Q  to stop"),
        ]
        for num, text in steps:
            row = tk.Frame(how_card, bg=BG2)
            row.pack(fill='x', pady=3)
            c = tk.Canvas(row, width=22, height=22, bg=BG2, highlightthickness=0)
            c.create_oval(1, 1, 21, 21, fill=ACCENT, outline="")
            c.create_text(11, 11, text=num, fill=TEXT,
                          font=("Segoe UI", 8, "bold"))
            c.pack(side='left', padx=(0, 10))
            tk.Label(row, text=text, bg=BG2, fg=TEXT_MID,
                     font=("Segoe UI", 9)).pack(side='left')

        # ── RIGHT column ──────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side='right', fill='both', expand=True)

        # Session dashboard card
        dash_card = self._card(right, "Session Dashboard")
        dash_card.master.pack(fill='x', pady=(0, 12))

        chips_row = tk.Frame(dash_card, bg=BG2)
        chips_row.pack(fill='x', pady=(0, 8))
        self.chip_user   = self._stat_chip(chips_row, "CURRENT USER",  "—",       CYAN)
        self.chip_status = self._stat_chip(chips_row, "STATUS",        "Ready",   GREEN)
        self.chip_conf   = self._stat_chip(chips_row, "CONFIDENCE",    "—",       ACCENT2)

        self._sep(dash_card)
        tk.Label(dash_card,
                 text="⏱  Hold any gesture 0.4s to trigger   ·   🔒  Face re-auth runs in background",
                 bg=BG2, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(anchor='w')

        # Gesture reference card
        ref_card = self._card(right, "Gesture Reference")
        ref_card.master.pack(fill='both', expand=True)

        # Scrollable reference table
        ref_outer = tk.Frame(ref_card, bg=BG2)
        ref_outer.pack(fill='both', expand=True)
        ref_canvas = tk.Canvas(ref_outer, bg=BG2, highlightthickness=0)
        ref_sb = ttk.Scrollbar(ref_outer, orient='vertical',
                               command=ref_canvas.yview)
        ref_sf = tk.Frame(ref_canvas, bg=BG2)
        ref_sf.bind("<Configure>",
                    lambda e: ref_canvas.configure(
                        scrollregion=ref_canvas.bbox("all")))
        ref_canvas.create_window((0, 0), window=ref_sf, anchor='nw')
        ref_canvas.configure(yscrollcommand=ref_sb.set)
        ref_canvas.pack(side='left', fill='both', expand=True)
        ref_sb.pack(side='right', fill='y')

        sections = [
            ("HAND SHAPE", [
                ("✊  Fist",            "Play / Pause"),
                ("✊  4 Fingers",       "Play / Pause"),
                ("☝️   1 Finger",       "Volume Up"),
                ("✌️   2 Fingers",      "Volume Down"),
                ("🤟  3 Fingers",       "Mute"),
                ("🤞  Peace Sign",      "Screenshot"),
                ("👌  OK Sign",         "Screen Record"),
                ("👍  Thumbs Up",       "Brightness Up"),
                ("👎  Thumbs Down",     "Brightness Down"),
            ]),
            ("OPEN PALM POSITION", [
                ("🖐  Palm CENTER",      "Screenshot"),
                ("👋  Palm LEFT",       "Prev / Back"),
                ("👋  Palm RIGHT",      "Next / Forward"),
                ("👋  Palm TOP",        "Scroll Up"),
                ("👋  Palm BOTTOM",     "Scroll Down"),
            ]),
            ("MULTI-FINGER", [
                ("✋  6 Fingers",       "Escape"),
                ("✋  7 Fingers",       "Task View"),
                ("✋  8 Fingers",       "Lock Screen"),
                ("✋  9 Fingers",       "Show Desktop"),
                ("🙌  10 Fingers",      "Shutdown PC (5s)"),
            ]),
        ]

        for sec_title, rows in sections:
            hdr_row = tk.Frame(ref_sf, bg=BG3)
            hdr_row.pack(fill='x', pady=(8, 2))
            tk.Label(hdr_row, text=sec_title, bg=BG3, fg=ACCENT2,
                     font=("Segoe UI", 8, "bold")).pack(
                         side='left', padx=10, pady=5)
            for i, (gesture, action) in enumerate(rows):
                rb = BG2 if i % 2 == 0 else SURFACE
                row = tk.Frame(ref_sf, bg=rb)
                row.pack(fill='x')
                tk.Label(row, text=gesture, bg=rb, fg=TEXT_MID,
                         font=("Segoe UI", 9), width=24,
                         anchor='w').pack(side='left', padx=10, pady=5)
                tk.Label(row, text="→", bg=rb, fg=TEXT_DIM,
                         font=("Segoe UI", 9)).pack(side='left')
                tk.Label(row, text=action, bg=rb, fg=GREEN2,
                         font=("Segoe UI", 9, "bold"),
                         anchor='w').pack(side='left', padx=8, pady=5)

    # ── Users Page ────────────────────────────────────────────────────────────

    def _build_users_page(self):
        p = self._pages["users"]
        self._page_header(p, "User Management",
                          "Register users with multi-angle face capture", "👤")

        body = tk.Frame(p, bg=BG)
        body.pack(fill='both', expand=True, padx=28, pady=16)

        # ── LEFT: user list ───────────────────────────────────────────
        left = tk.Frame(body, bg=BG2, width=240)
        left.pack(side='left', fill='y', padx=(0, 14))
        left.pack_propagate(False)
        left_inner = tk.Frame(left, bg=BG2)
        left_inner.pack(fill='both', expand=True, padx=12, pady=12)

        # Header row
        lh = tk.Frame(left_inner, bg=BG2)
        lh.pack(fill='x', pady=(0, 8))
        tk.Label(lh, text="Registered Users", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side='left')
        self.user_count_badge = tk.Label(lh, text="0", bg=ACCENT,
                                         fg=TEXT, font=("Segoe UI", 8, "bold"),
                                         padx=6, pady=1)
        self.user_count_badge.pack(side='right')

        # Listbox
        lf = tk.Frame(left_inner, bg=BG3)
        lf.pack(fill='both', expand=True)
        sb = ttk.Scrollbar(lf)
        sb.pack(side='right', fill='y')
        self.user_listbox = tk.Listbox(
            lf, yscrollcommand=sb.set,
            bg=BG3, fg=TEXT, selectbackground=ACCENT,
            selectforeground=TEXT, font=("Segoe UI", 10),
            borderwidth=0, highlightthickness=0,
            activestyle='none', relief='flat')
        self.user_listbox.pack(side='left', fill='both', expand=True, padx=2, pady=2)
        sb.config(command=self.user_listbox.yview)
        self.user_listbox.bind('<<ListboxSelect>>', self._on_user_select)

        # User info panel
        self.user_info_frame = tk.Frame(left_inner, bg=BG3)
        self.user_info_frame.pack(fill='x', pady=(8, 0))
        self.user_info_label = tk.Label(
            self.user_info_frame,
            text="Select a user to view details",
            bg=BG3, fg=TEXT_DIM, font=("Segoe UI", 8),
            justify='left', anchor='w', padx=8, pady=8)
        self.user_info_label.pack(fill='x')

        # Action buttons
        self._sep(left_inner)
        bf = tk.Frame(left_inner, bg=BG2)
        bf.pack(fill='x')
        GlowButton(bf, "🗑  Delete User", command=self.delete_user,
                   bg_color=RED, hover_color=RED2,
                   width=120, height=32,
                   font=("Segoe UI", 9, "bold")).pack(side='left', padx=(0, 6))
        GlowButton(bf, "↻", command=self.refresh_user_list,
                   bg_color=BG3, hover_color=SURFACE,
                   width=36, height=32,
                   font=("Segoe UI", 10)).pack(side='left')

        # ── RIGHT: registration wizard ────────────────────────────────
        right = tk.Frame(body, bg=BG2)
        right.pack(side='right', fill='both', expand=True)
        right_inner = tk.Frame(right, bg=BG2)
        right_inner.pack(fill='both', expand=True, padx=16, pady=14)

        # Title
        rh = tk.Frame(right_inner, bg=BG2)
        rh.pack(fill='x', pady=(0, 12))
        tk.Label(rh, text="Register New User", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(side='left')
        PillBadge(rh, "Multi-Angle", ACCENT).pack(side='right', pady=2)

        # Username input
        self.username_entry = self._input_field(right_inner, "FULL NAME / USERNAME")

        # Camera + steps layout
        cam_area = tk.Frame(right_inner, bg=BG2)
        cam_area.pack(fill='both', expand=True)

        # Camera column
        cam_col = tk.Frame(cam_area, bg=BG2)
        cam_col.pack(side='left', fill='both', expand=True)

        CAM_W, CAM_H = 440, 290
        cam_border = tk.Frame(cam_col, bg=BORDER2, padx=2, pady=2)
        cam_border.pack(pady=(0, 6))
        cam_container = tk.Frame(cam_border, bg=BG3, width=CAM_W, height=CAM_H)
        cam_container.pack()
        cam_container.pack_propagate(False)
        self.camera_label = tk.Label(
            cam_container,
            text="📷\n\nCamera Preview\n\nClick  Start Camera  to begin",
            bg=BG3, fg=TEXT_DIM, font=("Segoe UI", 10))
        self.camera_label.place(relwidth=1, relheight=1)

        # Progress dots
        dots_row = tk.Frame(cam_col, bg=BG2)
        dots_row.pack()
        tk.Label(dots_row, text="Progress:", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 8)).pack(side='left', padx=(0, 8))
        self.angle_dots = []
        for i in range(5):
            c = tk.Canvas(dots_row, width=18, height=18,
                          bg=BG2, highlightthickness=0)
            c.create_oval(2, 2, 16, 16, fill=BG3, outline=BORDER2, width=1,
                          tags="dot")
            c.pack(side='left', padx=4)
            self.angle_dots.append(c)

        self.angle_progress_label = tk.Label(
            cam_col, text="0 / 5 angles captured",
            bg=BG2, fg=YELLOW, font=("Segoe UI", 9, "bold"))
        self.angle_progress_label.pack(pady=(4, 0))

        # Camera control buttons
        cf = tk.Frame(cam_col, bg=BG2)
        cf.pack(pady=8)
        self.start_cam_btn = GlowButton(cf, "📷  Start Camera",
                                        command=self.start_camera,
                                        bg_color=ACCENT, hover_color=ACCENT2,
                                        width=140, height=34,
                                        font=("Segoe UI", 9, "bold"))
        self.start_cam_btn.pack(side='left', padx=(0, 6))
        self.capture_btn = GlowButton(cf, "📸  Capture",
                                      command=self.capture_angle,
                                      bg_color=CYAN, hover_color="#22d3ee",
                                      width=100, height=34,
                                      font=("Segoe UI", 9, "bold"))
        self.capture_btn.config_state('disabled')
        self.capture_btn.pack(side='left', padx=(0, 6))
        self.stop_cam_btn = GlowButton(cf, "⏹  Stop",
                                       command=self.stop_camera,
                                       bg_color=RED, hover_color=RED2,
                                       width=80, height=34,
                                       font=("Segoe UI", 9, "bold"))
        self.stop_cam_btn.config_state('disabled')
        self.stop_cam_btn.pack(side='left')

        # Steps guide column
        steps_col = tk.Frame(cam_area, bg=BG2, width=210)
        steps_col.pack(side='right', fill='y', padx=(14, 0))
        steps_col.pack_propagate(False)

        tk.Label(steps_col, text="CAPTURE GUIDE", bg=BG2, fg=ACCENT2,
                 font=("Segoe UI", 8, "bold")).pack(anchor='w', pady=(0, 8))

        angle_steps = [
            ("1", "😐", "Face Front",   "Look straight at camera"),
            ("2", "😶", "Slight Left",  "Turn head slightly left"),
            ("3", "😶", "Slight Right", "Turn head slightly right"),
            ("4", "🙂", "Tilt Up",      "Tilt chin slightly up"),
            ("5", "🙂", "Tilt Down",    "Tilt chin slightly down"),
        ]
        self.step_labels = []
        for num, emoji, title, hint in angle_steps:
            step_f = tk.Frame(steps_col, bg=BG3)
            step_f.pack(fill='x', pady=3)
            num_c = tk.Canvas(step_f, width=24, height=24,
                              bg=BG3, highlightthickness=0)
            num_c.create_oval(2, 2, 22, 22, fill=ACCENT, outline="")
            num_c.create_text(12, 12, text=num, fill=TEXT,
                              font=("Segoe UI", 8, "bold"))
            num_c.pack(side='left', padx=(8, 6), pady=8)
            inner = tk.Frame(step_f, bg=BG3)
            inner.pack(side='left', fill='x', expand=True, pady=6)
            lbl = tk.Label(inner, text=f"{emoji}  {title}", bg=BG3, fg=TEXT_DIM,
                           font=("Segoe UI", 9, "bold"), anchor='w')
            lbl.pack(anchor='w')
            tk.Label(inner, text=hint, bg=BG3, fg=TEXT_DIM,
                     font=("Segoe UI", 8), anchor='w').pack(anchor='w')
            self.step_labels.append(lbl)

        # Register / Upload row
        self._sep(right_inner)
        bottom = tk.Frame(right_inner, bg=BG2)
        bottom.pack(fill='x')
        self.register_btn = GlowButton(bottom, "✅  Register User",
                                       command=self.register_multi_angle,
                                       bg_color=GREEN, hover_color=GREEN2,
                                       width=180, height=38,
                                       font=("Segoe UI", 10, "bold"))
        self.register_btn.config_state('disabled')
        self.register_btn.pack(side='left', padx=(0, 12))
        tk.Label(bottom, text="or", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side='left', padx=(0, 12))
        GlowButton(bottom, "📁  Upload Photo",
                   command=self.upload_image,
                   bg_color=BG3, hover_color=SURFACE,
                   width=150, height=38,
                   font=("Segoe UI", 10, "bold")).pack(side='left')

    # ── Gestures Page ─────────────────────────────────────────────────────────

    def _build_gestures_page(self):
        p = self._pages["gestures"]
        self._page_header(p, "Gesture Mappings",
                          "Assign an action to each gesture — changes save immediately", "✋")

        outer = tk.Frame(p, bg=BG)
        outer.pack(fill='both', expand=True, padx=28, pady=12)

        # Column headers
        hdr = tk.Frame(outer, bg=BG3)
        hdr.pack(fill='x', pady=(0, 2))
        tk.Label(hdr, text="  GESTURE", bg=BG3, fg=ACCENT2,
                 font=("Segoe UI", 8, "bold"), width=36,
                 anchor='w').pack(side='left', padx=8, pady=8)
        tk.Label(hdr, text="MAPPED ACTION", bg=BG3, fg=ACCENT2,
                 font=("Segoe UI", 8, "bold"), width=28,
                 anchor='w').pack(side='left', padx=8, pady=8)

        # Scrollable rows
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas, bg=BG)
        sf.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.gesture_widgets = {}
        actions = list(self.config_manager.config["action_commands"].keys())

        for i, (gesture, current) in enumerate(
                self.config_manager.config["gesture_mappings"].items()):
            rb = BG2 if i % 2 == 0 else SURFACE
            row = tk.Frame(sf, bg=rb)
            row.pack(fill='x', pady=1)

            # Gesture icon + name
            display = GESTURE_DISPLAY.get(
                gesture, gesture.replace('_', ' ').title())
            tk.Label(row, text=display, bg=rb, fg=TEXT,
                     font=("Segoe UI", 10), width=36,
                     anchor='w').pack(side='left', padx=12, pady=8)

            # Combobox
            combo = ttk.Combobox(row, values=actions, state='readonly',
                                 width=28, font=("Segoe UI", 10))
            combo.set(current)
            combo.pack(side='left', padx=10, pady=6)
            self.gesture_widgets[gesture] = combo

        # Save button
        save_row = tk.Frame(sf, bg=BG)
        save_row.pack(fill='x', pady=14)
        GlowButton(save_row, "💾  Save All Mappings",
                   command=self.save_gesture_mappings,
                   bg_color=GREEN, hover_color=GREEN2,
                   width=200, height=38,
                   font=("Segoe UI", 10, "bold")).pack(side='left', padx=10)

    # ── Settings Page ─────────────────────────────────────────────────────────

    def _build_settings_page(self):
        p = self._pages["settings"]
        self._page_header(p, "System Settings",
                          "Tune detection sensitivity, cooldown, and auth parameters", "⚙️")

        outer = tk.Frame(p, bg=BG)
        outer.pack(fill='both', expand=True, padx=28, pady=12)

        # Two-column layout
        left_col  = tk.Frame(outer, bg=BG)
        right_col = tk.Frame(outer, bg=BG)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        right_col.pack(side='right', fill='both', expand=True, padx=(10, 0))

        self.setting_widgets = {}

        descriptions = {
            "cooldown":             ("⏱", "Cooldown",             "Seconds between gesture actions",          "float"),
            "gesture_stability":    ("🎯", "Gesture Stability",    "Frames gesture must be stable",            "int"),
            "face_tolerance":       ("🔍", "Face Tolerance",       "Match tolerance — lower = stricter",       "float"),
            "auth_check_frequency": ("🔄", "Auth Frequency",       "Seconds between background auth checks",   "float"),
            "max_auth_failures":    ("🚫", "Max Auth Failures",    "Consecutive failures before session ends", "int"),
            "enable_logging":       ("📝", "Enable Logging",       "Write activity to log file",               "bool"),
            "detection_confidence": ("👁", "Detection Confidence", "MediaPipe hand detection confidence",      "float"),
            "tracking_confidence":  ("📍", "Tracking Confidence",  "MediaPipe hand tracking confidence",       "float"),
        }

        items = list(self.config_manager.config["settings"].items())
        left_items  = items[:4]
        right_items = items[4:]

        for col_frame, col_items in ((left_col, left_items), (right_col, right_items)):
            for key, value in col_items:
                icon, label, desc, dtype = descriptions.get(
                    key, ("•", key.replace('_', ' ').title(), "", "str"))

                card = tk.Frame(col_frame, bg=BG2)
                card.pack(fill='x', pady=(0, 8))
                inner = tk.Frame(card, bg=BG2)
                inner.pack(fill='x', padx=14, pady=10)

                # Left accent bar
                tk.Frame(card, bg=ACCENT, width=3).place(x=0, y=0, relheight=1)

                top_row = tk.Frame(inner, bg=BG2)
                top_row.pack(fill='x')
                tk.Label(top_row, text=f"{icon}  {label}", bg=BG2, fg=TEXT,
                         font=("Segoe UI", 10, "bold")).pack(side='left')

                tk.Label(inner, text=desc, bg=BG2, fg=TEXT_DIM,
                         font=("Segoe UI", 8)).pack(anchor='w', pady=(2, 6))

                if dtype == "bool":
                    var = tk.BooleanVar(value=value)
                    toggle_f = tk.Frame(inner, bg=BG2)
                    toggle_f.pack(anchor='w')
                    cb = tk.Checkbutton(toggle_f, variable=var,
                                        bg=BG2, fg=TEXT,
                                        activebackground=BG2,
                                        selectcolor=ACCENT,
                                        font=("Segoe UI", 9))
                    cb.pack(side='left')
                    tk.Label(toggle_f, text="Enabled", bg=BG2, fg=TEXT_MID,
                             font=("Segoe UI", 9)).pack(side='left', padx=4)
                else:
                    var = tk.StringVar(value=str(value))
                    e = tk.Entry(inner, textvariable=var, width=14,
                                 bg=BG3, fg=TEXT, insertbackground=ACCENT2,
                                 relief='flat', font=("Segoe UI", 10),
                                 highlightthickness=1,
                                 highlightbackground=BORDER2,
                                 highlightcolor=ACCENT)
                    e.pack(anchor='w')
                self.setting_widgets[key] = var

        # Save button
        save_row = tk.Frame(outer, bg=BG)
        save_row.pack(fill='x', pady=16, side='bottom')
        GlowButton(save_row, "💾  Save Settings",
                   command=self.save_settings,
                   bg_color=GREEN, hover_color=GREEN2,
                   width=180, height=40,
                   font=("Segoe UI", 10, "bold")).pack(side='left')

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Logs Page ─────────────────────────────────────────────────────────────

    def _build_logs_page(self):
        p = self._pages["logs"]
        self._page_header(p, "Activity Logs",
                          "Real-time log of auth events, gestures, and session activity", "📋")

        outer = tk.Frame(p, bg=BG)
        outer.pack(fill='both', expand=True, padx=28, pady=12)

        # Toolbar
        toolbar = tk.Frame(outer, bg=BG)
        toolbar.pack(fill='x', pady=(0, 8))
        GlowButton(toolbar, "↻  Refresh",
                   command=self.refresh_logs,
                   bg_color=ACCENT, hover_color=ACCENT2,
                   width=110, height=32,
                   font=("Segoe UI", 9, "bold")).pack(side='left', padx=(0, 8))
        GlowButton(toolbar, "🗑  Clear",
                   command=self.clear_logs,
                   bg_color=RED, hover_color=RED2,
                   width=100, height=32,
                   font=("Segoe UI", 9, "bold")).pack(side='left')
        self.log_count_label = tk.Label(toolbar, text="",
                                        bg=BG, fg=TEXT_DIM,
                                        font=("Segoe UI", 8))
        self.log_count_label.pack(side='right')

        # Log text area
        log_frame = tk.Frame(outer, bg=BG3,
                             highlightthickness=1,
                             highlightbackground=BORDER2)
        log_frame.pack(fill='both', expand=True)
        sb = ttk.Scrollbar(log_frame)
        sb.pack(side='right', fill='y')
        self.log_text = tk.Text(
            log_frame, yscrollcommand=sb.set,
            bg=BG3, fg=TEXT_MID, font=("Consolas", 9),
            wrap='word', borderwidth=0, highlightthickness=0,
            insertbackground=TEXT, padx=10, pady=8,
            selectbackground=ACCENT, selectforeground=TEXT)
        self.log_text.pack(side='left', fill='both', expand=True)
        sb.config(command=self.log_text.yview)

        # Colour tags for log levels
        self.log_text.tag_config("AUTH",    foreground=GREEN2)
        self.log_text.tag_config("GESTURE", foreground=CYAN)
        self.log_text.tag_config("BLOCKED", foreground=RED2)
        self.log_text.tag_config("SESSION", foreground=ACCENT2)
        self.log_text.tag_config("CONFIG",  foreground=YELLOW2)
        self.log_text.tag_config("RECORD",  foreground=YELLOW)

        self.refresh_logs()

    # ── User management logic ─────────────────────────────────────────────────

    def refresh_user_list(self):
        self.user_listbox.delete(0, tk.END)
        users = self.config_manager.get_all_users()
        for user in users:
            self.user_listbox.insert(tk.END, f"  {user}")
        if hasattr(self, 'user_count_badge'):
            self.user_count_badge.config(text=str(len(users)))

    def _on_user_select(self, event):
        sel = self.user_listbox.curselection()
        if not sel:
            return
        username = self.user_listbox.get(sel[0]).strip()
        info = self.config_manager.get_user_info(username)
        if not info:
            return
        last = (info.get('last_login') or 'Never')[:10]
        self.user_info_label.config(
            text=(f"  User      :  {username}\n"
                  f"  Created   :  {info.get('created_at', 'N/A')[:10]}\n"
                  f"  Last Login:  {last}\n"
                  f"  Sessions  :  {info.get('total_sessions', 0)}"),
            fg=TEXT_MID)

    def delete_user(self):
        sel = self.user_listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "Please select a user to delete")
            return
        username = self.user_listbox.get(sel[0]).strip()
        if messagebox.askyesno("Confirm Delete",
                               f"Permanently delete user '{username}'?"):
            if self.config_manager.remove_user(username):
                messagebox.showinfo("Deleted", f"User '{username}' removed.")
                self.refresh_user_list()
            else:
                messagebox.showerror("Error", "Failed to delete user")

    # ── Camera / Registration logic ───────────────────────────────────────────

    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        self.camera_active   = True
        self.captured_angles = []
        self.angle_progress_label.config(
            text="0 / 5 angles captured  —  Next: FRONT", fg=YELLOW)
        for c in self.angle_dots:
            c.delete("dot")
            c.create_oval(2, 2, 16, 16, fill=BG3, outline=BORDER2,
                          width=1, tags="dot")
        for i, lbl in enumerate(self.step_labels):
            lbl.config(fg=YELLOW if i == 0 else TEXT_DIM)
        self.start_cam_btn.config_state('disabled')
        self.capture_btn.config_state('normal')
        self.register_btn.config_state('disabled')
        self.stop_cam_btn.config_state('normal')
        self._update_camera_feed()

    def capture_angle(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame available")
            return
        self.captured_angles.append(
            cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
        count = len(self.captured_angles)

        for i, c in enumerate(self.angle_dots):
            c.delete("dot")
            if i < count:
                c.create_oval(2, 2, 16, 16, fill=GREEN, outline="",
                              tags="dot")
            else:
                c.create_oval(2, 2, 16, 16, fill=BG3, outline=BORDER2,
                              width=1, tags="dot")

        for i, lbl in enumerate(self.step_labels):
            if i < count:
                lbl.config(fg=GREEN2)
            elif i == count:
                lbl.config(fg=YELLOW)
            else:
                lbl.config(fg=TEXT_DIM)

        hints = ["front", "slight left", "slight right", "tilt up", "tilt down"]
        if count < self.required_angles:
            self.angle_progress_label.config(
                text=f"{count} / {self.required_angles} captured  —  Next: {hints[count].upper()}",
                fg=YELLOW)
        else:
            self.register_btn.config_state('normal')
            self.angle_progress_label.config(
                text="✅  All 5 angles captured — ready to register!", fg=GREEN2)

    def register_multi_angle(self):
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showwarning("Warning", "Please enter a username")
            return
        if username in self.config_manager.get_all_users():
            messagebox.showwarning("Warning", "Username already exists")
            return
        if len(self.captured_angles) < 3:
            messagebox.showwarning("Warning", "Need at least 3 captured angles")
            return
        self.register_btn.config_state('disabled')
        self.angle_progress_label.config(text="⏳  Processing encodings...",
                                         fg=YELLOW)

        def _run():
            success, msg = self.authenticator.register_multi_angle(
                username, self.captured_angles)
            self.root.after(0, lambda: self._on_register_done(success, msg, username))

        threading.Thread(target=_run, daemon=True).start()

    def _on_register_done(self, success, message, username):
        if success:
            messagebox.showinfo("Registered",
                                f"User '{username}' registered successfully!")
            self.username_entry.delete(0, tk.END)
            self.captured_angles = []
            self.angle_progress_label.config(
                text="0 / 5 angles captured", fg=YELLOW)
            for c in self.angle_dots:
                c.delete("dot")
                c.create_oval(2, 2, 16, 16, fill=BG3, outline=BORDER2,
                              width=1, tags="dot")
            for lbl in self.step_labels:
                lbl.config(fg=TEXT_DIM)
            self.register_btn.config_state('disabled')
            self.refresh_user_list()
            self.stop_camera()
        else:
            self.register_btn.config_state('normal')
            messagebox.showerror("Registration Failed", message)

    def _update_camera_feed(self):
        if self.camera_active and self.camera:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (440, 290))
                self.current_frame = frame.copy()
                img = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.camera_label.imgtk = img
                self.camera_label.configure(image=img, text='')
            self.root.after(10, self._update_camera_feed)

    def stop_camera(self):
        self.camera_active = False
        if self.camera:
            self.camera.release()
        self.start_cam_btn.config_state('normal')
        self.capture_btn.config_state('disabled')
        self.register_btn.config_state('disabled')
        self.stop_cam_btn.config_state('disabled')
        self.camera_label.configure(
            image='',
            text="📷\n\nCamera Preview\n\nClick  Start Camera  to begin")

    def upload_image(self):
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showwarning("Warning", "Please enter a username")
            return
        if username in self.config_manager.get_all_users():
            messagebox.showwarning("Warning", "Username already exists")
            return
        path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        success, msg = self.authenticator.register_new_user(username, path)
        if success:
            messagebox.showinfo("Success", f"User '{username}' registered!")
            self.username_entry.delete(0, tk.END)
            self.refresh_user_list()
        else:
            messagebox.showerror("Error", msg)

    # ── Gesture / Settings logic ──────────────────────────────────────────────

    def save_gesture_mappings(self):
        for gesture, combo in self.gesture_widgets.items():
            action = combo.get()
            self.config_manager.set_gesture_mapping(gesture, action)
            self.logger.log("CONFIG",
                            f"Gesture '{gesture}' mapped to '{action}'",
                            self.authenticator.current_user)
        messagebox.showinfo("Saved", "Gesture mappings saved!")

    def refresh_gesture_mappings(self):
        if hasattr(self, 'gesture_widgets'):
            for gesture, combo in self.gesture_widgets.items():
                val = self.config_manager.get_gesture_mapping(gesture)
                if val:
                    combo.set(val)

    def save_settings(self):
        int_keys   = {"gesture_stability", "max_auth_failures"}
        float_keys = {"cooldown", "face_tolerance", "detection_confidence",
                      "tracking_confidence", "auth_check_frequency"}
        for key, var in self.setting_widgets.items():
            val = var.get()
            try:
                if key in int_keys:
                    val = int(val)
                elif key in float_keys:
                    val = float(val)
                elif key == "enable_logging":
                    val = bool(val)
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    f"Please enter a valid number for '{key.replace('_', ' ')}'.")
                return
            self.config_manager.set_setting(key, val)
            self.logger.log_config_change(key, val, self.authenticator.current_user)
        messagebox.showinfo("Saved", "Settings saved!")

    # ── Control logic ─────────────────────────────────────────────────────────

    def start_system(self):
        if self.controller is not None:
            return
        if not self.config_manager.get_all_users():
            messagebox.showwarning("No Users",
                                   "Please register a user first (Users tab)")
            return
        self.start_btn.config_state('disabled')
        self.stop_btn.config_state('normal')
        self._set_status("Authenticating...", YELLOW)

        def _run():
            controller = GestureController(
                self.config_manager, self.authenticator,
                self.logger, recorder=self.recorder)
            self.controller = controller
            controller.start()
            self.root.after(0, self._on_system_stopped)

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(500, self._poll_auth_status)

    def _on_close(self):
        if self.controller:
            self.controller.stop()
        if self.recorder.is_recording():
            self.recorder.stop(self.authenticator.current_user)
        self.root.destroy()

    def toggle_recording(self):
        state = self.recorder.toggle(self.authenticator.current_user)
        if state == "started":
            self.record_btn.config_text("⏹   Stop Recording")
            self.record_btn.config_colors(RED, RED2)
            self.record_status.config(
                text=f"● REC  {os.path.basename(self.recorder.current_file or '')}",
                fg=RED2)
        else:
            self.record_btn.config_text("⏺   Start Recording")
            self.record_btn.config_colors(ACCENT, ACCENT2)
            self.record_status.config(
                text=f"Saved: {os.path.basename(self.recorder.current_file or '')}",
                fg=GREEN2)

    def _poll_auth_status(self):
        if self.controller is None:
            return
        user = self.authenticator.current_user
        if user:
            self.current_user_label.config(
                text=f"Logged in as:  {user}", fg=GREEN2)
            self.chip_user.winfo_children()[1].config(text=user)
            self.chip_status.winfo_children()[1].config(text="Running", fg=GREEN2)
            conf = getattr(self.authenticator, 'last_confidence', 0)
            self.chip_conf.winfo_children()[1].config(
                text=f"{conf:.0f}%")
            self._set_status(f"Running  ·  {user}", GREEN)
            self.root.after(1000, self._poll_recording_status)
        else:
            self.chip_status.winfo_children()[1].config(
                text="Authenticating...", fg=YELLOW)
            self._set_status("Authenticating...", YELLOW)
            self.root.after(500, self._poll_auth_status)

    def _poll_recording_status(self):
        if self.recorder.is_recording():
            self.record_btn.config_text("⏹   Stop Recording")
            self.record_btn.config_colors(RED, RED2)
            self.record_status.config(
                text=f"● REC  {os.path.basename(self.recorder.current_file or '')}",
                fg=RED2)
        else:
            self.record_btn.config_text("⏺   Start Recording")
            self.record_btn.config_colors(ACCENT, ACCENT2)
            if self.recorder.current_file:
                self.record_status.config(
                    text=f"Saved: {os.path.basename(self.recorder.current_file)}",
                    fg=GREEN2)
        if self.authenticator.current_user:
            self.root.after(1000, self._poll_recording_status)

    def _set_status(self, text, color):
        self.control_status.config(text=text, fg=color)
        self.status_dot_canvas.itemconfig(self._status_oval, fill=color)

    def stop_system(self):
        self._set_status("Stopping...", YELLOW)
        if self.controller:
            self.controller.stop()
        self.chip_status.winfo_children()[1].config(text="Stopping...", fg=YELLOW)

    def _on_system_stopped(self):
        self.start_btn.config_state('normal')
        self.stop_btn.config_state('disabled')
        self.current_user_label.config(text="Not authenticated", fg=TEXT_DIM)
        self.chip_user.winfo_children()[1].config(text="—")
        self.chip_status.winfo_children()[1].config(text="Ready", fg=GREEN)
        self.chip_conf.winfo_children()[1].config(text="—")
        self._set_status("Ready to start", TEXT_DIM)
        self.authenticator.current_user = None
        self.controller = None
        self.refresh_logs()

    # ── Logs logic ────────────────────────────────────────────────────────────

    def refresh_logs(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        lines = self.logger.get_recent_logs(200)
        for line in lines:
            tag = None
            if "[AUTH"    in line: tag = "AUTH"
            elif "[GESTURE" in line: tag = "GESTURE"
            elif "[BLOCKED" in line: tag = "BLOCKED"
            elif "[SESSION" in line: tag = "SESSION"
            elif "[CONFIG"  in line: tag = "CONFIG"
            elif "[SCREEN"  in line: tag = "RECORD"
            if tag:
                self.log_text.insert(tk.END, line, tag)
            else:
                self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        if hasattr(self, 'log_count_label'):
            self.log_count_label.config(text=f"{len(lines)} entries")

    def clear_logs(self):
        if messagebox.askyesno("Confirm", "Clear all activity logs?"):
            if os.path.exists(self.logger.log_file):
                os.remove(self.logger.log_file)
            self.refresh_logs()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    GestureControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
