import json
import os
from datetime import datetime


class ConfigManager:
    """Manages user profiles, gesture mappings, and system settings."""

    def __init__(self, config_file="system_config.json"):
        self.config_file = config_file
        self.config = self._load()

        # Cache hot-path settings to avoid dict lookups every frame
        s = self.config["settings"]
        self._cooldown   = s["cooldown"]
        self._stability  = s["gesture_stability"]
        self._tolerance  = s["face_tolerance"]

    # ── Load / Save ────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Config load error: {e}")
        return self._default()

    def _default(self):
        return {
            "users": {},
            "current_user": None,
            "gesture_mappings": {
                "count_1":        "volumeup",
                "count_2":        "volumedown",
                "count_3":        "mute",
                "count_4":        "playpause",
                "fist":           "playpause",
                "count_5_center": "screenshot",
                "count_5_left":   "swipe_left",
                "count_5_right":  "swipe_right",
                "count_5_up":     "scroll_up",
                "count_5_down":   "scroll_down",
                "thumbs_up":      "brightness_up",
                "thumbs_down":    "brightness_down",
                "ok_sign":        "screen_record",
                "peace_sign":     "screenshot",
                "count_6":        "escape",
                "count_7":        "task_view",
                "count_8":        "lock_screen",
                "count_9":        "desktop",
                "count_10":       "shutdown",
            },
            "action_commands": {
                "playpause":      {"type": "key",    "value": "playpause"},
                "volumeup":       {"type": "key",    "value": "volumeup"},
                "volumedown":     {"type": "key",    "value": "volumedown"},
                "prev_track":     {"type": "key",    "value": "prevtrack"},
                "next_track":     {"type": "key",    "value": "nexttrack"},
                "screenshot":     {"type": "hotkey", "value": ["win", "prtsc"]},
                "brightness_up":  {"type": "custom", "value": "brightness_up"},
                "brightness_down":{"type": "custom", "value": "brightness_down"},
                "mute":           {"type": "key",    "value": "volumemute"},
                "task_view":      {"type": "hotkey", "value": ["win", "tab"]},
                "close_window":   {"type": "hotkey", "value": ["alt", "f4"]},
                "minimize":       {"type": "hotkey", "value": ["win", "down"]},
                "maximize":       {"type": "hotkey", "value": ["win", "up"]},
                "desktop":        {"type": "hotkey", "value": ["win", "d"]},
                "lock_screen":    {"type": "hotkey", "value": ["win", "l"]},
                "swipe_left":     {"type": "custom", "value": "swipe_left"},
                "swipe_right":    {"type": "custom", "value": "swipe_right"},
                "scroll_up":      {"type": "custom", "value": "scroll_up"},
                "scroll_down":    {"type": "custom", "value": "scroll_down"},
                "screen_record":  {"type": "custom", "value": "screen_record"},
                "escape":         {"type": "key",    "value": "escape"},
                "shutdown":       {"type": "custom", "value": "shutdown"},
                "restart":        {"type": "custom", "value": "restart"},
            },
            "settings": {
                "cooldown":             1.0,
                "gesture_stability":    3,
                "face_tolerance":       0.5,
                "auth_check_frequency": 3,
                "max_auth_failures":    5,
                "enable_logging":       True,
                "detection_confidence": 0.7,
                "tracking_confidence":  0.7,
            },
        }

    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Config save error: {e}")
            return False

    # ── Users ──────────────────────────────────────────────────────────

    def add_user(self, username, face_encoding_path):
        if username in self.config["users"]:
            return False
        self.config["users"][username] = {
            "face_encoding":  face_encoding_path,
            "gesture_model":  None,   # set after training via gesture_collector
            "created_at":     datetime.now().isoformat(),
            "last_login":     None,
            "total_sessions": 0,
        }
        self.save_config()
        return True

    def remove_user(self, username):
        if username not in self.config["users"]:
            return False
        del self.config["users"][username]
        if self.config["current_user"] == username:
            self.config["current_user"] = None
        self.save_config()
        return True

    def set_current_user(self, username):
        if username not in self.config["users"]:
            return False
        # Only write to disk if something actually changed
        user = self.config["users"][username]
        now  = datetime.now().isoformat()
        changed = (self.config["current_user"] != username)
        self.config["current_user"] = username
        user["last_login"]     = now
        user["total_sessions"] += 1
        if changed:
            self.save_config()   # only write on user switch, not every re-auth
        return True

    def get_all_users(self):
        return list(self.config["users"].keys())

    def get_user_info(self, username):
        return self.config["users"].get(username)

    def get_gesture_model_path(self, username):
        """Return the trained gesture model path for a user, or None."""
        info = self.config["users"].get(username)
        if info:
            return info.get("gesture_model")
        return None

    # ── Gesture mappings ───────────────────────────────────────────────

    def get_gesture_mapping(self, gesture):
        return self.config["gesture_mappings"].get(gesture)

    def set_gesture_mapping(self, gesture, action):
        self.config["gesture_mappings"][gesture] = action
        self.save_config()

    def get_action_command(self, action):
        return self.config["action_commands"].get(action)

    # ── Settings ───────────────────────────────────────────────────────

    def get_setting(self, key):
        return self.config["settings"].get(key)

    def set_setting(self, key, value):
        self.config["settings"][key] = value
        # Keep cache in sync for hot-path keys
        if key == "cooldown":          self._cooldown  = value
        if key == "gesture_stability": self._stability = value
        if key == "face_tolerance":    self._tolerance = value
        self.save_config()
