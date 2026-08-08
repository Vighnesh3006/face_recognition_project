import os
import threading
from datetime import datetime

class ActivityLogger:
    """Logs system activities and security events"""
    
    def __init__(self, log_file="activity_log.txt", enabled=True):
        self.log_file = log_file
        self.enabled = enabled
        self._lock = threading.Lock()
    
    def log(self, event_type, message, user=None):
        """Log an event"""
        if not self.enabled:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_str = f"[{user}]" if user else "[SYSTEM]"
        log_entry = f"{timestamp} {user_str} [{event_type}] {message}\n"
        
        try:
            with self._lock:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry)
        except Exception as e:
            print(f"Logging error: {e}")

    def log_auth_success(self, user):
        """Log successful authentication"""
        self.log("AUTH_SUCCESS", "Face authentication successful", user)

    def log_auth_failure(self, reason, user=None):
        """Log failed authentication"""
        self.log("AUTH_FAILURE", f"Authentication failed: {reason}", user)

    def log_gesture_action(self, gesture, action, user):
        """Log gesture action execution"""
        self.log("GESTURE", f"Gesture '{gesture}' executed action '{action}'", user)
    
    def log_blocked_action(self, gesture, reason, user=None):
        """Log blocked action attempt"""
        self.log("BLOCKED", f"Gesture '{gesture}' blocked: {reason}", user)
    
    def log_session_start(self, user):
        """Log session start"""
        self.log("SESSION_START", "Gesture control session started", user)
    
    def log_session_end(self, user, reason="Normal exit"):
        """Log session end"""
        self.log("SESSION_END", f"Session ended: {reason}", user)
    
    def log_config_change(self, setting, value, user):
        """Log configuration change"""
        self.log("CONFIG", f"Setting '{setting}' changed to '{value}'", user)
    
    def get_recent_logs(self, lines=50):
        """Get recent log entries"""
        if not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except Exception as e:
            print(f"Error reading logs: {e}")
            return []
