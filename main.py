"""
Gesture Control System — Entry Point
Run this file to launch the application:  python main.py
"""

import sys


def check_dependencies():
    missing = []
    for module in ['cv2', 'mediapipe', 'face_recognition', 'pyautogui', 'PIL',
                   'tkinter', 'screen_brightness_control', 'win32api',
                   'win32con', 'win32gui', 'sklearn']:
        try:
            __import__('tkinter' if module == 'tkinter' else module)
        except ImportError:
            missing.append(module)
    return missing


def main():
    missing = check_dependencies()
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)

    from gui_app import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
