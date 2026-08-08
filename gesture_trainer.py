"""
Gesture Model Trainer
=====================
Trains a per-user SVM classifier on landmark CSV data collected by
gesture_collector.py and saves the model to faces/<username>_model.pkl.

Also updates system_config.json with the model path so the gesture
controller loads it automatically on next session start.

Can be run standalone:
    python gesture_trainer.py --user vighnesh

Or called programmatically from gesture_collector.py.
"""

import os
import csv
import pickle
import argparse
import numpy as np

from config_manager import ConfigManager


class GestureTrainer:

    MODEL_DIR = "faces"

    def train(self, username: str, csv_path: str) -> tuple[bool, str]:
        """
        Train an SVM on the CSV data for `username`.
        Returns (success: bool, message: str).
        """
        # ── 1. Load data ───────────────────────────────────────────────
        X, y = [], []
        gesture_counts: dict[str, int] = {}

        try:
            with open(csv_path, 'r') as f:
                for row in csv.reader(f):
                    if not row or len(row) < 2:
                        continue
                    label = row[0].strip()
                    try:
                        features = [float(v) for v in row[1:]]
                    except ValueError:
                        continue
                    if len(features) != 126:
                        continue
                    X.append(features)
                    y.append(label)
                    gesture_counts[label] = gesture_counts.get(label, 0) + 1
        except Exception as e:
            return False, f"Failed to read CSV: {e}"

        if len(X) < 10:
            return False, f"Not enough data ({len(X)} samples). Collect more."

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        # ── 2. Check class balance ─────────────────────────────────────
        min_samples = min(gesture_counts.values())
        if min_samples < 10:
            low = [g for g, c in gesture_counts.items() if c < 10]
            return False, (f"Some gestures have too few samples: {low}. "
                           "Need at least 10 per gesture.")

        print(f"Training on {len(X)} samples across {len(gesture_counts)} gestures:")
        for g, c in sorted(gesture_counts.items()):
            print(f"  {g}: {c} samples")

        # ── 3. Train ───────────────────────────────────────────────────
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return False, ("scikit-learn not installed. "
                           "Run: pip install scikit-learn")

        # Pipeline: scale → SVM with RBF kernel
        # probability=True enables confidence scores for adaptive firing
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=42)),
        ])

        # Cross-validation to report accuracy before saving
        try:
            scores = cross_val_score(pipeline, X, y, cv=min(5, min_samples),
                                     scoring="accuracy")
            cv_acc = scores.mean() * 100
            print(f"Cross-validation accuracy: {cv_acc:.1f}% ± {scores.std()*100:.1f}%")
        except Exception:
            cv_acc = 0.0

        # Train on full dataset
        pipeline.fit(X, y)

        # ── 4. Save model ──────────────────────────────────────────────
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.MODEL_DIR, f"{username}_model.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    "model":    pipeline,
                    "classes":  list(pipeline.classes_),
                    "username": username,
                    "samples":  len(X),
                    "gestures": gesture_counts,
                    "cv_acc":   cv_acc,
                }, f)
        except Exception as e:
            return False, f"Failed to save model: {e}"

        # ── 5. Register model path in config ──────────────────────────
        try:
            config = ConfigManager()
            info = config.get_user_info(username)
            if info is not None:
                info["gesture_model"] = model_path
                config.save_config()
        except Exception as e:
            print(f"Warning: could not update config: {e}")

        msg = (f"Trained on {len(X)} samples, "
               f"{len(gesture_counts)} gestures, "
               f"CV accuracy: {cv_acc:.1f}%")
        print(f"✅ Model saved to {model_path}")
        return True, msg

    @staticmethod
    def load_model(username: str, config=None) -> dict | None:
        """
        Load a trained model for `username`. Returns None if not found.
        Pass an existing ConfigManager instance to avoid re-reading JSON.
        """
        # Try config first (avoids disk read if config already loaded)
        try:
            if config is None:
                config = ConfigManager()
            info = config.get_user_info(username)
            if info:
                path = info.get("gesture_model")
                if path and os.path.exists(path):
                    with open(path, 'rb') as f:
                        return pickle.load(f)
        except Exception:
            pass

        # Fallback: conventional path
        path = os.path.join("faces", f"{username}_model.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train gesture model for a user")
    parser.add_argument("--user", required=True, help="Username to train for")
    args = parser.parse_args()

    csv_path = os.path.join("faces", f"{args.user}_gestures.csv")
    if not os.path.exists(csv_path):
        print(f"❌ No data found at {csv_path}")
        print(f"   Run gesture_collector.py first to collect samples.")
        return

    trainer = GestureTrainer()
    success, msg = trainer.train(args.user, csv_path)
    if success:
        print(f"✅ {msg}")
    else:
        print(f"❌ {msg}")


if __name__ == "__main__":
    main()
