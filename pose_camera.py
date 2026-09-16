"""Webcam capture + MediaPipe Pose landmark detection + hand gesture
recognition."""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
GESTURE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_recognizer.task")

# Minimum confidence to trust the top predicted gesture category.
GESTURE_CONFIDENCE_THRESHOLD = 0.6


def _ensure_downloaded(path: str, url: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading model to {path} ...")
    urllib.request.urlretrieve(url, path)


def create_landmarker() -> vision.PoseLandmarker:
    _ensure_downloaded(MODEL_PATH, MODEL_URL)
    # Force the CPU delegate: some platforms/builds crash trying to open a
    # GPU service for the pose detector's calculators, and the extra
    # latency of CPU inference doesn't matter for a single-person, single
    # camera feed like this one.
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_gesture_recognizer() -> vision.GestureRecognizer:
    _ensure_downloaded(GESTURE_MODEL_PATH, GESTURE_MODEL_URL)
    options = vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=GESTURE_MODEL_PATH,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.GestureRecognizer.create_from_options(options)


class PoseCamera:
    """Opens a webcam and yields (frame, pose_landmarks, gesture_name)
    triples.

    Frames are mirrored (flipped horizontally) before detection so the
    on-screen view feels natural, like looking in a mirror.
    """

    def __init__(self, camera_index: int = 0, mirrored: bool = True, enable_gestures: bool = True):
        self.camera_index = camera_index
        self.mirrored = mirrored
        self.enable_gestures = enable_gestures
        self._cap = None
        self._landmarker = None
        self._gesture_recognizer = None
        self._start_time = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")
        self._landmarker = create_landmarker()
        if self.enable_gestures:
            self._gesture_recognizer = create_gesture_recognizer()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
        if self._landmarker is not None:
            self._landmarker.close()
        if self._gesture_recognizer is not None:
            self._gesture_recognizer.close()

    def read(self):
        """Read one frame. Returns (frame_bgr, pose_landmarks_or_None,
        gesture_name_or_None)."""
        ok, frame = self._cap.read()
        if not ok:
            return None, None, None

        if self.mirrored:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        pose_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

        gesture_name = None
        if self._gesture_recognizer is not None:
            gesture_result = self._gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
            if gesture_result.gestures:
                top = gesture_result.gestures[0][0]
                if top.score >= GESTURE_CONFIDENCE_THRESHOLD:
                    gesture_name = top.category_name

        return frame, landmarks, gesture_name
