"""Camera demo: run the webcam through MediaPipe HandLandmarker and print the
wheel speeds hand_gesture_controller.GestureController computes from it.

No LEGO motor calls yet -- this is just to see the hand tracking + gesture
mapping work live before wiring it to actual motors.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from hand_gesture_controller import GestureController

MODEL_PATH = "hand_landmarker.task"
MIRRORED = True  # flip the frame for a natural "mirror" webcam view

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2,
)

landmarker = HandLandmarker.create_from_options(options)
controller = GestureController()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

start_time = time.time()
print("Press 'q' to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        if MIRRORED:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        speeds = controller.update(result, mirrored=MIRRORED)

        cv2.putText(
            frame,
            f"L: {speeds.left:6.1f}%  R: {speeds.right:6.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
