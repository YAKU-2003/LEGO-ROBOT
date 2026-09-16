"""Camera demo: drive wheel speeds from arm height (PoseLandmarker) while
also watching for a discrete hand gesture (GestureRecognizer) as an
emergency-stop override -- both running together off the same PoseCamera
feed, one frame at a time.

No LEGO motor calls yet -- this is just to see the combined pose + gesture
pipeline work live before wiring it to actual motors.
"""

import cv2

from pose_camera import PoseCamera
from pose_gesture_controller import GestureController, WheelSpeeds

MIRRORED = True

# Any of these recognized gestures immediately zeroes both wheels,
# overriding whatever the arms are doing -- a quick way to kill motion
# without having to lower both arms back to neutral first.
STOP_GESTURES = {"Closed_Fist", "Open_Palm"}

controller = GestureController()

print("Press 'q' to quit.")

with PoseCamera(mirrored=MIRRORED, enable_gestures=True) as cam:
    while True:
        frame, landmarks, gesture = cam.read()
        if frame is None:
            print("Failed to read frame from webcam.")
            break

        if landmarks is not None:
            speeds = controller.update(landmarks, mirrored=MIRRORED)
        else:
            # No one detected in frame -- don't let a stale smoothed speed
            # keep the bot moving after the person steps away.
            controller.reset()
            speeds = WheelSpeeds(0.0, 0.0)

        if gesture in STOP_GESTURES:
            speeds = WheelSpeeds(0.0, 0.0)
            controller.reset()

        if landmarks is not None:
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"L: {speeds.left:6.1f}%  R: {speeds.right:6.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        if gesture is not None:
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

        cv2.imshow("Pose + Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
