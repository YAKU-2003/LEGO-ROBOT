"""Webcam AprilTag detection using OpenCV's ArUco module.

Tracks one specific tag ID and prints which side of the frame it's on,
so a robot could use this to steer toward it:

    no tag seen at all      -> "Searching..."
    tags seen, none match   -> ignored
    TARGET_TAG_ID seen      -> "Turn Left" / "Turn Right" based on which
                                half of the frame its center falls in
"""

import cv2
from cv2 import aruco

TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = 0  # the tag ID to track; change to match your physical tag

dictionary = aruco.getPredefinedDictionary(TAG_FAMILY)
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detector_params)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

print("Press 'q' to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is None:
            print("Searching...")
        else:
            aruco.drawDetectedMarkers(frame, corners, ids)

            target_index = None
            for i, tag_id in enumerate(ids.flatten()):
                if tag_id == TARGET_TAG_ID:
                    target_index = i
                    break

            if target_index is not None:
                tag_center_x = corners[target_index][0].mean(axis=0)[0]
                frame_center_x = frame.shape[1] / 2
                if tag_center_x < frame_center_x:
                    print("Turn Left")
                else:
                    print("Turn Right")

        cv2.imshow("AprilTag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
