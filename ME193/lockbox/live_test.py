"""
Live test of the finger-counting CNN with the external webcam.

  python live_test.py            (camera 0 = the USB webcam)

Shows the camera with the capture box (same roi.json used for training), the predicted
finger count, its confidence, and a bar for each of the 6 classes.  q quits.
"""

import argparse
import platform

import cv2
import numpy as np

from finger_model import load_model, load_roi, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)   # 0 = the USB webcam on this laptop
    args = ap.parse_args()

    model = load_model()
    x, y, w, h = load_roi()
    cap = (cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if platform.system() == "Windows"
           else cv2.VideoCapture(args.camera))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        k, conf, probs = predict(model, frame[y:y + h, x:x + w])

        color = (0, 200, 0) if conf > 0.8 else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{k} fingers  ({conf:.0%})", (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        for c, p in enumerate(probs):          # probability bars
            top = 60 + c * 32
            cv2.putText(frame, str(c), (10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (35, top), (35 + int(250 * p), top + 24),
                          (0, 200, 0) if c == k else (180, 180, 180), -1)
        cv2.imshow("Finger CNN live test (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
