"""
Finger-code safe - simple OpenCV-window version (the web dashboard is app.py).

  python lock_pc.py            (camera 0 = USB webcam)

  SPACE = the safe's button, then show the 3 digits (hold each until its dot fills;
  make a fist between two equal digits).   d = debug view   q = quit
The lock policy itself lives in lock_logic.py.
"""

import argparse
import platform
import time

import cv2

from finger_model import load_model, load_roi, predict
from lock_logic import MAX_TRIES, LockEngine

COLORS = {"IDLE": (90, 90, 90), "ENTER": (0, 200, 255), "OPEN": (0, 220, 0),
          "WRONG": (0, 0, 255), "LOCKOUT": (0, 0, 180)}


def draw(frame, roi, eng, k, conf, debug, now):
    x, y, w, h = roi
    H, W = frame.shape[:2]
    ring = COLORS[eng.state]
    cv2.rectangle(frame, (x, y), (x + w, y + h), ring, 3)
    n = len(eng.reader.digits)
    for i in range(3):
        cx, cy = W // 2 - 90 + i * 90, 60
        filled = i < n or eng.state in ("OPEN", "WRONG", "LOCKOUT")
        cv2.circle(frame, (cx, cy), 28, ring if filled else (60, 60, 60), -1 if filled else 3)
        if eng.state == "ENTER" and i == n and eng.reader.progress > 0:
            cv2.ellipse(frame, (cx, cy), (28, 28), -90, 0, int(360 * eng.reader.progress), (255, 255, 255), 5)
    msg = {"IDLE": "LOCKED - press SPACE", "ENTER": f"Show digit {n + 1} of 3", "OPEN": "UNLOCKED",
           "WRONG": f"WRONG CODE ({eng.wrong}/{MAX_TRIES})", "LOCKOUT": "TOO MANY TRIES - LOCKED OUT"}[eng.state]
    cv2.putText(frame, msg, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, ring, 3)
    t = eng.time_left(now)
    if t is not None:
        cv2.putText(frame, f"{t:4.1f}s", (W - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, ring, 3)
    if debug:
        cv2.putText(frame, f"CNN: {k} ({conf:.0%})", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "entered: " + " ".join(map(str, eng.reader.digits)), (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    model, roi = load_model(), load_roi()
    x, y, w, h = roi
    cap = (cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if platform.system() == "Windows"
           else cv2.VideoCapture(args.camera))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    eng = LockEngine(on_event=lambda kind, text: print(text))
    debug = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        k, conf, _ = predict(model, frame[y:y + h, x:x + w])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug
        if key == ord(" "):
            eng.press_button(now)
        eng.update(k, conf, now)
        cv2.imshow("Finger-code safe", draw(frame, roi, eng, k, conf, debug, now))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
