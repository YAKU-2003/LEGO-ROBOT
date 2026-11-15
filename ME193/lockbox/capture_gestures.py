"""
Gesture capture script for the UNO Q finger-counting lock.

Saves images to ~/Downloads/gestures as <class>_<n>.jpg  (e.g. 0_1.jpg ... 5_50.jpg)

Choosing the camera:
  python capture_gestures.py              -> lists cameras found and asks which one
  python capture_gestures.py --camera 1   -> use camera 1 directly
  python capture_gestures.py --list       -> just list cameras and exit

Controls (click the preview window first so it has keyboard focus):
  0-5   start a burst of 50 photos for that finger count (3 s countdown first)
  c     switch to the next camera (live), to check which one is the webcam
  r     re-draw the capture box with the mouse (drag, then press ENTER/SPACE)
  q     quit

Tips:
  - Keep your hand moving a little during the burst (tilt, rotate, switch fingers).
  - Re-running a class continues the numbering (0_51, 0_52 ...) so nothing is overwritten.
  - The capture box is saved to gestures/roi.json - the training and lock code will
    reuse it so every stage crops exactly the same region.

Install:  pip install opencv-python
"""

import argparse
import cv2
import json
import time
import platform
from pathlib import Path

# ---------------- settings ----------------
CAMERA_INDEX = 0        # default suggestion in the camera menu (external webcam is usually 1)
SHOTS_PER_BURST = 300
SHOT_INTERVAL = 0.2     # seconds between shots -> a 50-shot burst takes ~10 s
COUNTDOWN = 3           # seconds before the burst starts
CLASSES = "012345"      # 0 = no hand / fist, 1-5 = finger counts
SAVE_FULL_FRAME = False # True also saves the uncropped frame into gestures/full/
# ------------------------------------------

OUT_DIR = Path.home() / "Downloads" / "gestures"
ROI_FILE = OUT_DIR / "roi.json"


def _raw_open(index):
    if platform.system() == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def camera_name(index):
    """Best-effort device name (Linux only; other OSes just show the index)."""
    p = Path(f"/sys/class/video4linux/video{index}/name")
    return p.read_text().strip() if p.exists() else f"Camera {index}"


def find_cameras(max_index=6):
    """Probe indices 0..max_index-1 and return the ones that deliver frames."""
    found = []
    for i in range(max_index):
        cap = _raw_open(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, f"{camera_name(i)} ({w}x{h})"))
        cap.release()
    return found


def choose_camera(cameras):
    print("\nCameras found:")
    for i, desc in cameras:
        print(f"  [{i}] {desc}")
    default = CAMERA_INDEX if any(i == CAMERA_INDEX for i, _ in cameras) else cameras[0][0]
    choice = input(f"Pick a camera number (Enter = {default}): ").strip()
    if choice.isdigit() and any(i == int(choice) for i, _ in cameras):
        return int(choice)
    return default


def open_camera(index):
    cap = _raw_open(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def load_roi(frame):
    if ROI_FILE.exists():
        return tuple(json.loads(ROI_FILE.read_text()))
    h, w = frame.shape[:2]
    s = min(h, w) // 2                      # default: centered square, half the frame height
    return ((w - s) // 2, (h - s) // 2, s, s)


def save_roi(roi):
    ROI_FILE.write_text(json.dumps([int(v) for v in roi]))


def next_index(cls):
    """Next free number for this class, so re-runs don't overwrite."""
    nums = [int(p.stem.split("_")[1]) for p in OUT_DIR.glob(f"{cls}_*.jpg")
            if p.stem.split("_")[1].isdigit()]
    return max(nums, default=0) + 1


def counts():
    return {c: len(list(OUT_DIR.glob(f"{c}_*.jpg"))) for c in CLASSES}


def draw_overlay(frame, roi, text, color=(0, 0, 255)):
    x, y, w, h = roi
    view = frame.copy()
    cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
    cv2.putText(view, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    c = counts()
    summary = "  ".join(f"{k}:{v}" for k, v in c.items())
    cv2.putText(view, summary, (10, view.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return view


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_FULL_FRAME:
        (OUT_DIR / "full").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Capture finger-gesture images.")
    parser.add_argument("--camera", type=int, help="camera index to use (skips the menu)")
    parser.add_argument("--list", action="store_true", help="list cameras and exit")
    args = parser.parse_args()

    print("Looking for cameras...")
    cameras = find_cameras()
    if not cameras:
        raise SystemExit("No cameras found. Is the webcam plugged in?")
    if args.list:
        for i, desc in cameras:
            print(f"  [{i}] {desc}")
        return

    cam_ids = [i for i, _ in cameras]
    cam_index = args.camera if args.camera is not None else choose_camera(cameras)
    cap = open_camera(cam_index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frame.")
    roi = load_roi(frame)
    print(f"Using camera {cam_index}. Saving to: {OUT_DIR}")
    print("Press 0-5 to capture, c to switch camera, r to redraw box, q to quit.")

    win = "Gesture capture"
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow(win, draw_overlay(
            frame, roi, f"Cam {cam_index} | 0-5: capture | c: camera | r: box | q: quit"))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and len(cam_ids) > 1:
            cap.release()
            pos = cam_ids.index(cam_index) if cam_index in cam_ids else -1
            cam_index = cam_ids[(pos + 1) % len(cam_ids)]
            cap = open_camera(cam_index)
            print(f"Switched to camera {cam_index}")
            continue

        if key == ord("r"):
            sel = cv2.selectROI(win, frame, showCrosshair=False)
            if sel[2] > 0 and sel[3] > 0:
                roi = sel
                save_roi(roi)
                print(f"New box saved: {roi}")

        if chr(key) in CLASSES:
            cls = chr(key)
            save_roi(roi)

            # countdown
            t_end = time.time() + COUNTDOWN
            while time.time() < t_end:
                ok, frame = cap.read()
                remaining = int(t_end - time.time()) + 1
                cv2.imshow(win, draw_overlay(frame, roi, f"Class {cls}: get ready... {remaining}",
                                             (0, 255, 255)))
                cv2.waitKey(1)

            # burst
            start = next_index(cls)
            x, y, w, h = roi
            for i in range(SHOTS_PER_BURST):
                ok, frame = cap.read()
                if not ok:
                    continue
                n = start + i
                crop = frame[y:y + h, x:x + w]
                cv2.imwrite(str(OUT_DIR / f"{cls}_{n}.jpg"), crop)
                if SAVE_FULL_FRAME:
                    cv2.imwrite(str(OUT_DIR / "full" / f"{cls}_{n}.jpg"), frame)
                cv2.imshow(win, draw_overlay(frame, roi,
                                             f"Class {cls}: {i + 1}/{SHOTS_PER_BURST}  keep moving!",
                                             (0, 255, 0)))
                cv2.waitKey(1)
                time.sleep(SHOT_INTERVAL)
            print(f"Class {cls}: saved {cls}_{start} to {cls}_{start + SHOTS_PER_BURST - 1}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
