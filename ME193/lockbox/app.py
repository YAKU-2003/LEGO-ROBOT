"""Finger-Code Safe dashboard -- the lock, data capture and training in one local web page.

    python app.py            then open http://localhost:8001
    python app.py --camera 1 (if the USB webcam is not camera 0)

Tabs
  Safe     live camera with the capture box, the 3 code LEDs, the button (or SPACE),
           the CNN's 6 probabilities, an attempt log, change-code
  Capture  burst-capture more training photos (0-5) from the SAME camera + box
  Model    retrain the CNN, watch the log, see accuracy / confusion matrix / lighting test

The camera, the CNN and the lock all run here in Python - the page only shows them - so this
is exactly the code path the UNO Q will use.
"""

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

import finger_model as fm
from lock_logic import (CONF_MIN, ENTRY_TIMEOUT_S, HOLD_S, LOCKOUT_S, MAX_TRIES, OPEN_S,
                        LockEngine, save_code)

HERE = Path(__file__).parent
WEB = HERE / "web"
PORT = 8001                     # 8000 is the gait dashboard
CLASSES = range(6)

lock = threading.Lock()
cam = {"index": 0, "fps": 0.0, "ok": False, "jpeg": None, "frame": None}
pred = {"k": 0, "conf": 0.0, "probs": [0.0] * 6}
events = deque(maxlen=40)
capture = {"running": False, "cls": None, "done": 0, "total": 0, "countdown": 0.0, "message": ""}
training = {"running": False, "log": [], "finished": None}
model_holder = {"model": fm.load_model() if fm.MODEL_FILE.exists() else None}


def log_event(kind, text):
    events.appendleft({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})


latch = {"link": None}           # LatchLink when started with --latch (real box over MQTT)


def latch_cmd(command):
    # in --board mode without --latch the link is only used for the camera: the board runs
    # its own lock, so the dashboard must not open the lid
    if latch["link"] is not None and latch.get("link_cmds", True):
        latch["link"].send(command)


engine = LockEngine(on_event=log_event,
                    on_leds=lambda on: None,              # UNO Q: LED strip on/off
                    on_unlock=lambda: latch_cmd("open"),  # real latch (only with --latch)
                    on_lock=lambda: latch_cmd("close"))


# ── camera + CNN + lock loop ─────────────────────────────────────────────────

class BoardCamera:
    """
    --board mode: the webcam is on the UNO Q. The board streams its raw hand-box crop over MQTT
    (lunchlock/crops); this pastes it into an empty 1280x720 frame at the same box position, so the
    rest of the dashboard (model, grid view, Capture tab) works exactly as with a local camera.
    """

    def __init__(self, link):
        self.link, self.last, self.t_keep, self.source = link, None, 0.0, ""
        x, y, w, h = fm.load_roi()
        self.box = (x, y, w, h)
        self.canvas = np.full((720, 1280, 3), 45, np.uint8)
        cv2.putText(self.canvas, "UNO Q camera (only the box is sent)", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2)

    def _crop_from_preview(self, jpg):
        """Older board app (no raw crops): cut the box out of the 640-wide preview, inside its border."""
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        s = img.shape[1] / 1280.0
        x, y, w, h = (int(v * s) for v in self.box)
        m = 5                                                   # skip the drawn box outline
        return img[y + m:y + h - m, x + m:x + w - m]

    def read(self):
        t0 = time.time()
        while time.time() - t0 < 1.5:
            if time.time() - self.t_keep > 2:
                self.link.client.publish("lunchlock/crops_on", "1")
                self.link.keep_preview_on()
                self.t_keep = time.time()
            crop = None
            jpg = self.link.last_board_crop
            if jpg is not None and jpg is not self.last:
                self.last, self.source = jpg, "raw crop"
                crop = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            elif self.link.last_board_crop is None:
                pj = self.link.last_preview
                if pj is not None and pj is not self.last and time.time() - self.link.t_preview < 3:
                    self.last, self.source = pj, "preview (update the board app for full quality)"
                    crop = self._crop_from_preview(pj)
            if crop is not None and crop.size:
                x, y, w, h = self.box
                frame = self.canvas.copy()
                frame[y:y + h, x:x + w] = cv2.resize(crop, (w, h)) if crop.shape[:2] != (h, w) else crop
                cv2.putText(frame, f"source: {self.source}", (40, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (150, 150, 150), 2)
                return True, frame
            time.sleep(0.01)
        return False, None

    def release(self):
        pass


def open_camera(index):
    if cam.get("board"):
        return BoardCamera(latch["link"])
    c = (cv2.VideoCapture(index, cv2.CAP_DSHOW) if platform.system() == "Windows"
         else cv2.VideoCapture(index))
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # USB webcams: 30 fps at 720p (raw YUY2 ~10)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return c


def grid_view(crop, size=288):
    """Picture of what the grid branch sees: skin mask, 12x12 density cells, finger-height profile."""
    mask, density, profile = fm.explain(crop)
    m = cv2.resize((mask * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) // 3                     # dim mask in the background
    cell = size / fm.GRID
    for r in range(fm.GRID):
        for c in range(fm.GRID):
            d = float(density[r, c])
            if d > 0.05:                                                 # cell shaded by hand coverage
                x0, y0 = int(c * cell), int(r * cell)
                overlay = view[y0:int(y0 + cell), x0:int(x0 + cell)]
                overlay[:] = (overlay * (1 - d) + np.array([255, 156, 124]) * d).astype(np.uint8)
    for i in range(1, fm.GRID):                                          # grid lines
        p = int(i * cell)
        cv2.line(view, (p, 0), (p, size), (60, 60, 70), 1)
        cv2.line(view, (0, p), (size, p), (60, 60, 70), 1)
    colw = size / fm.PROFILE_COLS                                        # finger profile (yellow line)
    pts = [(int((i + 0.5) * colw), int(size * (1 - v))) for i, v in enumerate(profile)]
    cv2.polylines(view, [np.array(pts, np.int32)], False, (71, 181, 255), 2)
    return view


def camera_loop():
    roi = fm.load_roi()
    x, y, w, h = roi
    current, cap = None, None
    n, t_fps = 0, time.monotonic()
    bad = 0                                   # consecutive failed / black frames
    while True:
        if current != cam["index"] or cam.pop("reopen", False) or bad >= 15:
            if cap is not None:
                cap.release()
            if bad >= 15:
                log_event("camera", f"Camera {cam['index']} stopped sending images - reopening")
                time.sleep(1.0)
            current, bad = cam["index"], 0
            cap = open_camera(current)
        ok, frame = cap.read()
        # A webcam that was unplugged / grabbed by another program often keeps "working" but
        # returns black frames - treat those as failures so the camera gets reopened.
        if ok and frame is not None and frame.mean() < 3:
            ok = False
        cam["ok"] = bool(ok)
        if not ok or frame is None:
            bad += 1
            time.sleep(0.2)
            continue
        bad = 0
        now = time.time()
        model = model_holder["model"]
        crop = frame[y:y + h, x:x + w]
        gv = grid_view(crop)
        ok_g, gjpg = cv2.imencode(".jpg", gv, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok_g:
            cam["grid_jpeg"] = gjpg.tobytes()
        if model is not None:
            k, conf, probs = fm.predict(model, crop)
            with lock:
                pred.update(k=k, conf=conf, probs=[float(p) for p in probs])
                engine.update(k, conf, now)

        color = {"IDLE": (160, 160, 160), "ENTER": (0, 190, 255), "OPEN": (110, 220, 60),
                 "WRONG": (90, 80, 255), "LOCKOUT": (90, 80, 255)}[engine.state]
        if capture["running"]:
            color = (255, 160, 120)
        view = frame.copy()
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 3)
        small = cv2.resize(view, (view.shape[1] * 540 // view.shape[0], 540))
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with lock:
            cam["frame"] = frame
            cam["jpeg"] = jpg.tobytes() if ok else cam["jpeg"]
        n += 1
        if time.monotonic() - t_fps >= 1.0:
            cam["fps"] = n / (time.monotonic() - t_fps)
            n, t_fps = 0, time.monotonic()


# ── capture + training jobs ─────────────────────────────────────────────────

def next_index(cls):
    nums = [int(p.stem.split("_")[1]) for p in fm.DATA_DIR.glob(f"{cls}_*.jpg")
            if p.stem.split("_")[1].isdigit()]
    return max(nums, default=0) + 1


def class_counts():
    return [len(list(fm.DATA_DIR.glob(f"{c}_*.jpg"))) for c in CLASSES]


def run_capture(cls, total, interval=0.2, countdown=3.0):
    x, y, w, h = fm.load_roi()
    capture.update(running=True, cls=cls, done=0, total=total, message="")
    t_end = time.time() + countdown
    while time.time() < t_end:
        capture["countdown"] = t_end - time.time()
        time.sleep(0.05)
    capture["countdown"] = 0.0
    start = next_index(cls)
    for i in range(total):
        with lock:
            frame = cam["frame"]
        if frame is not None:
            cv2.imwrite(str(fm.DATA_DIR / f"{cls}_{start + i}.jpg"), frame[y:y + h, x:x + w])
            capture["done"] = i + 1
        time.sleep(interval)
    capture.update(running=False, message=f"Saved {cls}_{start} ... {cls}_{start + total - 1}")
    log_event("capture", f"Captured {total} photos of {cls} fingers")


def run_training(epochs):
    training.update(running=True, log=[], finished=None)
    proc = subprocess.Popen([sys.executable, "-u", str(HERE / "train_cnn.py"), "--epochs", str(epochs)],
                            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        training["log"].append(line.rstrip())
        training["log"] = training["log"][-400:]
    proc.wait()
    if proc.returncode == 0:
        model_holder["model"] = fm.load_model()
        log_event("model", "New model trained and loaded")
    training.update(running=False, finished=proc.returncode)


def model_report():
    p = HERE / "model_report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    r.pop("history", None)
    r["mtime"] = p.stat().st_mtime
    return r


def status():
    now = time.time()
    with lock:
        return {
            "state": engine.state, "entered": len(engine.reader.digits),
            "digits": engine.reader.digits, "progress": engine.reader.progress,
            "candidate": engine.reader.cand, "time_left": engine.time_left(now),
            "wrong": engine.wrong, "max_tries": MAX_TRIES,
            "pred": dict(pred), "events": list(events),
            "camera": {"index": cam["index"], "ok": cam["ok"], "fps": round(cam["fps"], 1)},
            "latch": None if latch["link"] is None else {
                "online": latch["link"].board_online, "position_ms": latch["link"].position_ms},
            "model_loaded": model_holder["model"] is not None,
            "policy": {"conf_min": CONF_MIN, "hold_s": HOLD_S, "entry_s": ENTRY_TIMEOUT_S,
                       "open_s": OPEN_S, "lockout_s": LOCKOUT_S},
            "capture": dict(capture), "counts": class_counts(),
            "training": {"running": training["running"], "finished": training["finished"],
                         "log": training["log"][-60:]},
        }


# ── HTTP ────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self.send(status())
        if path == "/api/model":
            return self.send(model_report() or {})
        if path in ("/video.mjpg", "/grid.mjpg"):
            key = "jpeg" if path == "/video.mjpg" else "grid_jpeg"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                last = None
                while True:
                    with lock:
                        jpg = cam.get(key)
                    if jpg is not None and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        if path.startswith("/img/"):
            name = path[5:]
            if name in ("training_curves.png", "confusion_matrix.png") and (HERE / name).exists():
                return self.send((HERE / name).read_bytes(), "image/png")
        if path.startswith("/sample/"):
            # one example photo per class for the Capture tab
            try:
                cls = int(path.split("/")[2])
                files = sorted(fm.DATA_DIR.glob(f"{cls}_*.jpg"), key=lambda p: p.stat().st_mtime)
                if files:
                    return self.send(files[-1].read_bytes(), "image/jpeg")
            except (ValueError, IndexError):
                pass
        self.send({"error": "not found"}, code=404)

    def do_POST(self):
        path, b = urlparse(self.path).path, self.body()
        if path == "/api/button":
            with lock:
                ok = engine.press_button()
            return self.send({"ok": ok})
        if path == "/api/cancel":
            with lock:
                engine.cancel()
            return self.send({"ok": True})
        if path == "/api/code":
            code = str(b.get("code", ""))
            if len(code) != 3 or any(c not in "12345" for c in code):
                return self.send({"ok": False, "error": "Code must be 3 digits, each 1-5"}, code=400)
            with lock:
                engine.code = tuple(int(c) for c in code)
                save_code(engine.code)
            log_event("code", "Code changed")
            return self.send({"ok": True})
        if path == "/api/camera":
            cam["index"] = int(b.get("index", 0))
            cam["reopen"] = True                      # also reopens the same camera if it hung
            return self.send({"ok": True})
        if path == "/api/capture":
            if capture["running"] or training["running"]:
                return self.send({"ok": False, "error": "busy"}, code=409)
            cls, n = int(b.get("cls", -1)), int(b.get("n", 200))
            if cls not in CLASSES or not 1 <= n <= 500:
                return self.send({"ok": False, "error": "bad class"}, code=400)
            threading.Thread(target=run_capture, args=(cls, n), daemon=True).start()
            return self.send({"ok": True})
        if path == "/api/train":
            if training["running"] or capture["running"]:
                return self.send({"ok": False, "error": "busy"}, code=409)
            epochs = max(5, min(200, int(b.get("epochs", 80))))
            threading.Thread(target=run_training, args=(epochs,), daemon=True).start()
            return self.send({"ok": True})
        self.send({"error": "not found"}, code=404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--latch", action="store_true",
                    help="drive the real latch on the UNO Q over MQTT (unoq-lunchlock-app must be running)")
    ap.add_argument("--board", action="store_true",
                    help="use the UNO Q's camera (its hand-box crop over MQTT) instead of a laptop camera: "
                         "live diagnosis + Capture tab saves training photos through the board's camera")
    args = ap.parse_args()
    cam["index"] = args.camera
    if args.latch or args.board:
        from latch_link import LatchLink
        latch["link"] = LatchLink(print_events=False)
        if args.latch:
            log_event("start", "Latch link on - correct code opens the real box")
    if args.board:
        cam["board"] = True
        latch["link_cmds"] = args.latch       # board runs its own lock; only drive the lid with --latch
        log_event("start", "Board mode - showing the UNO Q camera's hand box")
    threading.Thread(target=camera_loop, daemon=True).start()
    log_event("start", f"Dashboard started - camera {args.camera}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Finger-Code Safe dashboard: http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
