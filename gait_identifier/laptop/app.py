"""Gait Identifier dashboard -- record, train and watch both models live in the browser.

    python app.py            then open http://localhost:8000

Does the jobs of record.py, train.py and live.py from one page:
  Record  pick an activity + session, Start/Stop, live graph while recording
  Live    both models' guesses, model 1's peak against its thresholds,
          model 2's confidence, live accel/gyro graphs, a 60 s timeline
  Models  train both models, see accuracy, confusion matrices and plots

Also reachable from a phone on the same Wi-Fi at http://<laptop IP>:8000
(Windows Firewall has to allow port 8000 for that).
"""

import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import gaitlib

PORT = 8000
WEB_DIR = os.path.join(gaitlib.HERE, "web")
TRIM_S = 2.0

lock = threading.Lock()
stream = gaitlib.IMUStream(keep_s=30.0)
clf = gaitlib.LiveClassifier()

rec = {"state": "idle", "activity": None, "session": "strap1", "samples": [], "t_start": 0.0,
       "t_go": 0.0, "seconds": 60.0, "message": ""}
live = {"result": None, "timeline": deque(maxlen=120), "rate": 0.0}
training = {"running": False, "log": [], "finished": None}


def on_sample(s):
    if rec["state"] == "recording":
        rec["samples"].append(s)


stream.on_sample = on_sample


# ── background loop: recording timer + live classification ──────────────────

def finish_recording(save=True):
    with lock:
        samples, rec["samples"] = rec["samples"], []
        activity, session = rec["activity"], rec["session"]
        rec["state"] = "idle"
    if not save:
        rec["message"] = "Recording discarded."
        return
    try:
        path, n, secs = gaitlib.save_recording(activity, session, samples, TRIM_S)
        rec["message"] = f"Saved {os.path.basename(path)} ({secs:.0f} s, {n} samples)."
    except ValueError as e:
        rec["message"] = f"Nothing saved: {e}."


def background():
    last_rate_t = time.monotonic()
    while True:
        time.sleep(clf.TICK_S)
        now = time.monotonic()
        if now - last_rate_t >= 1.0:
            live["rate"] = stream.take_rate(now - last_rate_t)
            last_rate_t = now

        if rec["state"] == "countdown" and now >= rec["t_go"]:
            with lock:
                rec["samples"] = []
                rec["state"] = "recording"
                rec["t_start"] = now
        elif rec["state"] == "recording" and rec["seconds"] > 0 and \
                now - rec["t_start"] >= rec["seconds"] + 2 * TRIM_S:
            finish_recording()

        r = clf.update(stream.snapshot())
        live["result"] = r
        if r and r["fresh"]:
            stream.send_prediction(r["model1"], r["model2"])
            live["timeline"].append({"t": time.time(), "m1": r["model1"], "m2": r["model2"]})
        elif r is None and clf._tick % 10 == 0:
            stream.send_prediction(None, None)


# ── data for the page ────────────────────────────────────────────────────────

gyro_offsets = {}  # path -> (mtime, offset), so files are only scanned once


def recordings_info():
    out = []
    for path, label in gaitlib.recordings():
        try:
            with open(path) as f:
                n = sum(1 for _ in f) - 1
        except OSError:
            continue
        offset = gyro_offsets.get(path)
        if offset is None or offset[0] != os.path.getmtime(path):
            offset = gyro_offsets[path] = (os.path.getmtime(path), gaitlib.gyro_offset(gaitlib.load_csv(path)))
        out.append({"name": os.path.basename(path), "activity": label,
                    "session": gaitlib.session_of(path), "seconds": round(n / gaitlib.FS),
                    "gyro_offset": round(offset[1]), "gyro_bad": offset[1] > gaitlib.GYRO_OFFSET_LIMIT})
    return out


def model_info():
    if not clf.ready:
        return {"ready": False}
    rules = [{"lo": None if math.isinf(lo) else lo, "hi": None if math.isinf(hi) else hi, "activity": c}
             for lo, hi, c in clf.peak.get("rules", [])]
    summary = []
    report = os.path.join(gaitlib.MODEL_DIR, "report.txt")
    if os.path.exists(report):
        with open(report, encoding="utf-8", errors="replace") as f:
            summary = [l.strip() for l in f if l.startswith("SUMMARY")]
    plots = [p for p in ("confusion_matrices.png", "model1_peak_histogram.png", "example_windows.png")
             if os.path.exists(os.path.join(gaitlib.MODEL_DIR, p))]
    imp = clf.full["model"].feature_importances_
    top = sorted(zip(clf.full.get("features", []), imp), key=lambda t: -t[1])[:10]
    return {"ready": True, "direction": clf.peak["direction"], "rules": rules,
            "feature_set": clf.full.get("feature_set"), "summary": summary, "plots": plots,
            "m1": clf.peak.get("metrics"), "m2": clf.full.get("metrics"),
            "importances": [[n, round(float(v), 4)] for n, v in top],
            "trained_at": os.path.getmtime(os.path.join(gaitlib.MODEL_DIR, "full_model.pkl"))}


def status():
    now = time.monotonic()
    r = rec["state"]
    return {
        "board": {"connected": stream.connected, "rate": round(live["rate"]),
                  "gyro_offset": round(gaitlib.gyro_offset(recent_data(3.0))),
                  "accel_g": round(gaitlib.mean_accel_g(recent_data(1.0)), 2)},
        "rec": {"state": r, "activity": rec["activity"], "session": rec["session"],
                "countdown": max(0.0, rec["t_go"] - now) if r == "countdown" else 0,
                "elapsed": now - rec["t_start"] if r == "recording" else 0,
                "seconds": rec["seconds"], "trim": TRIM_S, "samples": len(rec["samples"]),
                "message": rec["message"]},
        "live": live["result"],
        "timeline": list(live["timeline"]),
        "model": model_info(),
        "training": {"running": training["running"], "finished": training["finished"]},
        "activities": gaitlib.ACTIVITIES,
    }


def recent_data(seconds):
    data = stream.snapshot()
    return data[data[:, 0] >= data[-1, 0] - seconds] if len(data) else data


def signal(seconds):
    data = stream.snapshot()
    if len(data) == 0:
        return {"t": [], "a": [[], [], []], "g": [[], [], []], "amag": [], "gmag": []}
    data = data[data[:, 0] >= data[-1, 0] - seconds][::2]  # 25 points/s is plenty to draw
    a = data[:, 1:4] / gaitlib.G
    g = np.degrees(data[:, 4:7])
    r2 = lambda v: [round(float(x), 3) for x in v]
    return {"t": r2(data[:, 0] - data[-1, 0]), "a": [r2(a[:, i]) for i in range(3)],
            "g": [r2(g[:, i]) for i in range(3)],
            "amag": r2(np.linalg.norm(a, axis=1)), "gmag": r2(np.linalg.norm(g, axis=1))}


def run_training():
    training.update(running=True, log=[], finished=None)
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(gaitlib.HERE, "train.py")], cwd=gaitlib.HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in proc.stdout:
        training["log"].append(line.rstrip())
    ok = proc.wait() == 0
    if ok:
        clf.reload()
    training.update(running=False, finished={"ok": ok, "at": time.time()})


# ── HTTP ─────────────────────────────────────────────────────────────────────

SAFE_NAME = re.compile(r"^[a-z]+_[A-Za-z0-9-]+_[0-9-]+\.csv$|^[a-z]+_[0-9-]+\.csv$")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep the console quiet

    def send(self, body, ctype="application/json", code=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                self.send(f.read(), ctype)
        except OSError:
            self.send({"error": "not found"}, code=404)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self.send_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            self.send(status())
        elif url.path == "/api/signal":
            self.send(signal(float(q.get("s", ["10"])[0])))
        elif url.path == "/api/recordings":
            self.send(recordings_info())
        elif url.path == "/api/train_log":
            self.send({"running": training["running"], "log": training["log"]})
        elif url.path == "/api/report":
            self.send_file(os.path.join(gaitlib.MODEL_DIR, "report.txt"), "text/plain; charset=utf-8")
        elif url.path.startswith("/plots/") and url.path.endswith(".png"):
            self.send_file(os.path.join(gaitlib.MODEL_DIR, os.path.basename(url.path)), "image/png")
        elif url.path == "/api/download":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for path, _ in gaitlib.recordings():
                    z.write(path, os.path.basename(path))
            self.send(buf.getvalue(), "application/zip",
                      headers={"Content-Disposition": "attachment; filename=gait_recordings.zip"})
        else:
            self.send({"error": "not found"}, code=404)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            body = {}

        if url.path == "/api/start":
            activity = body.get("activity")
            session = re.sub(r"[^A-Za-z0-9-]", "", str(body.get("session") or "strap1")) or "strap1"
            if activity not in gaitlib.ACTIVITIES:
                return self.send({"error": "pick an activity"}, code=400)
            if rec["state"] != "idle":
                return self.send({"error": "already recording"}, code=409)
            with lock:
                rec.update(activity=activity, session=session, samples=[], message="",
                           seconds=float(body.get("seconds", 60)),
                           t_go=time.monotonic() + float(body.get("countdown", 5)), state="countdown")
            self.send({"ok": True})
        elif url.path == "/api/stop":
            if rec["state"] == "recording":
                finish_recording(save=True)
            elif rec["state"] == "countdown":
                finish_recording(save=False)
            self.send({"ok": True, "message": rec["message"]})
        elif url.path == "/api/discard":
            if rec["state"] != "idle":
                finish_recording(save=False)
            self.send({"ok": True})
        elif url.path == "/api/delete":
            name = os.path.basename(str(body.get("name", "")))
            path = os.path.join(gaitlib.DATA_DIR, name)
            if not SAFE_NAME.match(name) or not os.path.exists(path):
                return self.send({"error": "no such recording"}, code=404)
            os.remove(path)
            self.send({"ok": True})
        elif url.path == "/api/train":
            if training["running"]:
                return self.send({"error": "already training"}, code=409)
            threading.Thread(target=run_training, daemon=True).start()
            self.send({"ok": True})
        else:
            self.send({"error": "not found"}, code=404)


def main():
    stream.connect()
    threading.Thread(target=background, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Gait Identifier dashboard: http://localhost:{PORT}   (Ctrl+C to stop)")
    print("Models loaded." if clf.ready else "No trained models yet -- record, then press Train on the Models tab.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stream.send_prediction(None, None)
        time.sleep(0.2)
        stream.close()


if __name__ == "__main__":
    main()
