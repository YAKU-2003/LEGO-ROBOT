"""Step counter dashboard -- record counted walks, train the net, count steps live.

    python app.py            then open http://localhost:8001

  Record  pick what you're doing, Start, walk while counting your steps out loud,
          Stop, type how many steps you took
  Count   the net counts your steps live (+/- its measured uncertainty); the
          count also shows on the board's LED matrix
  Model   train (PyTorch), see how good it is: held-out accuracy, error spread,
          false steps when not walking, plots

Uses the same board app and MQTT link as the gait identifier.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import steplib as sl

PORT = 8001
WEB_DIR = os.path.join(sl.HERE, "web")

lock = threading.Lock()
stream = sl.gaitlib.IMUStream(keep_s=30.0)
def load_best():
    """The net trained by train_steps.py (models/stepnet.pt), or (None, None) before the first training."""
    return sl.load_model()


net, metrics = load_best()
counter = sl.StreamCounter(net)

rec = {"state": "idle", "kind": None, "session": "strap1", "samples": [], "t_start": 0.0, "t_go": 0.0,
       "held": None, "message": ""}
live = {"running": False, "rate": 0.0, "sent": None, "started_at": None}
training = {"running": False, "log": [], "finished": None}


def on_sample(s):
    if rec["state"] == "recording":
        rec["samples"].append(s)


stream.on_sample = on_sample


def reload_model():
    global net, metrics
    net, metrics = load_best()
    counter.net = net


def send_count(n):
    if stream.client and n != live["sent"]:
        stream.client.publish(sl.STEPS_TOPIC, str(n))
        live["sent"] = n


# ── background loop: recording timer + live counting ─────────────────────────

def background():
    last_rate_t = last_hop = time.monotonic()
    while True:
        time.sleep(0.05)
        now = time.monotonic()
        if now - last_rate_t >= 1.0:
            live["rate"] = stream.take_rate(now - last_rate_t)
            last_rate_t = now
        if rec["state"] == "countdown" and now >= rec["t_go"]:
            with lock:
                rec.update(samples=[], state="recording", t_start=now)
        if now - last_hop >= sl.HOP_S:
            last_hop = now
            if live["running"] and stream.connected:
                counter.update(stream.snapshot())
                send_count(int(round(counter.count)))


# ── data for the page ────────────────────────────────────────────────────────

rec_cache = {}  # path -> (mtime, info), so each file is analysed once


def recordings_info():
    out = []
    for r in sl.recordings():
        m = os.path.getmtime(r["path"])
        hit = rec_cache.get(r["path"])
        if not hit or hit[0] != m:
            t, X, ok = sl.to_grid(sl.gaitlib.load_csv(r["path"]))
            _, info = sl.label_events(X, r["steps"])
            hit = rec_cache[r["path"]] = (m, {"seconds": round(len(X) / sl.FS), "gyro_steps": info.get("auto", 0),
                                              "matched": info["matched"], "accel_peaks": sl.baseline_accel_peaks(X)})
        out.append({k: r[k] for k in ("name", "kind", "steps", "session", "source")} | hit[1])
    return out


def uncertainty(n):
    """+/- for a count of n steps (95%), from the leave-one-recording-out test."""
    m = (metrics or {}).get("net")
    if not m:
        return None
    return round(1.96 * m["sd"] / 100 * n + 1, 1)


def status():
    now = time.monotonic()
    s = rec["state"]
    steps_t = [t for t in counter.steps]
    last = stream.snapshot()
    t_now = last[-1, 0] if len(last) else 0
    cadence = 6 * sum(1 for t in steps_t if t > t_now - sl.CONTEXT_S - 10)  # steps in the last 10 s x 6
    n = counter.count
    return {
        "board": {"connected": stream.connected, "rate": round(live["rate"])},
        "rec": {"state": s, "kind": rec["kind"], "session": rec["session"], "message": rec["message"],
                "countdown": max(0.0, rec["t_go"] - now) if s == "countdown" else 0,
                "elapsed": now - rec["t_start"] if s == "recording" else 0,
                "samples": len(rec["samples"]) if s == "recording" else len(rec["held"] or []),
                "gyro_guess": rec.get("gyro_guess")},
        "count": {"running": live["running"], "steps": n, "shown": int(round(n)), "pm": uncertainty(n),
                  "cadence": cadence, "elapsed": time.time() - live["started_at"] if live["started_at"] else 0},
        "model": {"ready": net is not None, "metrics": metrics,
                  "trained_at": os.path.getmtime(sl.MODEL_PATH) if os.path.exists(sl.MODEL_PATH) else None,
                  "plots": [p for p in ("uncertainty.png", "false_steps.png", "example_density.png",
                                        "training_curve.png") if os.path.exists(os.path.join(sl.MODEL_DIR, p))]},
        "training": {"running": training["running"], "finished": training["finished"]},
        "kinds": sl.KINDS, "zero_kinds": sorted(sl.ZERO_KINDS),
    }


def signal(seconds):
    data = stream.snapshot()
    if len(data) == 0:
        return {"t": [], "amag": [], "gyro": [], "dens_t": [], "dens": [], "steps": []}
    data = data[data[:, 0] >= data[-1, 0] - seconds]
    t_end = data[-1, 0]
    a = np.linalg.norm(data[:, 1:4], axis=1) / sl.G
    w = np.linalg.norm(data[:, 4:7], axis=1)
    r3 = lambda v: [round(float(x), 3) for x in v]
    tr = [(t, d) for t, d in counter.trace if t >= t_end - seconds]
    return {"t": r3(data[::2, 0] - t_end), "amag": r3(a[::2]), "gyro": r3(w[::2]),
            "dens_t": r3([t - t_end for t, _ in tr]), "dens": r3([d for _, d in tr]),
            "steps": r3([t - t_end for t in counter.steps if t >= t_end - seconds])}


def run_training():
    training.update(running=True, log=[], finished=None)
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(sl.HERE, "train_steps.py")], cwd=sl.HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in proc.stdout:
        training["log"].append(line.rstrip())
    ok = proc.wait() == 0
    if ok:
        reload_model()
    training.update(running=False, finished={"ok": ok, "at": time.time()})


# ── HTTP ─────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:
            pass  # the page was closed mid-request

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
            self.send(signal(min(30.0, float(q.get("s", ["10"])[0]))))
        elif url.path == "/api/recordings":
            self.send(recordings_info())
        elif url.path == "/api/cv":
            self.send_file(os.path.join(sl.MODEL_DIR, "cv_results.json"), "application/json")
        elif url.path == "/api/train_log":
            self.send({"running": training["running"], "log": training["log"]})
        elif url.path.startswith("/plots/") and url.path.endswith(".png"):
            self.send_file(os.path.join(sl.MODEL_DIR, os.path.basename(url.path)), "image/png")
        else:
            self.send({"error": "not found"}, code=404)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            body = {}
        p = url.path

        if p == "/api/start":
            kind = body.get("kind")
            session = re.sub(r"[^A-Za-z0-9-]", "", str(body.get("session") or "strap1")) or "strap1"
            if kind not in sl.KINDS:
                return self.send({"error": "pick what you're doing"}, code=400)
            if rec["state"] != "idle":
                return self.send({"error": "already recording"}, code=409)
            with lock:
                rec.update(kind=kind, session=session, samples=[], held=None, message="", gyro_guess=None,
                           t_go=time.monotonic() + float(body.get("countdown", 3)), state="countdown")
            self.send({"ok": True})
        elif p == "/api/stop":
            with lock:
                if rec["state"] == "recording":
                    rec.update(held=rec["samples"], samples=[], state="confirm")
                else:
                    rec.update(state="idle", samples=[], held=None, message="Cancelled.")
            if rec["state"] == "confirm" and rec["held"] and len(rec["held"]) > sl.FS:
                X = sl.to_grid(np.array(rec["held"]))[1]
                rec["gyro_guess"] = sl.baseline_gyro_strides(X)
            self.send({"ok": True})
        elif p == "/api/save":
            if rec["state"] != "confirm":
                return self.send({"error": "nothing to save"}, code=409)
            steps = 0 if rec["kind"] in sl.ZERO_KINDS else int(body.get("steps", -1))
            if steps < 0:
                return self.send({"error": "type how many steps you took"}, code=400)
            try:
                path = sl.save_recording(rec["kind"], steps, rec["session"], rec["held"])
                rec["message"] = f"Saved {os.path.basename(path)} ({steps} steps)."
            except ValueError as e:
                rec["message"] = f"Nothing saved: {e}."
            rec.update(state="idle", held=None)
            self.send({"ok": True, "message": rec["message"]})
        elif p == "/api/discard":
            rec.update(state="idle", samples=[], held=None, message="Recording discarded.")
            self.send({"ok": True})
        elif p == "/api/delete":
            name = os.path.basename(str(body.get("name", "")))
            path = os.path.join(sl.DATA_DIR, name)
            if not sl.NAME_RE.match(name) or not os.path.exists(path):
                return self.send({"error": "no such recording (gait recordings are managed in the gait app)"},
                                 code=404)
            os.remove(path)
            self.send({"ok": True})
        elif p == "/api/counter":
            action = body.get("action")
            if action == "start":
                if not live["running"]:
                    counter.done_until = None  # don't count the time it was paused
                live.update(running=True, started_at=live["started_at"] or time.time())
            elif action == "pause":
                live["running"] = False
            elif action == "reset":
                counter.reset()
                live.update(started_at=time.time() if live["running"] else None)
                send_count(0)
            self.send({"ok": True})
        elif p == "/api/train":
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
    print(f"Step counter: http://localhost:{PORT}   (Ctrl+C to stop)")
    print("Net loaded." if net is not None else "No trained net yet -- record counted walks, then Train.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()


if __name__ == "__main__":
    main()
