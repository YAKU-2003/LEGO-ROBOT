# LunchLock -- the whole lock running ON the UNO Q (Linux side).
#
#   USB webcam -> crop (roi.json, same box as training) -> FingerNet (NumPy, numpy_model.py)
#   -> lock_logic.LockEngine (same policy as the laptop) -> Bridge -> sketch.ino:
#        latch_open / latch_close (TT motor, never past closed)   show_lock (LED matrix)
#
# The model is NOT trained here: finger_weights.npz is exported from the laptop by
# lockbox/export_for_unoq.py (which also copies roi.json, lock_config.json and lock_logic.py).
#
# Button: the real button on pin 2 (sketch -> Bridge "button"), or from the laptop over MQTT
# (lockbox/latch_cli.py, type  b ). The laptop is optional: without it the box works alone.
#
# MQTT (only if the laptop broker is reachable):
#   in   lunchlock/button            any message = press the button
#   in   lunchlock/latch             open | close | stop | home | jog <ms> | openms <ms>
#   in   lunchlock/snap              send the current camera crop back as JPEG
#   in   lunchlock/preview_on        stream a small live view for 5 s (board_view.py re-sends it)
#   in   lunchlock/crops_on          stream the RAW hand-box crop for 5 s (laptop app.py --board:
#                                    live diagnosis + capturing training photos through THIS camera)
#   out  lunchlock/crops             JPEG of the raw crop, exactly what the network sees (~8 per s)
#   out  lunchlock/preview           JPEG: whole camera view + hand box + the network's guess
#   out  lunchlock/status            JSON: state, finger guess, confidence, fps
#   out  lunchlock/state             latch position "ms,moving"
#   out  lunchlock/crop              JPEG of the crop (answer to snap)
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import *

from lock_logic import LockEngine
from numpy_model import FingerNet

HERE = Path(__file__).parent
MQTT_HOST = "192.168.4.241"     # <-- the laptop's IP (ipconfig on the laptop); optional
# (this board is 192.168.4.43 -> phone page: http://192.168.4.43:7000)
MQTT_PORT = 1883
CAMERA_INDEX = 0                # /dev/video0
TRAIN_W, TRAIN_H = 1280, 720    # frame size the training photos were cropped from

STATE_ID = {"IDLE": 0, "ENTER": 1, "OPEN": 2, "WRONG": 3, "LOCKOUT": 4}
lock = threading.Lock()
net = FingerNet(HERE / "finger_weights.npz")
roi = json.loads((HERE / "roi.json").read_text())
client = None
mqtt_ok = False
status = {"state": "IDLE", "k": 0, "conf": 0.0, "fps": 0.0, "entered": 0, "camera": "starting"}
snap_requested = False
crops_until = 0.0        # laptop app.py --board keeps this in the future
CROPS_EVERY_S = 0.12
preview_until = 0.0      # laptop's board_view.py / the phone page keep this in the future while watching
latest_preview = None    # newest preview JPEG (for the phone page)
WEB_PORT = 7000          # phone: http://<board IP>:7000
PREVIEW_EVERY_S = 0.2    # 5 preview frames per second, small JPEGs -> light on the Wi-Fi
STATE_COLOR = {"IDLE": (160, 160, 160), "ENTER": (0, 190, 255), "OPEN": (60, 220, 60),
               "WRONG": (60, 60, 255), "LOCKOUT": (60, 60, 255)}


def preview_jpeg(frame, box, st, k, conf, entered, progress):
    """Small picture of the whole camera view with the hand box and the network's guess."""
    x, y, w, h = box
    view = frame.copy()
    col = STATE_COLOR[st]
    cv2.rectangle(view, (x, y), (x + w, y + h), col, 6)
    view = cv2.resize(view, (640, int(640 * view.shape[0] / view.shape[1])))
    cv2.rectangle(view, (0, 0), (640, 44), (20, 20, 20), -1)
    cv2.putText(view, f"{st}  sees {k} ({conf:.0%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    for i in range(3):                                         # the 3 code boxes, like the LED matrix
        cx = 480 + i * 50
        cv2.rectangle(view, (cx, 8), (cx + 30, 38), col, -1 if i < entered else 2)
        if st == "ENTER" and i == entered and progress > 0:
            cv2.rectangle(view, (cx, 34), (cx + int(30 * progress), 38), (255, 255, 255), -1)
    ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return jpg.tobytes() if ok else None


def publish(topic, payload):
    if mqtt_ok:
        client.publish(topic, payload)


def on_event(kind, text):
    print(f"LOCK: {text}")
    publish("lunchlock/event", text)


engine = LockEngine(on_event=on_event,
                    on_unlock=lambda: Bridge.notify("latch_open"),
                    on_lock=lambda: Bridge.notify("latch_close"))


# ── from the sketch ─────────────────────────────────────────────────────────

def on_button():
    with lock:
        engine.press_button()


def on_latch_state(pos_ms, moving):
    publish("lunchlock/state", f"{pos_ms},{moving}")


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")


Bridge.provide("button", on_button)
Bridge.provide("latch_state", on_latch_state)
Bridge.provide("status", on_status)


# ── camera + network + lock ─────────────────────────────────────────────────

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TRAIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TRAIN_H)
    return cap


def camera_loop():
    global snap_requested
    cap = open_camera()
    x, y, w, h = roi
    warned, bad, last_shown, last_preview, last_crop_sent = False, 0, None, 0.0, 0.0
    n, t_fps = 0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.mean() < 3:          # no image / black frames
            bad += 1
            status["camera"] = "no image"
            if bad >= 15:
                print(f"CAMERA: no image from /dev/video{CAMERA_INDEX} - reopening")
                cap.release()
                time.sleep(1)
                cap = open_camera()
                bad = 0
            time.sleep(0.1)
            continue
        bad = 0
        H, W = frame.shape[:2]
        if (W, H) != (TRAIN_W, TRAIN_H):
            # different resolution than training: scale the box so it covers the same area
            sx, sy = W / TRAIN_W, H / TRAIN_H
            x, y, w, h = int(roi[0] * sx), int(roi[1] * sy), int(roi[2] * sx), int(roi[3] * sy)
            if not warned:
                print(f"CAMERA: frames are {W}x{H}, not {TRAIN_W}x{TRAIN_H} - box scaled, accuracy may drop")
                warned = True
        status["camera"] = f"{W}x{H}"
        crop = frame[y:y + h, x:x + w]

        k, conf, _ = net.predict(crop)
        now = time.time()
        with lock:
            engine.update(k, conf, now)
            st = engine.state
            shown = (STATE_ID[st], len(engine.reader.digits),
                     int(engine.reader.progress * 10) * 10 if st == "ENTER" else 0)
        if shown != last_shown:                    # only redraw the LED matrix when it changes
            Bridge.notify("show_lock", *shown)
            last_shown = shown

        if time.time() < preview_until and time.monotonic() - last_preview >= PREVIEW_EVERY_S:
            last_preview = time.monotonic()
            jpg = preview_jpeg(frame, (x, y, w, h), st, k, conf, shown[1] if st == "ENTER" else 0,
                               engine.reader.progress)
            if jpg:
                globals()["latest_preview"] = jpg
                publish("lunchlock/preview", jpg)

        if mqtt_ok and time.time() < crops_until and time.monotonic() - last_crop_sent >= CROPS_EVERY_S:
            last_crop_sent = time.monotonic()
            ok_c, cjpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok_c:
                publish("lunchlock/crops", cjpg.tobytes())

        if snap_requested:
            snap_requested = False
            ok_j, jpg = cv2.imencode(".jpg", crop)
            if ok_j:
                publish("lunchlock/crop", jpg.tobytes())

        n += 1
        if time.monotonic() - t_fps >= 1.0:
            fps = n / (time.monotonic() - t_fps)
            n, t_fps = 0, time.monotonic()
            status.update(state=st, k=k, conf=round(conf, 2), fps=round(fps, 1),
                          entered=shown[1] if st == "ENTER" else 0)
            print(f"{st:8s} sees {k} fingers ({conf:.0%})  {fps:.1f} fps")
            publish("lunchlock/status", json.dumps(status))


# ── laptop link (optional) ──────────────────────────────────────────────────

def on_message(c, userdata, msg):
    global snap_requested, preview_until, crops_until
    text = msg.payload.decode(errors="ignore").strip().lower()
    if msg.topic == "lunchlock/button":
        on_button()
    elif msg.topic == "lunchlock/crops_on":
        crops_until = time.time() + 5
    elif msg.topic == "lunchlock/preview_on":
        preview_until = time.time() + 5          # the laptop re-sends this every 2 s while watching
    elif msg.topic == "lunchlock/snap":
        snap_requested = True
    elif msg.topic == "lunchlock/latch":
        parts = text.split()
        try:
            if parts[0] in ("open", "close", "stop", "home"):
                Bridge.notify(f"latch_{parts[0]}")
            elif parts[0] == "jog":
                Bridge.notify("latch_jog", int(parts[1]))
            elif parts[0] == "openms":
                Bridge.notify("latch_set_open", int(parts[1]))
        except (IndexError, ValueError):
            print(f"bad latch command '{text}'")


def on_connect(c, userdata, flags, reason_code, properties=None):
    global mqtt_ok
    mqtt_ok = True
    for t in ("lunchlock/button", "lunchlock/latch", "lunchlock/snap", "lunchlock/preview_on", "lunchlock/crops_on"):
        c.subscribe(t)
    print(f"MQTT: connected to the laptop at {MQTT_HOST}")


def on_disconnect(c, userdata, flags, reason_code, properties=None):
    global mqtt_ok
    mqtt_ok = False


def mqtt_loop():
    global client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    while True:
        try:
            socket.create_connection((MQTT_HOST, MQTT_PORT), timeout=3).close()
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
            client.loop_forever(retry_first_connection=True)
        except OSError:
            time.sleep(5)          # laptop not there - the box still works on its own


# ── phone page: http://<board IP>:7000 ──────────────────────────────────────
# Live camera view + the safe's button. On purpose there is NO open/close button here:
# anyone on the Wi-Fi can load this page, so the only way to open the lid stays your finger code.

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LunchLock</title>
<style>
body{margin:0;background:#0b0d12;color:#e9ecf2;font:16px system-ui,sans-serif;text-align:center}
h1{font-size:20px;margin:14px 0 4px} #st{color:#8b93a5;margin-bottom:10px;min-height:22px}
img{width:100%;max-width:640px;display:block;margin:0 auto;background:#000}
button{margin:18px auto;display:block;width:80%;max-width:360px;padding:22px;font-size:22px;font-weight:700;
color:#fff;border:0;border-radius:999px;background:linear-gradient(135deg,#ff6b6b,#e0344f)}
button:active{transform:translateY(2px)} p{color:#8b93a5;font-size:13px;padding:0 16px}
</style></head><body>
<h1>&#129386; LunchLock</h1><div id="st">connecting&hellip;</div>
<img src="/stream.mjpg" alt="LunchLock camera">
<button onclick="fetch('/button',{method:'POST'})">Press button</button>
<p>Press, then show your 3 numbers in the box &mdash; hold each until its box fills.</p>
<script>
async function poll(){try{const s=await (await fetch('/status')).json();
document.getElementById('st').textContent=s.state+' \\u00b7 sees '+s.k+' ('+Math.round(s.conf*100)+'%) \\u00b7 '+s.entered+'/3 \\u00b7 '+s.fps+' fps';}
catch(e){document.getElementById('st').textContent='board offline';} setTimeout(poll,500);} poll();
</script></body></html>"""


class PhoneHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global preview_until
        if self.path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/status":
            return self._send(json.dumps(status).encode(), "application/json")
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    preview_until = time.time() + 5          # keep previews coming while someone watches
                    jpg = latest_preview
                    if jpg is not None and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path == "/button":
            on_button()
            return self._send(b'{"ok":true}', "application/json")
        self._send(b"not found", "text/plain", 404)


def web_loop():
    try:
        server = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), PhoneHandler)
    except OSError as e:
        print(f"PHONE PAGE: could not start on port {WEB_PORT}: {e}")
        return
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                      # no data sent; just finds the Wi-Fi address
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        ip = "<board IP>"
    print(f"PHONE PAGE: open http://{ip}:{WEB_PORT} on your phone (same Wi-Fi)")
    server.serve_forever()


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=mqtt_loop, daemon=True).start()
threading.Thread(target=web_loop, daemon=True).start()
App.run()
