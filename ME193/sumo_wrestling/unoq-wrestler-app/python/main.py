# Sumo wrestler -- runs on the UNO Q's Linux side.
#
# sketch.ino -> Bridge "sense" (IMU) -> this file -> decision tree -> Bridge "drive" -> motors
# laptop (LEGO colour sensor over Bluetooth) -> MQTT "sumo/color" -> this file
# laptop -> MQTT "sumo/cmd" -> this file   (auto / stop / test)
# this file -> MQTT "sumo/telemetry" -> laptop (sumo_console.py)
#
# The robot boots in IDLE: motors off, sensors streaming. Nothing moves until
# the laptop sends "auto" (or "test").
import json
import math
import socket
import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "10.243.40.195"  # <-- the laptop's IP (run ipconfig on the laptop if it changes)
MQTT_PORT = 1883
TELEMETRY_TOPIC = "sumo/telemetry"
COLOR_TOPIC = "sumo/color"
CMD_TOPIC = "sumo/cmd"
STATUS_TOPIC = "sumo/status"

# --- Tuning -------------------------------------------------------------------
LOOP_HZ = 50            # decision tree rate (sketch failsafe is 300 ms)
TELEMETRY_HZ = 10
START_DELAY_S = 3.0     # sumo rule: wait after "auto" before moving
SEARCH_SPEED = (45, 35) # gentle forward curve while looking for the opponent
PUSH_SPEED = 100
HIT_ACCEL = 6.0         # m/s^2 of sudden change (beyond the slow average) = contact
PUSH_HOLD_S = 0.6       # keep pushing this long after the last hit
COLOR_STALE_S = 0.5     # no colour reading this long -> stop (edge sensor lost)
TEST_SPEED = 40

# --- State --------------------------------------------------------------------
mode = "idle"           # idle | starting | auto | test
mode_since = time.monotonic()
latest = None           # (ax, ay, az, gz) from the IMU
received = 0
lp = None               # slow average of the accel vector (gravity + tilt)
last_hit = -1e9
color = None            # last colour name from the laptop
blue = False
color_time = -1e9
decision = "idle"
connected = False
lock = threading.Lock()


# --- 1. sensor data ----------------------------------------------------------

def on_sense(ax, ay, az, gz, light_l=0, light_r=0):
    global latest, received, lp, last_hit
    with lock:
        latest = (ax, ay, az, gz)
        received += 1
        a = (ax, ay, az)
        if lp is None:
            lp = list(a)
        jolt = math.dist(a, lp)
        for k in range(3):
            lp[k] += 0.02 * (a[k] - lp[k])  # ~0.5 s time constant at 100 Hz
        if jolt > HIT_ACCEL and (ax or ay or az):
            last_hit = time.monotonic()


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")
    publish(STATUS_TOPIC, msg)


# --- 2. the decision tree ----------------------------------------------------
#
#   mode == idle?                         -> STOP
#   mode == test?                         -> motor test sequence
#   still in start delay?                 -> WAIT
#   colour sensor sees blue?              -> STOP (ring edge) and go idle
#   colour readings stale?                -> STOP (no edge sensor = not safe)
#   bumped within PUSH_HOLD_S?            -> PUSH (full forward)
#   otherwise                             -> SEARCH (forward curve)

def decide(now):
    """Return (left, right, leaf_name)."""
    if mode == "idle":
        return 0, 0, "idle"

    if mode == "test":
        t = now - mode_since
        for end, l, r, name in ((1, TEST_SPEED, 0, "test:left fwd"), (1.5, 0, 0, "test:pause"),
                                (2.5, 0, TEST_SPEED, "test:right fwd"), (3, 0, 0, "test:pause"),
                                (4, TEST_SPEED, TEST_SPEED, "test:both fwd")):
            if t < end:
                return l, r, name
        set_mode("idle")
        return 0, 0, "idle"

    if mode == "starting":
        if now - mode_since < START_DELAY_S:
            return 0, 0, "wait"
        set_mode("auto")

    if blue:
        set_mode("idle")
        publish(STATUS_TOPIC, "BLUE -- ring edge, stopped")
        return 0, 0, "edge:stop"

    if now - color_time > COLOR_STALE_S:
        return 0, 0, "no color data"

    if now - last_hit < PUSH_HOLD_S:
        return PUSH_SPEED, PUSH_SPEED, "push"

    return SEARCH_SPEED[0], SEARCH_SPEED[1], "search"


def set_mode(m):
    global mode, mode_since
    mode, mode_since = m, time.monotonic()
    print(f"MODE -> {m}")
    publish(STATUS_TOPIC, f"mode {m}")


def control_loop():
    global decision
    sent_zero = False
    while True:
        now = time.monotonic()
        with lock:
            left, right, leaf = decide(now)
            if leaf != "idle" or not decision.startswith("edge"):
                decision = leaf  # keep showing "edge:stop" after the edge stop
        if left or right:
            Bridge.notify("drive", int(left), int(right))
            sent_zero = False
        elif not sent_zero:
            Bridge.notify("drive", 0, 0)
            sent_zero = True
        time.sleep(1 / LOOP_HZ)


# --- 3. MQTT -----------------------------------------------------------------

def publish(topic, payload):
    if connected:
        client.publish(topic, payload)


def on_message(client, userdata, msg):
    global color, blue, color_time
    text = msg.payload.decode(errors="replace").strip()
    with lock:
        if msg.topic == COLOR_TOPIC:
            try:
                data = json.loads(text)
            except ValueError:
                return
            color, blue, color_time = data.get("color"), bool(data.get("blue")), time.monotonic()
            if blue and mode != "idle":
                Bridge.notify("drive", 0, 0)  # stop right now, don't wait for the next tick
            return

        cmd = text.lower()
        if cmd == "stop":
            set_mode("idle")
        elif cmd == "auto":
            set_mode("starting")
        elif cmd == "test":
            set_mode("test")
        else:
            publish(STATUS_TOPIC, f"unknown command '{cmd}'")


def on_connect(client, userdata, flags, reason_code, properties=None):
    global connected
    connected = True
    try:
        client.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError):
        pass
    print(f"LAPTOP CONNECTED ({MQTT_HOST}:{MQTT_PORT})")
    client.subscribe([(CMD_TOPIC, 0), (COLOR_TOPIC, 0)])
    client.publish(STATUS_TOPIC, "UNO Q sumo connected")


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    global connected
    connected = False
    print(f"lost the laptop ({reason_code}); retrying")


def telemetry_loop():
    global received
    last_print = 0.0
    while True:
        time.sleep(1 / TELEMETRY_HZ)
        with lock:
            rate, received = received * TELEMETRY_HZ, 0
            snap, dec, m, c, b = latest, decision, mode, color, blue
            color_ok = time.monotonic() - color_time < COLOR_STALE_S
            hit = time.monotonic() - last_hit < PUSH_HOLD_S
        if snap is None:
            if time.monotonic() - last_print > 2:
                print("no readings from the sketch yet -- is the sketch running?")
                last_print = time.monotonic()
            continue
        ax, ay, az, gz = snap
        msg = {"mode": m, "decision": dec, "rate": rate,
               "ax": round(ax, 2), "ay": round(ay, 2), "az": round(az, 2), "gz": round(gz, 2),
               "color": c if color_ok else None, "blue": b, "hit": hit}
        publish(TELEMETRY_TOPIC, json.dumps(msg))
        if time.monotonic() - last_print > 1:
            last_print = time.monotonic()
            print(f"{rate:3.0f}/s  mode={m:8s} decision={dec:13s} "
                  f"a=({ax:5.1f},{ay:5.1f},{az:5.1f}) gz={gz:5.2f} color={c if color_ok else '--'}  "
                  + ("LAPTOP OK" if connected else "no laptop"))


print("Sumo wrestler starting (IDLE -- motors off until the laptop sends 'auto' or 'test')")
Bridge.provide("sense", on_sense)
Bridge.provide("status", on_status)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="unoq-sumo")
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)
client.loop_start()

threading.Thread(target=control_loop, daemon=True).start()
threading.Thread(target=telemetry_loop, daemon=True).start()
App.run()
