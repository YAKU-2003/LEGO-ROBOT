# Ping-pong paddle -- runs on the UNO Q's Linux side.
#
# sketch.ino -> Bridge "imu" -> this file -> MQTT -> the laptop's game
# The game -> MQTT "pingpong/paddle/feedback" -> this file -> Bridge "show_icon" -> LED matrix
#
# The console (App Lab's Python output) shows, every half second:
#   SENSORS  what the MPU6050 + light sensor read right now   <- check this first
#   LAPTOP   whether the readings are reaching the laptop
import socket
import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "192.168.4.26"  # <-- the laptop's IP (run ipconfig on the laptop if it changes)
MQTT_PORT = 1883
IMU_TOPIC = "pingpong/paddle/imu"
FEEDBACK_TOPIC = "pingpong/paddle/feedback"
STATUS_TOPIC = "pingpong/paddle/status"
PRINT_EVERY_S = 0.5

# feedback name -> (icon id in sketch.ino, seconds to show it; 0 = keep)
ICONS = {
    "clear": (0, 0), "hit": (1, 0.4), "miss": (2, 0.8), "smash": (3, 0.6), "point": (4, 1.0),
    "win": (5, 0), "lose": (6, 0), "select": (7, 0.6), "serve": (7, 0.3),
}

connected = False
latest = None          # last reading from the sketch
peak_g = 0.0           # biggest gyro |g| since the last print (so swings show up)
received = 0           # readings from the sketch since the last print
sent = 0               # readings sent to the laptop since the last print
mqtt_problem = "not connected yet"
clear_timer = None


# ── 1. readings from the sketch ──────────────────────────────────────────────

def on_imu(ax, ay, az, gx, gy, gz, light):
    global latest, received, sent, peak_g
    latest = (ax, ay, az, gx, gy, gz, light)
    received += 1
    peak_g = max(peak_g, (gx * gx + gy * gy + gz * gz) ** 0.5)
    if connected:
        # The board's own timestamp lets the laptop undo Wi-Fi jitter.
        client.publish(IMU_TOPIC, f"{time.monotonic():.4f},{ax:.3f},{ay:.3f},{az:.3f},"
                                  f"{gx:.3f},{gy:.3f},{gz:.3f},{int(light)}", qos=0)
        sent += 1


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")
    if connected:
        client.publish(STATUS_TOPIC, msg)


# ── 2. the laptop (MQTT) ─────────────────────────────────────────────────────

def check_laptop_reachable():
    """Plain TCP test so the console says WHY the laptop can't be reached."""
    global mqtt_problem
    try:
        with socket.create_connection((MQTT_HOST, MQTT_PORT), timeout=3):
            print(f"NETWORK OK: the laptop's broker at {MQTT_HOST}:{MQTT_PORT} is reachable")
            return True
    except ConnectionRefusedError:
        mqtt_problem = "laptop reachable but no broker -- start pingpong.py or test_paddle.py on the laptop"
    except socket.timeout:
        mqtt_problem = (f"no answer from {MQTT_HOST} -- wrong IP in MQTT_HOST, board on a different Wi-Fi, "
                        "or the laptop firewall is blocking port 1883")
    except OSError as e:
        mqtt_problem = f"can't reach {MQTT_HOST}: {e}"
    print(f"NETWORK PROBLEM: {mqtt_problem}")
    return False


def on_connect(client, userdata, flags, reason_code, properties=None):
    global connected, mqtt_problem
    connected = True
    mqtt_problem = ""
    # Send every reading immediately instead of letting TCP bundle small messages
    # together (Nagle's algorithm) -- that bundling made swings arrive ~60 ms late.
    try:
        client.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError) as e:
        print(f"(couldn't set TCP_NODELAY: {e})")
    print(f"LAPTOP CONNECTED ({MQTT_HOST}:{MQTT_PORT})")
    client.subscribe(FEEDBACK_TOPIC)
    client.publish(STATUS_TOPIC, "UNO Q paddle connected")


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    global connected, mqtt_problem
    connected = False
    mqtt_problem = f"lost the laptop ({reason_code}); retrying"


def on_connect_fail(client, userdata):
    global mqtt_problem
    if not check_laptop_reachable():
        return
    mqtt_problem = "laptop reachable but MQTT connect failed; retrying"


def on_feedback(client, userdata, msg):
    global clear_timer
    icon, hold = ICONS.get(msg.payload.decode(errors="replace"), (0, 0))
    Bridge.notify("show_icon", icon)
    if clear_timer is not None:
        clear_timer.cancel()
        clear_timer = None
    if hold:
        clear_timer = threading.Timer(hold, lambda: Bridge.notify("show_icon", 0))
        clear_timer.start()


# ── 3. the console report ────────────────────────────────────────────────────

SWING_THRESHOLD = 5.8  # rad/s, from the laptop's calibration (paddle_config_unoq.json) -- display only


def report():
    global received, sent, peak_g
    quiet = 0
    while True:
        time.sleep(PRINT_EVERY_S)
        rate = received / PRINT_EVERY_S
        if latest is None or received == 0:
            quiet += 1
            if quiet % 4 == 1:
                print("SENSORS: no readings from the sketch yet -- is the sketch running? "
                      "(an 'MPU6050 NOT FOUND' message means check SDA/SCL wiring)")
        else:
            quiet = 0
            ax, ay, az, gx, gy, gz, light = latest
            amag = (ax * ax + ay * ay + az * az) ** 0.5
            bar = "#" * min(30, int(peak_g * 2))  # 1 # per 0.5 rad/s
            swing = "  <-- SWING" if peak_g >= SWING_THRESHOLD else ""
            print(f"SENSORS {rate:3.0f}/s  |a|={amag:5.2f}  peak |g|={peak_g:5.2f} rad/s {bar:<30}{swing}  "
                  f"||  LAPTOP: " + (f"sending {sent / PRINT_EVERY_S:.0f}/s" if connected else mqtt_problem))
        received = 0
        sent = 0
        peak_g = 0.0


print("Ping-pong paddle starting")
Bridge.provide("imu", on_imu)
Bridge.provide("status", on_status)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="unoq-paddle")
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_connect_fail = on_connect_fail
client.on_message = on_feedback
check_laptop_reachable()
client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)  # keeps retrying until the laptop's broker is up
client.loop_start()

threading.Thread(target=report, daemon=True).start()
App.run()
