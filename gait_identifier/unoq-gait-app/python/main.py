# Gait wearable -- runs on the UNO Q's Linux side.
#
# sketch.ino -> Bridge "imu" -> this file -> MQTT "gait/imu" -> the laptop (record.py / live.py)
# live.py -> MQTT "gait/prediction" ("2,3" = model1 walking, model2 running) -> Bridge "show_pred" -> LED matrix
#
# The console (App Lab's Python output) shows, every second:
#   SENSORS  what the MPU6050 reads right now   <- check this first
#   LAPTOP   whether the readings are reaching the laptop
import socket
import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "192.168.4.231"  # <-- the laptop's IP (run ipconfig on the laptop if it changes)
MQTT_PORT = 1883
IMU_TOPIC = "gait/imu"
PREDICTION_TOPIC = "gait/prediction"
STATUS_TOPIC = "gait/status"
PRINT_EVERY_S = 1.0

connected = False
latest = None          # last reading from the sketch
peak_a = 0.0           # biggest |a| since the last print (so steps show up)
received = 0           # readings from the sketch since the last print
sent = 0               # readings sent to the laptop since the last print
mqtt_problem = "not connected yet"


# ── 1. readings from the sketch ──────────────────────────────────────────────

def on_imu(ax, ay, az, gx, gy, gz):
    global latest, received, sent, peak_a
    latest = (ax, ay, az, gx, gy, gz)
    received += 1
    peak_a = max(peak_a, (ax * ax + ay * ay + az * az) ** 0.5)
    if connected:
        # The board's own timestamp lets the laptop ignore Wi-Fi jitter and spot dropped samples.
        client.publish(IMU_TOPIC, f"{time.monotonic():.4f},{ax:.3f},{ay:.3f},{az:.3f},"
                                  f"{gx:.3f},{gy:.3f},{gz:.3f}", qos=0)
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
        mqtt_problem = "laptop reachable but no broker -- start record.py or live.py on the laptop"
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
    # Send every reading immediately instead of letting TCP bundle small messages (Nagle).
    try:
        client.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError) as e:
        print(f"(couldn't set TCP_NODELAY: {e})")
    print(f"LAPTOP CONNECTED ({MQTT_HOST}:{MQTT_PORT})")
    client.subscribe(PREDICTION_TOPIC)
    client.publish(STATUS_TOPIC, "UNO Q gait wearable connected")


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    global connected, mqtt_problem
    connected = False
    mqtt_problem = f"lost the laptop ({reason_code}); retrying"


def on_connect_fail(client, userdata):
    global mqtt_problem
    if not check_laptop_reachable():
        return
    mqtt_problem = "laptop reachable but MQTT connect failed; retrying"


def on_prediction(client, userdata, msg):
    try:
        m1, m2 = (int(v) for v in msg.payload.decode().split(","))
    except ValueError:
        return
    Bridge.notify("show_pred", m1, m2)


# ── 3. the console report ────────────────────────────────────────────────────

def report():
    global received, sent, peak_a
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
            ax, ay, az, gx, gy, gz = latest
            bar = "#" * min(30, int(peak_a / 9.81 * 3))  # 3 # per g
            print(f"SENSORS {rate:3.0f}/s  a=({ax:6.2f},{ay:6.2f},{az:6.2f})  peak |a|={peak_a / 9.81:4.1f} g {bar:<30}"
                  f"||  LAPTOP: " + (f"sending {sent / PRINT_EVERY_S:.0f}/s" if connected else mqtt_problem))
        received = 0
        sent = 0
        peak_a = 0.0


print("Gait wearable starting")
Bridge.provide("imu", on_imu)
Bridge.provide("status", on_status)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="unoq-gait")
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_connect_fail = on_connect_fail
client.on_message = on_prediction
check_laptop_reachable()
client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)  # keeps retrying until the laptop's broker is up
client.loop_start()

threading.Thread(target=report, daemon=True).start()
App.run()
