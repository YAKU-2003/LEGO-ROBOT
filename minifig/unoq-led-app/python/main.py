# Runs on the UNO Q's Linux side. Subscribes to the laptop's MQTT broker and
# tells the microcontroller (sketch.ino) where to draw the dot via Bridge.
import json

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "192.168.1.100"  # <-- CHANGE to your laptop's IP (run ipconfig on the laptop)
MQTT_PORT = 1883
TOPIC = "minifig/pos"

MATRIX_COLS = 13
MATRIX_ROWS = 8


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
    except ValueError:
        return

    if not payload.get("found"):
        Bridge.notify("clear_matrix")
        return

    col = min(MATRIX_COLS - 1, max(0, int(payload["x"] * MATRIX_COLS)))
    row = min(MATRIX_ROWS - 1, max(0, int(payload["y"] * MATRIX_ROWS)))
    Bridge.notify("show_dot", col, row)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to {MQTT_HOST}:{MQTT_PORT} (rc={reason_code})")
    client.subscribe(TOPIC)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)  # keeps retrying if broker isn't up yet
client.loop_start()

print(f"Waiting for messages on '{TOPIC}'")
App.run()
