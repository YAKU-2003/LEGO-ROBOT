"""
Laptop <-> UNO Q latch link over MQTT (same setup as the gait project).

Starts the Mosquitto broker on this laptop if it isn't running, sends latch commands
("open", "close", "jog 50", ...) on lunchlock/latch and keeps the latest position the
board reports on lunchlock/state.
"""

import socket
import subprocess
import time
from pathlib import Path

import paho.mqtt.client as mqtt

MOSQUITTO_EXE = r"C:\Program Files\mosquitto\mosquitto.exe"
MOSQUITTO_CONF = Path(__file__).resolve().parent / "mosquitto.conf"
CMD_TOPIC = "lunchlock/latch"
STATE_TOPIC = "lunchlock/state"


def ensure_broker(host="localhost", port=1883):
    def up():
        try:
            socket.create_connection((host, port), timeout=1).close()
            return True
        except OSError:
            return False
    if up():
        return
    if not Path(MOSQUITTO_EXE).exists():
        raise ConnectionError(f"Mosquitto not found at {MOSQUITTO_EXE}")
    subprocess.Popen([MOSQUITTO_EXE, "-c", str(MOSQUITTO_CONF)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(20):
        time.sleep(0.25)
        if up():
            print("[latch] started Mosquitto broker")
            return
    raise ConnectionError("Started Mosquitto but it isn't accepting connections")


class LatchLink:
    def __init__(self, host="localhost", port=1883, on_state=None, print_events=True):
        ensure_broker(host, port)
        self.print_events = print_events
        self.position_ms, self.moving, self.t_state = None, False, 0.0
        self.on_state = on_state
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.board_status, self.last_crop = None, None
        self.last_preview, self.t_preview, self.events = None, 0.0, []
        self.last_board_crop = None
        self.client.on_connect = lambda c, u, f, rc, p=None: c.subscribe(
            [(STATE_TOPIC, 0), ("lunchlock/status", 0), ("lunchlock/crop", 0), ("lunchlock/event", 0),
             ("lunchlock/preview", 0), ("lunchlock/crops", 0)])
        self.client.on_message = self._on_message
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def _on_message(self, client, userdata, msg):
        if msg.topic == "lunchlock/crop":
            self.last_crop = msg.payload
            return
        if msg.topic == "lunchlock/preview":
            self.last_preview, self.t_preview = msg.payload, time.time()
            return
        if msg.topic == "lunchlock/crops":
            self.last_board_crop = msg.payload
            return
        if msg.topic == "lunchlock/status":
            self.board_status, self.t_state = msg.payload.decode(), time.time()
            return
        if msg.topic == "lunchlock/event":
            text = msg.payload.decode()
            self.events.append((time.strftime("%H:%M:%S"), text))
            self.events = self.events[-6:]
            if self.print_events:
                print(f"\n[board] {text}")
            return
        try:
            pos, moving = msg.payload.decode().split(",")
            self.position_ms, self.moving, self.t_state = int(pos), moving == "1", time.time()
            if self.on_state:
                self.on_state(self.position_ms, self.moving)
        except ValueError:
            pass

    @property
    def board_online(self):
        return time.time() - self.t_state < 2.0     # the board reports 4x per second

    def send(self, command):
        self.client.publish(CMD_TOPIC, command)

    def press_button(self):
        self.client.publish("lunchlock/button", "press")

    def keep_preview_on(self):
        self.client.publish("lunchlock/preview_on", "1")

    def request_snap(self):
        self.last_crop = None
        self.client.publish("lunchlock/snap", "1")
