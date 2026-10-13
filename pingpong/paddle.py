"""Ping-pong paddles: anything with an IMU that can tell the game "I swung".

    UnoQPaddle      Arduino UNO Q + MPU6050 + light sensor (the default).
                    The board's app (pingpong/unoq-paddle-app) streams the IMU
                    over MQTT to a Mosquitto broker on this laptop; the game
                    sends back icons for the board's LED matrix (hit / miss /
                    ...). Covering the light sensor = a "serve / rematch" button.
    Paddle          LEGO Education Double Motor over Bluetooth (built-in IMU),
                    buzzes its motors for haptic feedback.
    KeyboardPaddle  no hardware: SPACE = swing, X = smash, C = light-sensor button.

All of them share IMUPaddle's swing detection: a swing is the gyro magnitude
rising through a threshold (with hysteresis + a refractory period). Each paddle
type has its own calibration file, written by test_paddle.py, because the units
differ (LEGO raw units vs MPU6050 rad/s).

Usage:
    paddle = UnoQPaddle()
    paddle.connect()
    paddle.swings_between(t0, t1)   # -> list of swing times
    paddle.buzz("hit")
    paddle.close()
"""

import json
import math
import os
import queue
import socket
import subprocess
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
SWING_REFRACTORY_S = 0.35  # ignore new swings this soon after the last one


# ── calibration files ────────────────────────────────────────────────────────

def config_path(kind):
    return os.path.join(HERE, "paddle_config.json" if kind == "lego" else f"paddle_config_{kind}.json")


CONFIG_PATH = config_path("lego")  # (old name, kept for anything that imports it)


def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_threshold(path=CONFIG_PATH):
    """Swing threshold saved by test_paddle.py, or None if not calibrated yet."""
    try:
        return float(load_config(path)["swing_threshold"])
    except (KeyError, ValueError, TypeError):
        return None


def load_typical_peak(path=CONFIG_PATH):
    """Median swing peak recorded by test_paddle.py calibration, or None."""
    peaks = sorted(load_config(path).get("swing_peaks") or [])
    return float(peaks[len(peaks) // 2]) if peaks else None


def save_config(data, path=CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── shared IMU + swing detection ─────────────────────────────────────────────

class IMUPaddle:
    """Sample buffer + swing detection. Subclasses feed it with _add_sample()."""

    kind = "imu"
    name = "IMU"
    has_button = False  # a "serve / rematch" button (the UNO Q's light sensor)
    latency = 0.0       # how late (s) this paddle's readings typically reach us; the game waits this long
    DEFAULT_THRESHOLD = 1.0

    def __init__(self, swing_threshold=None):
        self.config_path = config_path(self.kind)
        self.calibrated = load_threshold(self.config_path) is not None
        self.threshold = swing_threshold or load_threshold(self.config_path) or self.DEFAULT_THRESHOLD
        # Your normal swing's peak gyro (calibration puts the threshold at ~40% of it).
        self.typical_peak = load_typical_peak(self.config_path) or self.threshold * 2.5

        self.samples = deque(maxlen=2000)  # (t, ax, ay, az, gx, gy, gz, gyro_mag)
        self.swings = deque(maxlen=50)     # monotonic times of detected swings
        self._lock = threading.Lock()
        self._above = False
        self._last_swing = -1.0
        self._quiet_until = 0.0            # ignore swings until then (e.g. while buzzing)

    def _add_sample(self, t, ax, ay, az, gx, gy, gz):
        if any(math.isnan(v) for v in (ax, ay, az, gx, gy, gz)):
            return
        gmag = math.sqrt(gx * gx + gy * gy + gz * gz)
        with self._lock:
            self.samples.append((t, ax, ay, az, gx, gy, gz, gmag))
        # Swing = rising edge of gyro magnitude through the threshold.
        if gmag > self.threshold and not self._above:
            self._above = True
            if t - self._last_swing > SWING_REFRACTORY_S and t > self._quiet_until:
                self._last_swing = t
                with self._lock:
                    self.swings.append(t)
        elif gmag < self.threshold * 0.6:  # hysteresis
            self._above = False

    def latest(self):
        """Most recent sample (t, ax, ay, az, gx, gy, gz, gyro_mag) or None."""
        with self._lock:
            return self.samples[-1] if self.samples else None

    def gyro_mag(self):
        s = self.latest()
        return s[7] if s else 0.0

    def accel_mag(self):
        s = self.latest()
        return math.sqrt(s[1] ** 2 + s[2] ** 2 + s[3] ** 2) if s else 0.0

    def samples_since(self, t0):
        with self._lock:
            return [s for s in self.samples if s[0] >= t0]

    def swing_between(self, t0, t1):
        """Earliest swing time in [t0, t1], or None."""
        with self._lock:
            for t in self.swings:
                if t0 <= t <= t1:
                    return t
        return None

    def swings_between(self, t0, t1):
        with self._lock:
            return [t for t in self.swings if t0 <= t <= t1]

    def swing_power(self, t, window_s=0.12):
        """How hard the swing that started at time t was, compared to your
        normal swing: peak gyro magnitude around t / typical calibrated peak
        (1.0 = your usual swing, 1.4 = 40% faster)."""
        with self._lock:
            peak = max((s[7] for s in self.samples if t - 0.05 <= s[0] <= t + window_s), default=0.0)
        return peak / self.typical_peak if self.typical_peak else 0.0

    def serve_pressed(self):
        """True once each time the 'serve' button is pressed (paddles without one: never)."""
        return False


# ── Arduino UNO Q + MPU6050 + light sensor (over MQTT) ───────────────────────

MOSQUITTO_EXE = r"C:\Program Files\mosquitto\mosquitto.exe"
MOSQUITTO_CONF = os.path.join(HERE, "mosquitto.conf")
IMU_TOPIC = "pingpong/paddle/imu"            # board -> laptop: "t,ax,ay,az,gx,gy,gz,light"
FEEDBACK_TOPIC = "pingpong/paddle/feedback"  # laptop -> board: "hit", "miss", ...
STATUS_TOPIC = "pingpong/paddle/status"      # board -> laptop: text messages

# Light sensor as a button. On this paddle the reading goes UP when it's covered.
LIGHT_COVERED_ABOVE = 165  # reading above this = covered (pressed); at or below = not covered
LIGHT_HYSTERESIS = 5       # must drop to 165 - 5 = 160 to count as uncovered again (no flicker)
LIGHT_HOLD_SAMPLES = 4     # must stay covered this many readings (~80 ms) -- ignores blips


def _port_open(host, port, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class UnoQPaddle(IMUPaddle):
    kind = "unoq"
    name = "UNO Q"
    has_button = True
    DEFAULT_THRESHOLD = 4.0  # rad/s (~230 deg/s); run test_paddle.py to calibrate

    def __init__(self, broker="localhost", port=1883, swing_threshold=None, start_broker=True):
        super().__init__(swing_threshold)
        self.broker, self.port = broker, port
        self.start_broker = start_broker
        self.light = None
        self.rate = 0.0               # IMU samples per second actually arriving
        self._covered = False
        self._dark_count = 0
        self._serve_presses = 0
        self._last_rx = 0.0
        self._offsets = deque(maxlen=300)  # laptop_time - board_time, for de-jittering
        self._extra = deque(maxlen=250)    # how much later than the fastest packet each one arrived
        self._broker_proc = None
        self._client = None
        self._rx_count, self._rx_t0 = 0, time.monotonic()

    # The board stamps each sample with its own clock. Wi-Fi delivers packets in
    # bursts, so instead of "time it arrived" we use board_time + the smallest
    # offset seen recently (= the least-delayed packet), which keeps the real spacing.
    def _laptop_time(self, board_t, now):
        self._offsets.append(now - board_t)
        t = board_t + min(self._offsets)
        self._extra.append(now - t)
        return t

    @property
    def latency(self):
        """90th-percentile Wi-Fi delay of recent readings (seconds, capped at 0.3).
        The game waits this long before deciding you didn't swing."""
        if len(self._extra) < 20:
            return 0.1
        extra = sorted(self._extra)
        return min(0.3, extra[int(0.9 * (len(extra) - 1))])

    @property
    def connected(self):
        return time.monotonic() - self._last_rx < 1.0

    def ensure_broker(self):
        """Start Mosquitto (listening on the network) if nothing is on the MQTT port yet."""
        if _port_open(self.broker, self.port):
            return
        if self.broker not in ("localhost", "127.0.0.1") or not self.start_broker:
            raise ConnectionError(f"No MQTT broker at {self.broker}:{self.port}")
        if not os.path.exists(MOSQUITTO_EXE):
            raise ConnectionError(f"Mosquitto not found at {MOSQUITTO_EXE}; install it or start a broker yourself")
        log = open(os.path.join(HERE, "mosquitto.log"), "w")
        self._broker_proc = subprocess.Popen([MOSQUITTO_EXE, "-c", MOSQUITTO_CONF, "-v"], stdout=log,
                                             stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(30):
            if _port_open(self.broker, self.port):
                print(f"[unoq] started Mosquitto broker (log: pingpong/mosquitto.log)")
                return
            time.sleep(0.1)
        raise ConnectionError("Started Mosquitto but it isn't accepting connections (see pingpong/mosquitto.log)")

    def connect(self, wait_s=0.0):
        """Connect to the broker; optionally wait up to wait_s for the board's data."""
        import paho.mqtt.client as mqtt

        self.ensure_broker()
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pingpong-laptop-{os.getpid()}")
        self._client.on_connect = lambda c, u, f, rc, p=None: c.subscribe([(IMU_TOPIC, 0), (STATUS_TOPIC, 0)])
        self._client.on_message = self._on_message
        self._client.connect(self.broker, self.port, keepalive=10)
        self._client.loop_start()
        t0 = time.monotonic()
        while wait_s and not self.connected and time.monotonic() - t0 < wait_s:
            time.sleep(0.05)
        return self

    def _on_message(self, client, userdata, msg):
        if msg.topic == STATUS_TOPIC:
            print(f"[unoq] {msg.payload.decode(errors='replace')}")
            return
        try:
            bt, ax, ay, az, gx, gy, gz, light = (float(v) for v in msg.payload.decode().split(","))
        except ValueError:
            return
        now = time.monotonic()
        self._last_rx = now
        self._add_sample(self._laptop_time(bt, now), ax, ay, az, gx, gy, gz)

        self._update_light(light)

        self._rx_count += 1
        if now - self._rx_t0 >= 1.0:
            self.rate, self._rx_count, self._rx_t0 = self._rx_count / (now - self._rx_t0), 0, now

    def _update_light(self, light):
        """Light sensor as a button: above LIGHT_COVERED_ABOVE = covered (one press per cover)."""
        self.light = light
        if not self._covered:
            if light > LIGHT_COVERED_ABOVE:
                self._dark_count += 1
                if self._dark_count >= LIGHT_HOLD_SAMPLES:
                    self._covered = True
                    self._serve_presses += 1
            else:
                self._dark_count = 0
        elif light < LIGHT_COVERED_ABOVE - LIGHT_HYSTERESIS:
            self._covered = False
            self._dark_count = 0

    @property
    def covered(self):
        return self._covered

    def serve_pressed(self):
        if self._serve_presses:
            self._serve_presses = 0
            return True
        return False

    def buzz(self, pattern="hit"):
        """No motor on this paddle: show an icon on the UNO Q's LED matrix instead."""
        if self._client is not None:
            self._client.publish(FEEDBACK_TOPIC, pattern, qos=0)

    def close(self):
        if self._client is not None:
            try:
                self._client.publish(FEEDBACK_TOPIC, "clear", qos=0)
                time.sleep(0.05)
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        if self._broker_proc is not None:
            self._broker_proc.terminate()
            self._broker_proc = None


# ── LEGO Education Double Motor (Bluetooth) ──────────────────────────────────

NOTIFY_MS = 20        # LEGO IMU update period in ms (library allows 15..1000)
HAPTIC_QUIET_S = 0.15  # ignore swings this long after a buzz ends (buzzing shakes the IMU)

# Each pattern is a list of (clockwise?, speed %, duration ms) steps.
# Quick direction flips feel like a buzz in your hand.
HAPTIC_PATTERNS = {
    "hit":    [(True, 80, 60), (False, 80, 60)],
    "miss":   [(True, 100, 140), (False, 100, 140), (True, 100, 140), (False, 100, 140)],
    "serve":  [(True, 40, 50)],
    "select": [(True, 60, 60), (False, 60, 60), (True, 60, 60)],
    "smash":  [(True, 100, 90), (False, 100, 90), (True, 100, 90)],
    "point":  [(True, 70, 50), (False, 70, 50), (True, 90, 90)],
    "win":    [(True, 50, 80), (False, 70, 80), (True, 90, 80), (False, 100, 200)],
    "lose":   [(True, 100, 400), (False, 100, 300)],
}

COLOR_NAMES = ("red", "yellow", "blue", "green", "purple", "magenta", "azure", "orange")


class Paddle(IMUPaddle):
    kind = "lego"
    name = "LEGO"
    DEFAULT_THRESHOLD = 300.0  # LEGO raw gyro units

    def __init__(self, card_serial=None, card_color=None, swing_threshold=None):
        import legoeducation as le  # only needed for this paddle

        super().__init__(swing_threshold)
        self.le = le
        self.card_serial = card_serial
        if card_color in COLOR_NAMES:
            card_color = getattr(le, f"LEGO_COLOR_{card_color.upper()}")
        self.card_color = card_color
        self.dm = le.DoubleMotor()
        self.connected = False
        self._haptics = queue.Queue(maxsize=3)
        self._running = False

    def connect(self):
        """Connect over Bluetooth (no card = first Double Motor found)."""
        for attempt in range(5):
            try:
                self.dm.connect(card_serial=self.card_serial, card_color=self.card_color,
                                device_notification_delay=NOTIFY_MS)
                break
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                else:
                    raise
        if not self.dm.connected:
            raise ConnectionError("Could not connect to the Double Motor.")
        self.connected = True
        self._running = True
        threading.Thread(target=self._imu_loop, daemon=True).start()
        threading.Thread(target=self._haptic_loop, daemon=True).start()
        return self

    def close(self):
        self._running = False
        if self.connected:
            try:
                self.dm.motor_stop(motor=self.le.MOTOR_BOTH)
            except Exception:
                pass
            self.dm.disconnect()
            self.connected = False

    def _imu_loop(self):
        last = None
        while self._running:
            imu = self.dm.imu_device
            if imu is not last:  # the library swaps in a new object per notification
                last = imu
                self._add_sample(time.monotonic(), imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ,
                                 imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
            time.sleep(0.004)

    def buzz(self, pattern="hit"):
        """Queue a haptic pattern; returns immediately. Dropped if busy."""
        try:
            self._haptics.put_nowait(pattern)
        except queue.Full:
            pass

    def _haptic_loop(self):
        le = self.le
        while self._running:
            try:
                name = self._haptics.get(timeout=0.2)
            except queue.Empty:
                continue
            steps = HAPTIC_PATTERNS.get(name, HAPTIC_PATTERNS["hit"])
            total_s = sum(ms for _, _, ms in steps) / 1000.0
            self._quiet_until = time.monotonic() + total_s + 0.3
            try:
                for cw, speed, ms in steps:
                    direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if cw else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                    self.dm.motor_run_for_time(ms, direction=direction, motor=le.MOTOR_BOTH,
                                               speed=speed, blocking=True)
            except Exception as e:
                print(f"[paddle] haptic error: {e}")
            self._quiet_until = time.monotonic() + HAPTIC_QUIET_S


# ── no hardware ──────────────────────────────────────────────────────────────

class KeyboardPaddle:
    """Stand-in paddle for testing without hardware: SPACE = swing, X = smash, C = light sensor."""

    kind = "keyboard"
    name = "keyboard"
    has_button = True  # C key
    latency = 0.0
    connected = True
    calibrated = True
    threshold = 1.0
    typical_peak = 2.5

    def __init__(self):
        self.swings = deque(maxlen=50)
        self.powers = {}  # swing time -> power
        self._serve = False

    def connect(self, *args, **kwargs):
        return self

    def close(self):
        pass

    def fake_swing(self, power=1.0):
        t = time.monotonic()
        self.swings.append(t)
        self.powers[t] = power

    def press_serve(self):
        self._serve = True

    def serve_pressed(self):
        pressed, self._serve = self._serve, False
        return pressed

    def swing_power(self, t, window_s=0.12):
        return self.powers.get(t, 1.0)

    def gyro_mag(self):
        if self.swings and time.monotonic() - self.swings[-1] < 0.15:
            return self.typical_peak * self.powers.get(self.swings[-1], 1.0)
        return 0.0

    def swing_between(self, t0, t1):
        for t in self.swings:
            if t0 <= t <= t1:
                return t
        return None

    def swings_between(self, t0, t1):
        return [t for t in self.swings if t0 <= t <= t1]

    def buzz(self, pattern="hit"):
        print(f"[buzz] {pattern}")
