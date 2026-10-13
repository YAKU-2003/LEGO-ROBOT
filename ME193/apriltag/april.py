"""Drive the LEGO Double Motor up to a MOTORISED AprilTag, then have the tag
turn itself square to the camera.

The AprilTag is mounted on a single motor that can spin it about its vertical
axis. That splits the job cleanly between two machines:

    the car        centres the tag in the picture, and drives in until the
                   tag's TOP EDGE nearly touches the top of the yellow box
    the turntable  spins the tag until its face is square to the camera --
                   left and right edges the same height in the picture

These are independent, which is what makes this version simple. (Steering the
car alone could never do both: spinning the car moves the tag's bearing and its
apparent yaw by the same amount, so the two corrections cancel each other out.)

Order of operations each cycle: aim, rough-square, drive, fine-square. The
turntable gets a loose tolerance while the car is far away and a tight one once
the tag is large in the frame, because yaw read off a small distant tag is very
noisy -- a pixel of corner error is several degrees. A badly angled tag also
looks too narrow, which would fool the car into driving in too close, hence the
rough pass first.

Every motion is a short pulse: move, stop, wait for the lagging stream to catch
up, measure, repeat. A phone stream lags well behind the real hardware, so
continuous control would overshoot and hunt.

Getting the iPhone video onto this computer -- set CAMERA_SOURCE below:
  * A "phone as webcam" app (Camo, EpocCam, iVCam, DroidCam) makes it show up
    as a normal webcam: use its index (0, 1, 2 ... whichever it is).
  * An IP-camera app that serves the video over Wi-Fi: use its URL as a string,
    e.g. "http://192.168.1.23:8080/video".
"""

import math
import os
import statistics
import sys
import threading
import time
from collections import deque

import cv2
import legoeducation as le
import numpy as np
from cv2 import aruco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "poserace"))
import lelib  # noqa: E402
from lelib import doubleMotor  # noqa: E402

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 1  # webcam index, or a stream URL string. 1 = DroidCam here; 0 is the laptop webcam
MIRRORED = False  # set True if the app mirrors the picture (tags then can't be read)
CAMERA_HFOV_DEG = 65  # only used for the distance readout and the safety floor now
STREAM_STALL_SECONDS = 1.0  # warn (and stop) if no new frame arrives for this long

# ── Tag ──────────────────────────────────────────────────────────────────────
TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = 0  # the tag ID to go to
TAG_SIZE_CM = 10.0  # side of the tag's BLACK square, measured edge to edge

# ── Hardware pairing ─────────────────────────────────────────────────────────
CARD_COLOR = le.LEGO_COLOR_GREEN  # the car's Double Motor
CARD_SERIAL = "0994"
TAG_CARD_COLOR = le.LEGO_COLOR_GREEN  # the single motor the tag is mounted on
TAG_CARD_SERIAL = "0994"  # same physical Single Motor + card as elsewhere in this project

# ── Where to stop (all in pixels: no distance calibration needed) ────────────
FIT_MARGIN_PX = 20  # the yellow box: the picture, inset by this much
TOP_GAP_PX = 6  # park with the tag's top edge this far below the box's top edge
EDGE_MARGIN_PX = 10  # ... and never let the other three edges get closer than this
SAFETY_DISTANCE_CM = 8.0  # hard floor: stop if the tag gets this close, whatever the pixels say

# ── Squaring the tag up ──────────────────────────────────────────────────────
SQUARE_COARSE_DEG = 10.0  # good enough while the car is still approaching
SQUARE_FINE_DEG = 2.0  # once the tag is big in the frame, get it properly square
FINE_SQUARE_MIN_HEIGHT_PX = 120  # "big in the frame" means taller than this
TAG_SPIN_SPEED = 20  # percent power for the turntable motor
INITIAL_SPIN_RATE = 25.0  # deg/s guess at TAG_SPIN_SPEED; re-measured as it goes

# ── Which way the motors move things ─────────────────────────────────────────
AUTO_CALIBRATE = True  # measure the three signs below at startup instead of trusting them
DRIVE_SIGN = 1  # +1 if positive tank speed drives the way the camera looks
TURN_SIGN = 1  # +1 if (+speed, -speed) turns the car to the camera's right
SPIN_SIGN = 1  # +1 if positive turntable power reduces a positive yaw reading
AUTO_FLIP_ON_WRONG_WAY = True  # flip a sign mid-run if two pulses in a row make things worse

CAL_DRIVE_SECONDS = 0.5  # length of the self-test drive pulse
CAL_TURN_SECONDS = 0.5
CAL_SPIN_SECONDS = 0.4
CAL_MIN_DRIVE_CM = 1.5  # a test pulse that moves less than this tells us nothing
CAL_MIN_TURN_DEG = 2.0
CAL_MIN_SPIN_DEG = 2.0
CAL_SAFE_DISTANCE_CM = 45.0  # if closer than this, the self-test drives backward instead

# ── Pulse control ────────────────────────────────────────────────────────────
PULSE_CORRECTION = 0.5  # each pulse tries to remove this fraction of the error
MIN_PULSE_SECONDS = 0.08  # shortest nudge that actually breaks the motors free
MAX_PULSE_SECONDS = 0.8
# Stopped time after each pulse. Must be longer than the total lag (Bluetooth +
# stream + coasting), or it measures while things are still moving. Phone
# streams often lag 0.3 - 1 s: raise this if it keeps overshooting.
SETTLE_SECONDS = 1.0
SAMPLE_FRAMES = 5  # readings (median) used for each decision

# Aiming the car. Error is degrees the car must turn right.
TURN_STOP_DEG = 2.0
TURN_START_DEG = 5.0
TURN_SPEED = 15  # percent
INITIAL_TURN_RATE = 30.0  # deg/s guess at TURN_SPEED; re-measured as it goes

# Driving the car. Error is pixels of growing room still left in the frame.
RANGE_STOP_PX = 8.0
RANGE_START_PX = 18.0
DRIVE_SPEED = 30  # percent
INITIAL_DRIVE_RATE = 60.0  # px/s guess at DRIVE_SPEED; re-measured as it goes

# ── Searching for a lost tag ─────────────────────────────────────────────────
LOST_FRAMES_BEFORE_SEARCH = 3  # tolerate a few missed detections before giving up
FOUND_FRAMES_BEFORE_GO = 2  # ... and want a couple of good ones before trusting it
SEARCH_PULSE_SECONDS = 0.35  # rotate this long, then stop and look
SEARCH_SETTLE_SECONDS = 0.7  # ... long enough for the lagging stream to show the new view
SEARCH_HOP_AFTER_FULL_TURN = True  # after a full circle with nothing, move and try again
SEARCH_HOP_SECONDS = 0.6  # length of that forward hop

# A row / column counts as picture (not a black bar) if anything in it is
# brighter than this. Phone apps often letterbox with black bars.
BAR_BRIGHTNESS = 24

MOTOR_WATCHDOG_SECONDS = 1.0  # any motion command older than this is cancelled

# Live copies of the direction signs; the self-test writes to these.
SIGNS = {"drive": DRIVE_SIGN, "turn": TURN_SIGN, "spin": SPIN_SIGN}

detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector_params.adaptiveThreshWinSizeMax = 53
detector_params.polygonalApproxAccuracyRate = 0.05
detector_params.errorCorrectionRate = 0.8
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(TAG_FAMILY), detector_params)

CONNECT_TIMEOUT_SECONDS = 15  # give up and say so, rather than hang silently forever


def connect_with_timeout(label, device, card_serial, card_color):
    """device.connect(...) can hang forever with NO error if it never finds a
    matching device (the underlying library scans with no time limit) -- most
    often because that device still thinks it's connected from an earlier,
    uncleanly-ended session and has stopped advertising, or the card serial is
    wrong. Run the real connect call on a background thread and give up loudly
    after a bounded time instead of sitting there in silence."""
    result = {"done": False, "error": None}

    def worker():
        try:
            device.connect(card_serial=card_serial, card_color=card_color)
        except Exception as e:  # noqa: BLE001 -- reporting it, not swallowing it
            result["error"] = e
        finally:
            result["done"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=CONNECT_TIMEOUT_SECONDS)

    if not result["done"]:
        raise TimeoutError(
            f"{label}: no response after {CONNECT_TIMEOUT_SECONDS}s (card {card_serial}/{card_color}). "
            "It's most likely still holding a stale connection from an earlier run that didn't "
            "disconnect cleanly and has stopped advertising, or the card serial is wrong. "
            "Power-cycle this device (off, then on) and try again. The connect attempt is still "
            "running in the background and can't be cancelled -- restart this script after power-cycling."
        )
    if result["error"] is not None:
        raise result["error"]
    if not device.connected:
        raise ConnectionError(f"{label}: connect() returned without connecting (card {card_serial}/{card_color}).")


# ═════════════════════════════════════════════════════════════════════════════
# THE ONLY PLACE THE TAG MOTOR'S API IS TOUCHED
#
# I don't know lelib's exact single-motor class and method names, so these are
# a guess in the style of doubleMotor. If it errors out, fix the three lines
# marked below -- nothing else in the file talks to that motor. Running the
# script prints the names lelib actually exports, to help.
# ═════════════════════════════════════════════════════════════════════════════
class Turntable:
    """The single motor the AprilTag is mounted on. Sends a command only when
    it changes, and cancels stale ones."""

    def __init__(self):
        motor_class = getattr(lelib, "singleMotor", None)
        if motor_class is None:
            raise RuntimeError(
                "lelib has no 'singleMotor'. Names it does export:\n  "
                + ", ".join(n for n in dir(lelib) if not n.startswith("_"))
            )
        self.motor = motor_class()  # <-- FIX 1: constructor
        connect_with_timeout("Tag motor", self.motor, TAG_CARD_SERIAL, TAG_CARD_COLOR)  # <-- FIX 2
        self.last = 0
        self.sent_at = 0.0

    def _apply(self, speed):
        if speed == 0:
            self.motor.motor_stop()  # <-- FIX 3a: stop
        else:
            self.motor.motor_run(speed, blocking=False)  # <-- FIX 3b: run at a power

    def send(self, sign, status=""):
        """sign: +1 / -1 to spin, 0 to stop."""
        speed = int(sign * TAG_SPIN_SPEED * SIGNS["spin"])
        if speed == self.last:
            return
        self._apply(speed)
        self.last = speed
        self.sent_at = time.time()
        print(f"{status} -> tag motor {speed}%")

    def check_watchdog(self):
        if self.last != 0 and time.time() - self.sent_at > MOTOR_WATCHDOG_SECONDS:
            self.send(0, "watchdog")

    def stop(self):
        self._apply(0)
        self.last = 0


def camera_matrix(width, height):
    f = (width / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def turn_wheels(sign):
    """sign +1 = turn the car to the camera's right (tag then moves left in view)."""
    speed = int(sign * TURN_SPEED * SIGNS["turn"])
    return (speed, -speed)


def drive_wheels(sign):
    """sign +1 = drive the way the camera looks."""
    speed = int(sign * DRIVE_SPEED * SIGNS["drive"])
    return (speed, speed)


def tag_yaw_deg(corners, focal_px):
    """How far the tag's face is turned away from the camera, in degrees.
    Positive = its right edge (in the PICTURE) is nearer than its left.

    Measured from the two side edges rather than from solvePnP: a square tag
    seen nearly head-on has two almost equally good 3D poses, and which one the
    solver picks can flip frame to frame, which would send the turntable the
    wrong way. The edge heights have no such ambiguity -- right taller than
    left means one thing only.

    "Left" and "right" are taken in the image, not the tag's own frame, so a
    tag mounted sideways or upside-down still reads correctly: pick whichever
    pair of opposite edges is the more vertical.

    A tag parallel to the image plane has equal left and right edge heights.
    With r = (right-left)/(right+left), sin(yaw) = 2 * f * r / height.
    """
    c = corners
    pairs = (((0, 3), (1, 2)), ((0, 1), (3, 2)))
    edge_a, edge_b = max(pairs, key=lambda p: sum(abs(c[i][1] - c[j][1]) for i, j in p))
    mean_x = lambda e: (c[e[0]][0] + c[e[1]][0]) / 2
    left_edge, right_edge = (edge_a, edge_b) if mean_x(edge_a) < mean_x(edge_b) else (edge_b, edge_a)

    left = float(np.linalg.norm(c[left_edge[0]] - c[left_edge[1]]))
    right = float(np.linalg.norm(c[right_edge[0]] - c[right_edge[1]]))
    if left + right < 1e-6:
        return 0.0
    ratio = (right - left) / (right + left)
    sin_yaw = 2 * focal_px * ratio / ((left + right) / 2)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_yaw))))


class Target:
    """One sighting of the tag."""

    def __init__(self, corners, focal_px, width):
        self.corners = corners
        xs, ys = corners[:, 0], corners[:, 1]
        self.left, self.right = float(xs.min()), float(xs.max())
        self.top, self.bottom = float(ys.min()), float(ys.max())
        self.centre = (float(xs.mean()), float(ys.mean()))
        self.height_px = self.bottom - self.top

        # Bearing: how far off the camera's axis the tag sits, in degrees.
        self.bearing = math.degrees(math.atan((self.centre[0] - width / 2) / focal_px))
        self.yaw = tag_yaw_deg(corners, focal_px)

        # Rough range, for the safety floor and the readout only. The two
        # longest edges survive foreshortening best.
        edges = sorted(float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])) for i in range(4))
        side_px = (edges[2] + edges[3]) / 2
        self.dist = focal_px * TAG_SIZE_CM / max(side_px, 1e-6)


def find_target(frame, focal_px):
    """Return a Target for the wanted tag, or None. Draws all detections."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None
    aruco.drawDetectedMarkers(frame, corners, ids)
    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if tag_id == TARGET_TAG_ID:
            return Target(tag_corners[0].astype(np.float64), focal_px, frame.shape[1])
    return None


def approach_error_px(target, area, image_centre_y):
    """How many more pixels of growth the tag has room for. Returns
    (error, binding, reachable).

    The goal is the tag's top edge sitting TOP_GAP_PX under the top of the
    yellow box. Driving forward scales the whole picture away from its centre,
    so the top edge only climbs if it is already above the image centre -- if
    it isn't, no amount of driving will ever bring it to the top, and the
    caller is told so rather than driving into the tag forever.

    The other three edges just need to stay inside the box, so the smallest of
    the four numbers is what actually limits the approach.
    """
    left, top, right, bottom = area
    top_error = target.top - (top + TOP_GAP_PX)  # + = still below the line, keep going
    room_bottom = (bottom - EDGE_MARGIN_PX) - target.bottom
    room_left = target.left - (left + EDGE_MARGIN_PX)
    room_right = (right - EDGE_MARGIN_PX) - target.right

    reachable = target.top < image_centre_y or top_error <= 0
    candidates = [("top edge", top_error), ("bottom edge", room_bottom),
                  ("left edge", room_left), ("right edge", room_right)]
    binding, error = min(candidates, key=lambda c: c[1])
    return error, binding, reachable


class VisibleArea:
    """Finds the real picture inside the frame, ignoring black letterbox /
    pillarbox bars. Watches the first few frames (the bars stay put) and then
    locks the result in."""

    WATCH_FRAMES = 15

    def __init__(self):
        self.seen = 0
        self.row_max = None
        self.col_max = None
        self.box = None  # (left, top, right, bottom) in pixels once known

    def update(self, frame):
        if self.box is not None:
            return
        brightest = frame.max(axis=2)
        rows, cols = brightest.max(axis=1), brightest.max(axis=0)
        self.row_max = rows if self.row_max is None else np.maximum(self.row_max, rows)
        self.col_max = cols if self.col_max is None else np.maximum(self.col_max, cols)
        self.seen += 1
        if self.seen >= self.WATCH_FRAMES:
            self.box = self._measure(frame.shape[1], frame.shape[0])
            print(f"Visible picture area: x {self.box[0]}-{self.box[2]}, y {self.box[1]}-{self.box[3]}")

    def _measure(self, width, height):
        rows = np.flatnonzero(self.row_max > BAR_BRIGHTNESS)
        cols = np.flatnonzero(self.col_max > BAR_BRIGHTNESS)
        if len(rows) < height // 2 or len(cols) < width // 2:  # a very dark scene: don't trust it
            return (0, 0, width - 1, height - 1)
        return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))

    def get(self, frame):
        if self.box is not None:
            return self.box
        return (0, 0, frame.shape[1] - 1, frame.shape[0] - 1)


class FrameGrabber:
    """Reads the stream on a background thread and keeps only the newest frame,
    so we never process old frames the stream had queued up."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source!r}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._count += 1

    def latest(self):
        with self._lock:
            return self._frame, self._count

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class PulseController:
    """Drives one error toward zero with pulse / settle / measure cycles.

    Feed every frame to observe(); call step() only on the axis whose turn it
    is. step() returns (action, status):
      None  -> inside the stop zone, nothing to do
      0     -> hold still (settling or measuring)
      +1/-1 -> pulse now, correcting an error of that sign
    It learns how fast a pulse really moves things and sizes the next one to
    match."""

    def __init__(self, label, unit, stop_zone, start_zone, initial_rate):
        self.label = label
        self.unit = unit
        self.stop_zone = stop_zone
        self.start_zone = start_zone
        self.rate = initial_rate  # error units per second of pulse
        self.wrong_way_strikes = 0
        self.reset()

    def reset(self):
        """Forget the readings so far (the rate learned so far is kept)."""
        self.state = "idle"  # idle -> pulse -> settle -> idle
        self.done = False
        self.history = deque(maxlen=SAMPLE_FRAMES)
        self.calibrating = False
        self.pulse_sign = 0
        self.pulse_end = 0.0
        self.pulse_seconds = 0.0
        self.pulse_start_error = 0.0
        self.settle_until = 0.0

    @property
    def busy(self):
        return self.state != "idle"

    def observe(self, error, now):
        """Record a reading -- but only ones taken while everything is still."""
        if self.state == "pulse":
            self.history.clear()
        elif self.state == "settle" and now < self.settle_until:
            self.history.clear()
        else:
            self.history.append(error)

    def step(self, now):
        if self.state == "pulse":
            if now < self.pulse_end:
                return self.pulse_sign, f"{self.label} pulse"
            self.state = "settle"
            self.settle_until = now + SETTLE_SECONDS

        if self.state == "settle":
            if now < self.settle_until or len(self.history) < SAMPLE_FRAMES:
                return 0, f"{self.label}: settling"
            self.state = "idle"

        if len(self.history) < SAMPLE_FRAMES:
            return 0, f"{self.label}: measuring"

        measured = statistics.median(self.history)
        if self.calibrating:
            self._learn_rate(measured)
        size = abs(measured)
        self.done = size <= self.stop_zone or (self.done and size <= self.start_zone)
        if self.done:
            return None, ""

        self.pulse_seconds = min(
            MAX_PULSE_SECONDS, max(MIN_PULSE_SECONDS, PULSE_CORRECTION * size / self.rate)
        )
        self.pulse_sign = 1 if measured > 0 else -1
        self.pulse_start_error = measured
        self.pulse_end = now + self.pulse_seconds
        self.calibrating = True
        self.state = "pulse"
        return self.pulse_sign, f"{self.label} pulse ({measured:+.1f}{self.unit})"

    def _learn_rate(self, error_after):
        self.calibrating = False
        start = self.pulse_start_error
        moved = (start - error_after) * math.copysign(1.0, start)  # + = moved toward / past zero
        print(f"{self.label} pulse {self.pulse_seconds:.2f}s: {start:+.1f} -> {error_after:+.1f}{self.unit}")
        if moved < -self.stop_zone:
            self.wrong_way_strikes += 1
            print(f"  {self.label} got WORSE after moving (strike {self.wrong_way_strikes})")
        elif moved > 0:
            self.wrong_way_strikes = 0
            self.rate = 0.5 * self.rate + 0.5 * max(
                self.rate / 20, min(self.rate * 20, moved / self.pulse_seconds)
            )


class SignFinder:
    """Works out which way each motor moves things, by trying it.

    Three tests, each a measure / pulse / settle / measure cycle:
      drive -- does a positive tank speed make the tag nearer or further?
      turn  -- does (+speed, -speed) push the tag left or right in the picture?
      spin  -- does positive turntable power reduce a positive yaw reading?
    Whatever it finds goes into SIGNS, and the measured speeds seed the pulse
    controllers so the first real pulses are about the right length.
    """

    STAGES = ("drive", "turn", "spin", "done")

    def __init__(self, controllers):
        self.controllers = controllers  # {"drive": ctl, "turn": ctl, "spin": ctl}
        self.stage = "drive"
        self.state = "measure"  # measure -> pulse -> settle -> verify
        self.history = deque(maxlen=SAMPLE_FRAMES)
        self.until = 0.0
        self.before = 0.0
        self.wheels = (0, 0)
        self.spin = 0
        self.try_sign = 1
        self.attempts = 0

    @property
    def done(self):
        return self.stage == "done"

    def _reading(self, target):
        if self.stage == "drive":
            return target.dist
        if self.stage == "turn":
            return target.bearing
        return target.yaw

    def _seconds(self):
        base = {"drive": CAL_DRIVE_SECONDS, "turn": CAL_TURN_SECONDS, "spin": CAL_SPIN_SECONDS}[self.stage]
        return base * (1 + 0.6 * self.attempts)  # a bit longer each retry, if it barely moved

    def _floor(self):
        return {"drive": CAL_MIN_DRIVE_CM, "turn": CAL_MIN_TURN_DEG, "spin": CAL_MIN_SPIN_DEG}[self.stage]

    def _unit(self):
        return "cm" if self.stage == "drive" else "deg"

    def _build_command(self, target):
        self.wheels, self.spin = (0, 0), 0
        if self.stage == "drive":
            # Don't drive at the tag if we are already close: test in reverse.
            self.try_sign = -1 if target.dist < CAL_SAFE_DISTANCE_CM else 1
            self.wheels = drive_wheels(self.try_sign)
        elif self.stage == "turn":
            self.try_sign = 1
            self.wheels = turn_wheels(1)
        else:
            self.try_sign = 1
            self.spin = 1

    def step(self, target, now):
        """Returns (wheel command, turntable sign, status)."""
        label = self.stage
        reading = self._reading(target)

        if self.state == "measure":
            self.history.append(reading)
            if len(self.history) < SAMPLE_FRAMES:
                return (0, 0), 0, f"self-test: reading {label} baseline"
            self.before = statistics.median(self.history)
            self.history.clear()
            self._build_command(target)
            self.until = now + self._seconds()
            self.state = "pulse"
            return self.wheels, self.spin, f"self-test: {label} pulse"

        if self.state == "pulse":
            if now < self.until:
                return self.wheels, self.spin, f"self-test: {label} pulse"
            self.state = "settle"
            self.until = now + SETTLE_SECONDS
            self.history.clear()
            return (0, 0), 0, f"self-test: {label} settling"

        if self.state == "settle":
            self.history.clear()
            if now < self.until:
                return (0, 0), 0, f"self-test: {label} settling"
            self.state = "verify"
            return (0, 0), 0, f"self-test: checking {label}"

        # verify
        self.history.append(reading)
        if len(self.history) < SAMPLE_FRAMES:
            return (0, 0), 0, f"self-test: checking {label}"
        after = statistics.median(self.history)
        moved = (self.before - after) * self.try_sign  # + = the command did what we hoped
        self._judge(moved, after)
        return (0, 0), 0, f"self-test: {label} done"

    def _judge(self, moved, after):
        stage, seconds, unit = self.stage, self._seconds(), self._unit()
        print(f"self-test {stage}: {self.before:+.1f} -> {after:+.1f}{unit} over {seconds:.2f}s")

        if abs(moved) < self._floor():
            self.attempts += 1
            if self.attempts <= 2:
                print("  barely moved - retrying with a longer pulse")
                self.state = "measure"
                self.history.clear()
                return
            print(f"  still barely moved - keeping SIGNS['{stage}'] = {SIGNS[stage]}. "
                  "Check the motor is free and the tag is really in view.")
        elif moved < 0:
            SIGNS[stage] *= -1
            print(f"  wrong way round - flipping SIGNS['{stage}'] to {SIGNS[stage]}")
        else:
            print(f"  correct as-is (SIGNS['{stage}'] = {SIGNS[stage]})")
            rate = abs(moved) / seconds
            ctl = self.controllers[stage]
            if stage != "drive":  # the drive controller works in pixels, not cm
                ctl.rate = max(ctl.rate / 5, min(ctl.rate * 5, rate))
            print(f"  measured {rate:.1f} {unit}/s")

        self.attempts = 0
        self.history.clear()
        self.state = "measure"
        self.stage = self.STAGES[self.STAGES.index(stage) + 1]
        if self.done:
            print(f"Self-test finished: drive {SIGNS['drive']:+d}, "
                  f"turn {SIGNS['turn']:+d}, spin {SIGNS['spin']:+d}")


class Searcher:
    """Hunts for a tag that isn't in view: rotate a pulse, stop, look, repeat.
    After a full circle with nothing, optionally hop forward and circle again
    (a tag can be hidden behind something from one spot but not the next)."""

    def __init__(self):
        self.direction = 1  # +1 = turn right
        self.reset()

    def reset(self):
        self.state = "turn"
        self.until = 0.0
        self.degrees_turned = 0.0

    def aim_at(self, bearing):
        """Search toward wherever the tag was last seen."""
        if abs(bearing) > 0.5:
            self.direction = 1 if bearing > 0 else -1

    def step(self, now, turn_rate):
        if now < self.until:
            if self.state == "turn":
                return turn_wheels(self.direction), "searching: turning"
            if self.state == "hop":
                return drive_wheels(1), "searching: moving to a new spot"
            return (0, 0), "searching: looking"

        if self.state == "turn":
            self.degrees_turned += turn_rate * SEARCH_PULSE_SECONDS
            self.state = "look"
            self.until = now + SEARCH_SETTLE_SECONDS
            return (0, 0), "searching: looking"

        if self.state == "look":
            if SEARCH_HOP_AFTER_FULL_TURN and self.degrees_turned >= 360.0:
                self.degrees_turned = 0.0
                self.state = "hop"
                self.until = now + SEARCH_HOP_SECONDS
                return drive_wheels(1), "searching: moving to a new spot"
            self.state = "turn"
            self.until = now + SEARCH_PULSE_SECONDS
            return turn_wheels(self.direction), "searching: turning"

        # finished a hop
        self.state = "look"
        self.until = now + SEARCH_SETTLE_SECONDS
        return (0, 0), "searching: looking"


class Drive:
    """The car's wheels. Sends commands only when they change, and cancels any
    motion command that has gone stale (stream stalled, program hung, ...)."""

    def __init__(self, dm):
        self.dm = dm
        self.last = (0, 0)
        self.sent_at = 0.0

    def send(self, cmd, status=""):
        cmd = (int(cmd[0]), int(cmd[1]))
        if cmd == self.last:
            return
        if cmd == (0, 0):
            self.dm.movement_stop()
        else:
            self.dm.movement_move_tank(cmd[0], cmd[1], blocking=False)
        self.last = cmd
        self.sent_at = time.time()
        print(f"{status} -> L {cmd[0]}%  R {cmd[1]}%")

    def check_watchdog(self):
        if self.last != (0, 0) and time.time() - self.sent_at > MOTOR_WATCHDOG_SECONDS:
            self.send((0, 0), "watchdog")

    def stop(self):
        self.dm.movement_stop()
        self.last = (0, 0)


def check_wrong_way(ctl, key):
    """If an axis keeps making things worse, its sign is backwards: flip it."""
    if not AUTO_FLIP_ON_WRONG_WAY or ctl.wrong_way_strikes < 2:
        return
    SIGNS[key] *= -1
    ctl.wrong_way_strikes = 0
    ctl.reset()
    print(f"*** {ctl.label} kept moving the wrong way - flipping SIGNS['{key}'] to {SIGNS[key]} ***")


def main():
    grabber = FrameGrabber(CAMERA_SOURCE)
    dm = None
    turntable = None
    try:
        print("Connecting to the car's Double Motor...")
        dm = doubleMotor()
        connect_with_timeout("Car's Double Motor", dm, CARD_SERIAL, CARD_COLOR)
        print("Connecting to the tag's motor...")
        turntable = Turntable()
        print("Connected! Press 'q' to quit.")

        drive = Drive(dm)
        turn_ctl = PulseController("aim", " deg", TURN_STOP_DEG, TURN_START_DEG, INITIAL_TURN_RATE)
        range_ctl = PulseController("approach", " px", RANGE_STOP_PX, RANGE_START_PX, INITIAL_DRIVE_RATE)
        spin_ctl = PulseController("square", " deg", SQUARE_COARSE_DEG, SQUARE_COARSE_DEG * 2, INITIAL_SPIN_RATE)
        searcher = Searcher()
        sign_finder = SignFinder({"drive": range_ctl, "turn": turn_ctl, "spin": spin_ctl}) \
            if AUTO_CALIBRATE else None
        visible = VisibleArea()

        focal = None
        mode = "search"  # search -> calibrate -> run
        last_count = 0
        last_frame_time = time.time()
        stalled = False
        frames_lost = 0
        frames_found = 0
        warned_unreachable = False
        status = "waiting for video"
        readout = ""

        while True:
            frame, count = grabber.latest()
            now = time.time()

            if frame is None or count == last_count:
                # No new picture yet: keep the watchdogs and window alive.
                if now - last_frame_time > STREAM_STALL_SECONDS and not stalled:
                    stalled = True
                    print("Video stream stalled - motors will stop.")
                    drive.send((0, 0), "stream stalled")
                    turntable.send(0, "stream stalled")
                drive.check_watchdog()
                turntable.check_watchdog()
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            last_count = count
            last_frame_time = now
            stalled = False
            if MIRRORED:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if focal is None:
                focal = float(camera_matrix(width, height)[0, 0])
            visible.update(frame)
            area = visible.get(frame)

            target = find_target(frame, focal)

            if target is None:
                frames_found = 0
                frames_lost += 1
                if frames_lost >= LOST_FRAMES_BEFORE_SEARCH and mode != "search":
                    print("Tag lost - searching.")
                    mode = "search"
                    for ctl in (turn_ctl, range_ctl, spin_ctl):
                        ctl.reset()
                    drive.send((0, 0), "lost")
                    turntable.send(0, "lost")
                if mode == "search":
                    cmd, status = searcher.step(now, turn_ctl.rate)
                    drive.send(cmd, status)
                    turntable.send(0)
                else:
                    status = "tag missed - holding"
                    drive.send((0, 0), status)
                    turntable.send(0)
                readout = "tag --"
            else:
                frames_lost = 0

                if mode == "search":
                    # Stop the sweep the moment it appears, then make sure it
                    # is really there before setting off.
                    drive.send((0, 0), "tag spotted")
                    frames_found += 1
                    status = "tag spotted - confirming"
                    if frames_found >= FOUND_FRAMES_BEFORE_GO:
                        searcher.reset()
                        for ctl in (turn_ctl, range_ctl, spin_ctl):
                            ctl.reset()
                        mode = "calibrate" if (sign_finder and not sign_finder.done) else "run"
                        print("Tag confirmed - " + ("running the direction self-test."
                                                    if mode == "calibrate" else "approaching."))

                gap, binding, reachable = approach_error_px(target, area, height / 2)
                if not reachable and not warned_unreachable:
                    warned_unreachable = True
                    print("The tag's top edge sits below the middle of the picture, so driving "
                          "closer pushes it DOWN, not up - it can never reach the top of the box. "
                          "Raise the tag or lower the camera.")
                if reachable:
                    warned_unreachable = False

                readout = (f"bearing {target.bearing:+.1f}  yaw {target.yaw:+.1f}  "
                           f"gap {gap:+.0f}px ({binding})  ~{target.dist:.0f}cm")

                if mode == "calibrate":
                    wheels, spin, status = sign_finder.step(target, now)
                    drive.send(wheels, status)
                    turntable.send(spin, status)
                    if sign_finder.done:
                        mode = "run"
                        for ctl in (turn_ctl, range_ctl, spin_ctl):
                            ctl.reset()

                elif mode == "run":
                    # Tighten the squaring tolerance once the tag is big enough
                    # in the frame for the yaw reading to be worth trusting.
                    fine = target.height_px >= FINE_SQUARE_MIN_HEIGHT_PX
                    spin_ctl.stop_zone = SQUARE_FINE_DEG if fine else SQUARE_COARSE_DEG
                    spin_ctl.start_zone = spin_ctl.stop_zone * 2

                    # Don't creep any closer than the safety floor.
                    drive_error = gap if (reachable and target.dist > SAFETY_DISTANCE_CM) else 0.0

                    turn_ctl.observe(target.bearing, now)
                    spin_ctl.observe(target.yaw, now)
                    range_ctl.observe(drive_error, now)

                    wheels, spin = (0, 0), 0
                    # Finish whatever is already mid-pulse; otherwise work
                    # through aim -> square -> approach in order.
                    if turn_ctl.busy:
                        action, status = turn_ctl.step(now)
                        wheels = turn_wheels(action) if action else (0, 0)
                    elif spin_ctl.busy:
                        action, status = spin_ctl.step(now)
                        spin = action if action else 0
                    elif range_ctl.busy:
                        action, status = range_ctl.step(now)
                        wheels = drive_wheels(action) if action else (0, 0)
                    else:
                        action, status = turn_ctl.step(now)
                        if action is not None:
                            wheels = turn_wheels(action) if action else (0, 0)
                        else:
                            action, status = spin_ctl.step(now)
                            if action is not None:
                                spin = action if action else 0
                            else:
                                action, status = range_ctl.step(now)
                                if action is None:
                                    status = "ON TARGET" if fine else "in position - squaring up"
                                else:
                                    wheels = drive_wheels(action) if action else (0, 0)

                    drive.send(wheels, status)
                    turntable.send(spin, status)
                    check_wrong_way(turn_ctl, "turn")
                    check_wrong_way(range_ctl, "drive")
                    check_wrong_way(spin_ctl, "spin")

                searcher.aim_at(target.bearing)  # if we lose it, hunt that way first

            drive.check_watchdog()
            turntable.check_watchdog()

            # Overlay: the yellow box, the line the top edge is aiming for,
            # the image centre, and the readouts.
            m = FIT_MARGIN_PX
            cv2.rectangle(frame, (area[0] + m, area[1] + m), (area[2] - m, area[3] - m), (0, 255, 255), 1)
            goal_y = area[1] + m + TOP_GAP_PX
            cv2.line(frame, (area[0] + m, goal_y), (area[2] - m, goal_y), (255, 0, 255), 1)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 0, 255), 1)
            if target is not None:
                cv2.circle(frame, (int(target.centre[0]), int(target.centre[1])), 6, (0, 255, 0), -1)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, readout, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"L {drive.last[0]}%  R {drive.last[1]}%  tag {turntable.last}%   "
                               f"signs d{SIGNS['drive']:+d} t{SIGNS['turn']:+d} s{SIGNS['spin']:+d}",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.imshow("AprilTag Follow (iPhone)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always stop everything and release it, even on error.
        if dm is not None:
            dm.movement_stop()
            dm.disconnect()
        if turntable is not None:
            try:
                turntable.stop()
                turntable.motor.disconnect()
            except Exception:
                pass
        grabber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
