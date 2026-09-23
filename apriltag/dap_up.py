"""Search for an AprilTag, drive up to it, and "dap up" with a 3D-printed
LEGO hand -- using ONLY the tag's 3D pose (no hand/skeleton tracking).

This reuses aajnf.py's proven approach machinery wholesale (PulseController,
SignFinder self-test, Searcher, Drive watchdog, the pulse/settle/measure
cycle -- all tuned on this same rig) and swaps its pixel-based "fit the tag
in a box" aiming for real 3D pose from solvePnP: bearing in degrees (for
steering) and Z distance in cm (for the drive/stop decision). The only new
behaviour is what happens once it arrives: DAP_DISTANCE_CM away, aimed and
stopped, it rotates slightly toward the hand to make contact, then rotates
back -- one "dap".

Mode flow: search -> calibrate (direction self-test, once) -> run (aim +
approach) -> dap (one-shot motor sequence) -> done.
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
from lelib import doubleMotor  # noqa: E402

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 1  # webcam index, or a stream URL string
MIRRORED = False  # set True if the app mirrors the picture (tags then can't be read)
# The phone is mounted vertically (portrait), so every frame needs rotating
# before anything else touches it. cv2.ROTATE_90_CLOCKWISE,
# cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, or None for no rotation.
ROTATE_90 = cv2.ROTATE_90_CLOCKWISE
CAMERA_HFOV_DEG = 65  # the PHYSICAL camera's horizontal FOV; with ROTATE_90 set,
# this is measured along what is now the frame's SHORT (vertical) edge -- re-measure
# it for accurate distance readings if 65 was only ever measured for landscape.

# ── Tag ──────────────────────────────────────────────────────────────────────
TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = 0  # the tag ID to approach (e.g. worn on a glove)
TAG_SIZE_CM = 10.0  # side of the tag's BLACK square, measured edge to edge

# ── Hardware pairing ─────────────────────────────────────────────────────────
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0994"
CONNECT_TIMEOUT_SECONDS = 15

# ── Dap thresholds ───────────────────────────────────────────────────────────
DAP_DISTANCE_CM = 18.0  # the Z distance where the hook hand reaches theirs
MIN_SAFE_DISTANCE_CM = 8.0  # hard floor: never drive closer than this even if pose is glitchy

# ── Which way the motors move things ─────────────────────────────────────────
AUTO_CALIBRATE = True  # measure the two signs below at startup instead of trusting them
DRIVE_SIGN = 1  # +1 if positive tank speed drives the way the camera looks
TURN_SIGN = 1  # +1 if (+speed, -speed) turns the car to the camera's right
AUTO_FLIP_ON_WRONG_WAY = True  # flip a sign mid-run if two pulses in a row make things worse

CAL_DRIVE_SECONDS = 0.5  # length of the self-test drive pulse
CAL_TURN_SECONDS = 0.5
CAL_MIN_DRIVE_CM = 1.5  # a test pulse that moves less than this tells us nothing
CAL_MIN_TURN_DEG = 2.0
CAL_SAFE_DISTANCE_CM = 45.0  # if closer than this, the self-test drives backward instead

# ── Pulse control ────────────────────────────────────────────────────────────
PULSE_CORRECTION = 0.5  # each pulse tries to remove this fraction of the error
MIN_PULSE_SECONDS = 0.08  # shortest nudge that actually breaks the motors free
MAX_PULSE_SECONDS = 0.4  # short -- this hardware moves more per second than modeled
SETTLE_SECONDS = 1.5  # time for BLE command / motor coast to finish before measuring
SAMPLE_FRAMES = 5  # readings (median) used for each decision

# Aiming the car. Error is degrees the car must turn right.
TURN_STOP_DEG = 2.0
TURN_START_DEG = 5.0
TURN_SPEED = 8  # percent -- slow and gentle, see aajnf.py's tuning notes
INITIAL_TURN_RATE = 30.0  # deg/s guess at TURN_SPEED; re-measured as it goes
MAX_TURN_DEGREES_PER_PULSE = 8.0  # hard cap so one pulse can't swing the tag out of frame

# Driving the car. Error is cm of distance still to close before DAP_DISTANCE_CM.
RANGE_STOP_CM = 2.0
RANGE_START_CM = 6.0
DRIVE_SPEED = 15  # percent
INITIAL_DRIVE_RATE = 10.0  # cm/s guess at DRIVE_SPEED; re-measured as it goes

# ── Searching for a lost tag ─────────────────────────────────────────────────
LOST_FRAMES_BEFORE_SEARCH = 3  # tolerate a few missed detections before giving up
FOUND_FRAMES_BEFORE_GO = 2  # ... and want a couple of good ones before trusting it
SEARCH_PULSE_SECONDS = 0.35  # rotate this long, then stop and look
SEARCH_SETTLE_SECONDS = 0.7  # ... long enough for the lagging stream to show the new view
SEARCH_HOP_AFTER_FULL_TURN = True  # after a full circle with nothing, move and try again
SEARCH_HOP_SECONDS = 0.6  # length of that forward hop

MOTOR_WATCHDOG_SECONDS = 1.0  # any motion command older than this is cancelled

# ── The dap motion itself ─────────────────────────────────────────────────────
# TODO tune on hardware: rotate slightly toward the hand to make contact with
# the hook, then rotate back. Punchier than the gentle aiming TURN_SPEED above
# since this is meant to read as a deliberate gesture, not a careful nudge.
DAP_TURN_SPEED = 28  # percent
DAP_TURN_SECONDS = 0.35  # how long to hold the "into the hand" turn
DAP_PAUSE_SECONDS = 0.2  # brief hold at full extension before rotating back
DAP_TURN_SIGN = -1  # -1 rotates opposite turn_wheels(+1); flip back to +1 if it dips the wrong way
PRE_DAP_PAUSE_SECONDS = 2.5  # hold still and announce before actually dapping

# Live copies of the direction signs; the self-test writes to these.
SIGNS = {"drive": DRIVE_SIGN, "turn": TURN_SIGN}

detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(TAG_FAMILY), detector_params)


# ── Connection helper (copied from aajnf.py) ─────────────────────────────────
def connect_with_timeout(label, device, card_serial, card_color):
    """device.connect(...) can hang forever with NO error if it never finds a
    matching device -- most often a stale connection from an earlier,
    uncleanly-ended session. Run the real connect call on a background thread
    and give up loudly after a bounded time instead of sitting there in
    silence."""
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
            "Power-cycle this device (off, then on) and try again."
        )
    if result["error"] is not None:
        raise result["error"]
    if not device.connected:
        raise ConnectionError(f"{label}: connect() returned without connecting (card {card_serial}/{card_color}).")


# ── Camera / pose math ───────────────────────────────────────────────────────
def camera_matrix(width, height):
    """Pinhole camera intrinsics K, estimated from the horizontal FOV. Good
    enough for steering/depth -- not a substitute for a real calibration if
    you need precise measurements elsewhere."""
    f = (width / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_tag_pose(corners, K, tag_size_cm):
    """The standard AprilTag/ArUco 3D pose recipe, via solvePnP:

    1. Define the tag's 4 corners in the tag's OWN flat coordinate frame,
       centered at the origin with Z = 0 (corner order must match the order
       detectMarkers() returns: top-left, top-right, bottom-right, bottom-left).
    2. solvePnP finds the rigid transform (rvec, tvec) that maps those object
       points onto where the corners actually landed in the image, given K.
    3. tvec = [X, Y, Z] is the tag's position in CAMERA coordinates, in the
       same units as tag_size_cm (cm here):
         X = left(-) / right(+) offset  -> steering
         Y = up(-) / down(+) offset     -> unused for a ground robot
         Z = straight-ahead depth       -> throttle / stop condition

    Returns (x_cm, y_cm, z_cm), or None if solvePnP fails.
    """
    half = tag_size_cm / 2.0
    obj_points = np.array([
        [-half,  half, 0.0],  # top-left
        [ half,  half, 0.0],  # top-right
        [ half, -half, 0.0],  # bottom-right
        [-half, -half, 0.0],  # bottom-left
    ], dtype=np.float64)
    dist_coeffs = np.zeros(5)  # assume negligible lens distortion

    ok, _rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist_coeffs)
    if not ok:
        return None
    x_cm, y_cm, z_cm = tvec.flatten()
    return x_cm, y_cm, z_cm


class Pose:
    """One sighting of the tag, in real-world cm/degrees rather than pixels.

    .dist and .bearing exist specifically so PulseController/SignFinder --
    written against aajnf.py's pixel-based Target -- work here unchanged."""

    def __init__(self, x_cm, y_cm, z_cm):
        self.x_cm = x_cm
        self.y_cm = y_cm
        self.z_cm = z_cm
        self.dist = z_cm
        # + = tag to the right of the camera's axis (matches aajnf.py's bearing sign).
        self.bearing = math.degrees(math.atan2(x_cm, z_cm))


def find_target(frame, K):
    """Return a Pose for the wanted tag, or None. Draws all detections."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None
    aruco.drawDetectedMarkers(frame, corners, ids)
    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if tag_id == TARGET_TAG_ID:
            pose = estimate_tag_pose(tag_corners[0].astype(np.float64), K, TAG_SIZE_CM)
            return Pose(*pose) if pose is not None else None
    return None


def turn_wheels(sign):
    """sign +1 = turn the car to the camera's right (tag then moves left in view)."""
    speed = int(sign * TURN_SPEED * SIGNS["turn"])
    return (speed, -speed)


def drive_wheels(sign):
    """sign +1 = drive the way the camera looks."""
    speed = int(sign * DRIVE_SPEED * SIGNS["drive"])
    return (speed, speed)


class PulseController:
    """Drives one error toward zero with pulse / settle / measure cycles.

    Feed every frame to observe(); call step() only on the axis whose turn it
    is. step() returns (action, status):
      None  -> inside the stop zone, nothing to do
      0     -> hold still (settling or measuring)
      +1/-1 -> pulse now, correcting an error of that sign
    It learns how fast a pulse really moves things and sizes the next one to
    match."""

    def __init__(self, label, unit, stop_zone, start_zone, initial_rate, max_pulse_units=None):
        self.label = label
        self.unit = unit
        self.stop_zone = stop_zone
        self.start_zone = start_zone
        self.rate = initial_rate  # error units per second of pulse
        self.max_pulse_units = max_pulse_units  # hard cap on error-units moved per pulse, if set
        self.wrong_way_strikes = 0
        self.calibrating = False
        self.reset()

    def reset(self):
        """Forget the readings so far (the rate learned so far is kept).

        If a pulse had already fired but never got to report its outcome
        (self.calibrating is still True -- e.g. the tag went out of view
        before settling/measuring could finish), that's itself evidence
        something is wrong: count it as a wrong-way strike. Otherwise a pulse
        that keeps knocking the tag out of frame never gets measured, so
        AUTO_FLIP_ON_WRONG_WAY never sees enough strikes to fire."""
        if self.calibrating:
            self.wrong_way_strikes += 1
            print(f"  {self.label} pulse never resolved before the tag was lost "
                  f"(strike {self.wrong_way_strikes})")
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

        pulse_cap = MAX_PULSE_SECONDS
        if self.max_pulse_units is not None:
            pulse_cap = min(pulse_cap, self.max_pulse_units / self.rate)
        self.pulse_seconds = min(
            pulse_cap, max(MIN_PULSE_SECONDS, PULSE_CORRECTION * size / self.rate)
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

    Two tests, each a measure / pulse / settle / measure cycle:
      drive -- does a positive tank speed make the tag nearer or further?
      turn  -- does (+speed, -speed) push the tag left or right in the picture?
    Whatever it finds goes into SIGNS, and the measured speeds seed the pulse
    controllers so the first real pulses are about the right length."""

    STAGES = ("drive", "turn", "done")

    def __init__(self, controllers):
        self.controllers = controllers  # {"drive": ctl, "turn": ctl}
        self.stage = "drive"
        self.state = "measure"  # measure -> pulse -> settle -> verify
        self.history = deque(maxlen=SAMPLE_FRAMES)
        self.until = 0.0
        self.before = 0.0
        self.wheels = (0, 0)
        self.try_sign = 1
        self.attempts = 0

    @property
    def done(self):
        return self.stage == "done"

    def _reading(self, target):
        return target.dist if self.stage == "drive" else target.bearing

    def _seconds(self):
        base = {"drive": CAL_DRIVE_SECONDS, "turn": CAL_TURN_SECONDS}[self.stage]
        return base * (1 + 0.6 * self.attempts)  # a bit longer each retry, if it barely moved

    def _floor(self):
        return {"drive": CAL_MIN_DRIVE_CM, "turn": CAL_MIN_TURN_DEG}[self.stage]

    def _unit(self):
        return "cm" if self.stage == "drive" else "deg"

    def _build_command(self, target):
        if self.stage == "drive":
            # Don't drive at the tag if we are already close: test in reverse.
            self.try_sign = -1 if target.dist < CAL_SAFE_DISTANCE_CM else 1
            self.wheels = drive_wheels(self.try_sign)
        else:
            self.try_sign = 1
            self.wheels = turn_wheels(1)

    def step(self, target, now):
        """Returns (wheel command, status)."""
        label = self.stage
        reading = self._reading(target)

        if self.state == "measure":
            self.history.append(reading)
            if len(self.history) < SAMPLE_FRAMES:
                return (0, 0), f"self-test: reading {label} baseline"
            self.before = statistics.median(self.history)
            self.history.clear()
            self._build_command(target)
            self.until = now + self._seconds()
            self.state = "pulse"
            return self.wheels, f"self-test: {label} pulse"

        if self.state == "pulse":
            if now < self.until:
                return self.wheels, f"self-test: {label} pulse"
            self.state = "settle"
            self.until = now + SETTLE_SECONDS
            self.history.clear()
            return (0, 0), f"self-test: {label} settling"

        if self.state == "settle":
            self.history.clear()
            if now < self.until:
                return (0, 0), f"self-test: {label} settling"
            self.state = "verify"
            return (0, 0), f"self-test: checking {label}"

        # verify
        self.history.append(reading)
        if len(self.history) < SAMPLE_FRAMES:
            return (0, 0), f"self-test: checking {label}"
        after = statistics.median(self.history)
        moved = (self.before - after) * self.try_sign  # + = the command did what we hoped
        self._judge(moved, after)
        return (0, 0), f"self-test: {label} done"

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
            if stage != "drive":  # the drive controller works in cm, not deg
                ctl.rate = max(ctl.rate / 5, min(ctl.rate * 5, rate))
            print(f"  measured {rate:.1f} {unit}/s")

        self.attempts = 0
        self.history.clear()
        self.state = "measure"
        self.stage = self.STAGES[self.STAGES.index(stage) + 1]
        if self.done:
            print(f"Self-test finished: drive {SIGNS['drive']:+d}, turn {SIGNS['turn']:+d}")


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
    ctl.calibrating = False  # this flip already accounts for the strikes; don't double-count on reset
    ctl.reset()
    print(f"*** {ctl.label} kept moving the wrong way - flipping SIGNS['{key}'] to {SIGNS[key]} ***")


# ── Frame grabber (copied from aajnf.py) ─────────────────────────────────────
class FrameGrabber:
    """Reads the stream on a background thread and keeps only the newest
    frame, so we never process old frames the stream had queued up."""

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


# ── The dap-up motion ─────────────────────────────────────────────────────────
def dap_up_sequence(dm):
    """Rotate slightly toward the hand to make contact with the hook, hold
    briefly, then rotate back to center. One "dap"."""
    speed = int(DAP_TURN_SIGN * DAP_TURN_SPEED * SIGNS["turn"])
    dm.movement_move_tank(speed, -speed, blocking=False)
    time.sleep(DAP_TURN_SECONDS)
    dm.movement_stop()
    time.sleep(DAP_PAUSE_SECONDS)
    dm.movement_move_tank(-speed, speed, blocking=False)
    time.sleep(DAP_TURN_SECONDS)
    dm.movement_stop()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    grabber = FrameGrabber(CAMERA_SOURCE)
    dm = None
    try:
        print("Connecting to the car's Double Motor...")
        dm = doubleMotor()
        connect_with_timeout("Car's Double Motor", dm, CARD_SERIAL, CARD_COLOR)
        print("Connected! Press 'q' to quit.")

        drive = Drive(dm)
        turn_ctl = PulseController("aim", " deg", TURN_STOP_DEG, TURN_START_DEG, INITIAL_TURN_RATE,
                                    max_pulse_units=MAX_TURN_DEGREES_PER_PULSE)
        range_ctl = PulseController("approach", " cm", RANGE_STOP_CM, RANGE_START_CM, INITIAL_DRIVE_RATE)
        searcher = Searcher()
        sign_finder = SignFinder({"drive": range_ctl, "turn": turn_ctl}) if AUTO_CALIBRATE else None

        K = None
        mode = "search"  # search -> calibrate -> run -> dap -> done
        last_count = 0
        frames_lost = 0
        frames_found = 0
        status = "waiting for video"
        readout = ""

        while True:
            frame, count = grabber.latest()
            now = time.time()

            if frame is None or count == last_count:
                drive.check_watchdog()
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            last_count = count
            if ROTATE_90 is not None:
                frame = cv2.rotate(frame, ROTATE_90)
            if MIRRORED:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if K is None:
                K = camera_matrix(width, height)

            pose = find_target(frame, K)

            if pose is None:
                frames_found = 0
                frames_lost += 1
                if frames_lost >= LOST_FRAMES_BEFORE_SEARCH and mode not in ("search", "dap", "done"):
                    print("Tag lost - searching.")
                    mode = "search"
                    for ctl in (turn_ctl, range_ctl):
                        ctl.reset()
                    drive.send((0, 0), "lost")
                if mode == "search":
                    cmd, status = searcher.step(now, turn_ctl.rate)
                    drive.send(cmd, status)
                elif mode not in ("dap", "done"):
                    status = "tag missed - holding"
                    drive.send((0, 0), status)
                readout = "tag --"
            else:
                frames_lost = 0

                if mode == "search":
                    drive.send((0, 0), "tag spotted")
                    frames_found += 1
                    status = "tag spotted - confirming"
                    if frames_found >= FOUND_FRAMES_BEFORE_GO:
                        searcher.reset()
                        for ctl in (turn_ctl, range_ctl):
                            ctl.reset()
                        mode = "calibrate" if (sign_finder and not sign_finder.done) else "run"
                        print("Tag confirmed - " + ("running the direction self-test."
                                                    if mode == "calibrate" else "approaching."))

                readout = f"x {pose.x_cm:+.1f}cm  z {pose.z_cm:.1f}cm  bearing {pose.bearing:+.1f}deg"

                if mode == "calibrate":
                    wheels, status = sign_finder.step(pose, now)
                    drive.send(wheels, status)
                    if sign_finder.done:
                        mode = "run"
                        for ctl in (turn_ctl, range_ctl):
                            ctl.reset()

                elif mode == "run":
                    # Don't creep any closer than the safety floor.
                    range_error = (pose.z_cm - DAP_DISTANCE_CM) if pose.z_cm > MIN_SAFE_DISTANCE_CM else 0.0

                    turn_ctl.observe(pose.bearing, now)
                    range_ctl.observe(range_error, now)

                    wheels = (0, 0)
                    # Finish whatever is already mid-pulse; otherwise aim, then approach.
                    if turn_ctl.busy:
                        action, status = turn_ctl.step(now)
                        wheels = turn_wheels(action) if action else (0, 0)
                    elif range_ctl.busy:
                        action, status = range_ctl.step(now)
                        wheels = drive_wheels(action) if action else (0, 0)
                    else:
                        action, status = turn_ctl.step(now)
                        if action is not None:
                            wheels = turn_wheels(action) if action else (0, 0)
                        else:
                            action, status = range_ctl.step(now)
                            if action is None:
                                status = "IN DAP RANGE"
                                mode = "dap"
                            else:
                                wheels = drive_wheels(action) if action else (0, 0)

                    drive.send(wheels, status)
                    check_wrong_way(turn_ctl, "turn")
                    check_wrong_way(range_ctl, "drive")

                elif mode == "done":
                    status = "DAPPED! Press 'q' to quit."

                searcher.aim_at(pose.bearing)  # if we lose it, hunt that way first

            if mode == "dap":
                drive.send((0, 0), "stopped - dapping")
                print("Chris detected")
                cv2.putText(frame, "Chris detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Dap Up", frame)
                cv2.waitKey(1)
                time.sleep(PRE_DAP_PAUSE_SECONDS)
                print("Dapping up!")
                dap_up_sequence(dm)
                mode = "done"
                status = "DAPPED! Press 'q' to quit."

            drive.check_watchdog()

            # Overlay: status, live pose readout, wheel speeds and signs.
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, readout, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"L {drive.last[0]}%  R {drive.last[1]}%   "
                               f"signs d{SIGNS['drive']:+d} t{SIGNS['turn']:+d}   mode {mode}",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.imshow("Dap Up", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always stop everything and release it, even on error.
        if dm is not None:
            dm.movement_stop()
            dm.disconnect()
        grabber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
