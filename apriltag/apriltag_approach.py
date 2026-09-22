"""Drive the LEGO Double Motor up to within ARRIVE_DISTANCE_CM of an AprilTag,
using an iPhone camera mounted ON the robot.

The tag itself sits on a separate, stationary mount driven by a Single Motor,
which spins the tag on the spot to keep it facing the camera -- so the tag
stays square and easy to read no matter which way the robot is currently
pointed. That means this script never has to fight the "can't be both
centred and square" problem: squaring is the tag's own motor's job, entirely
independent of the robot. The robot's job is simply:

  1. turn in place until the tag is centred in the camera's view, then
  2. drive forward / backward until it's ARRIVE_DISTANCE_CM away,

using the exact same short-pulse, settle, re-measure control as the rest of
this project (a phone stream lags well behind the real robot, so continuous
control overshoots and hunts).

Requires TWO separate LEGO Education devices, connected over Bluetooth:
  - a Double Motor on the robot (drives it)
  - a Single Motor on the stationary tag mount (spins the tag to face the
    camera; the AprilTag must be mounted CENTRED on this motor's shaft, like
    a turntable -- if it's off to one side, spinning it will also swing its
    position sideways in the frame and confuse the robot's centring)

Getting the iPhone video onto this computer -- set CAMERA_SOURCE below:
  * A "phone as webcam" app (Camo, EpocCam, iVCam, DroidCam) makes it show up
    as a normal webcam: use its index (0, 1, 2 ... whichever it is).
  * An IP-camera app that serves the video over Wi-Fi: use its URL as a string.
Use the phone's REAR camera, pointing the way the robot drives.
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
from lelib import doubleMotor, singleMotor  # noqa: E402

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 1  # webcam index, or a stream URL string. 1 = DroidCam here; 0 is the laptop webcam
MIRRORED = False  # set True if the app mirrors the picture (tags then can't be read)
CAMERA_HFOV_DEG = 65  # rough horizontal field of view of the phone's rear camera
STREAM_STALL_SECONDS = 1.0  # stop the robot if no new frame arrives for this long

# ── Tag ──────────────────────────────────────────────────────────────────────
TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = 0  # the tag ID to approach; change to match your physical tag
TAG_SIZE_CM = 10.0  # side of the tag's BLACK square, measured edge to edge

# ── Robot's Double Motor ─────────────────────────────────────────────────────
ROBOT_CARD_COLOR = le.LEGO_COLOR_GREEN
ROBOT_CARD_SERIAL = "0994"

# ── Tag mount's Single Motor ─────────────────────────────────────────────────
# ASSUMPTION: reusing the same card as the robot's other devices, since that's
# the only Single Motor pairing on record for this project. If the tag mount
# has its own separate Connection Card, change these.
MOUNT_CARD_COLOR = le.LEGO_COLOR_GREEN
MOUNT_CARD_SERIAL = "0994"

# ── Where to stop ────────────────────────────────────────────────────────────
ARRIVE_DISTANCE_CM = 20.0
# The whole tag must stay inside the picture (inset by FIT_MARGIN_PX). Since
# the tag keeps itself square, this only matters if it's mounted high/low
# relative to the camera's height -- the robot then stops further back.
FIT_MARGIN_PX = 20
MAX_FIT_DISTANCE_CM = 150.0
BAR_BRIGHTNESS = 24  # a row/col counts as picture (not a black bar) above this brightness

# True if the phone looks the way the robot drives (forward). If it looks
# backward, set False: both the turning and the driving flip.
CAMERA_FACES_FORWARD = True
# Flip if the console says the tag's own squaring "got WORSE after moving".
MOUNT_TURN_SIGN = 1

# ── Pulse control (shared by every axis: robot turn, robot drive, mount spin) ─
PULSE_CORRECTION = 0.5  # each pulse tries to remove this fraction of the error
MIN_PULSE_SECONDS = 0.05
MAX_PULSE_SECONDS = 0.8
# Stopped time after each pulse, long enough for stream + Bluetooth lag and
# coasting to fully settle before trusting a reading. Raise this if either
# motor keeps overshooting.
SETTLE_SECONDS = 1.0
SAMPLE_FRAMES = 5  # readings (median) used for each decision

# Robot turning (tag left / right of centre). Error = bearing, in degrees.
BEARING_STOP_DEG = 2.0
BEARING_START_DEG = 5.0
TURN_SPEED = 15  # percent
INITIAL_TURN_RATE = 30.0  # deg/s guess; re-measured after each pulse

# Robot driving (too far / too close). Error = distance minus goal, in cm.
RANGE_STOP_CM = 3.0
RANGE_START_CM = 6.0
DRIVE_SPEED = 30  # percent
INITIAL_DRIVE_RATE = 10.0  # cm/s guess; re-measured after each pulse

# Tag-mount squaring (how tilted the tag looks). Error = tag's own yaw, in degrees.
YAW_STOP_DEG = 3.0
YAW_START_DEG = 6.0
MOUNT_SPEED = 20  # percent
INITIAL_MOUNT_RATE = 40.0  # deg/s guess; re-measured after each pulse

LOST_FRAMES_BEFORE_STOP = 3  # tolerate a few missed detections before stopping everything
MOTOR_WATCHDOG_SECONDS = 1.0  # any motion command older than this is cancelled

detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector_params.adaptiveThreshWinSizeMax = 53
detector_params.polygonalApproxAccuracyRate = 0.05
detector_params.errorCorrectionRate = 0.8
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(TAG_FAMILY), detector_params)


def focal_px(frame_width):
    return (frame_width / 2) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2)


def bearing_deg(tag_x, frame_width):
    """Angle of the tag off the camera's axis. Positive = tag is to the right."""
    return math.degrees(math.atan((tag_x - frame_width / 2) / focal_px(frame_width)))


def distance_cm(side_px, frame_width):
    return focal_px(frame_width) * TAG_SIZE_CM / side_px


def tag_yaw_deg(tag_corners, frame_width):
    """How far the tag is turned away from facing the camera, in degrees.
    Positive = its right edge is closer to the camera than its left.

    "Left"/"right" are taken in the IMAGE, not the tag's own frame, so this
    works for a tag mounted at any multiple of 90 degrees. A tag parallel to
    the image plane has equal left/right edge heights; turning it makes the
    near edge look taller: with r = (right-left)/(right+left),
    sin(yaw) = 2 * f * r / height."""
    c = tag_corners
    pairs = (((0, 3), (1, 2)), ((0, 1), (3, 2)))
    edge_a, edge_b = max(pairs, key=lambda p: sum(abs(c[i][1] - c[j][1]) for i, j in p))
    mean_x = lambda e: (c[e[0]][0] + c[e[1]][0]) / 2
    left_edge, right_edge = (edge_a, edge_b) if mean_x(edge_a) < mean_x(edge_b) else (edge_b, edge_a)

    left = np.linalg.norm(c[left_edge[0]] - c[left_edge[1]])
    right = np.linalg.norm(c[right_edge[0]] - c[right_edge[1]])
    ratio = (right - left) / (right + left)
    sin_yaw = 2 * focal_px(frame_width) * ratio / ((left + right) / 2)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_yaw))))


def find_target(frame):
    """Return (x, y, distance_cm, bearing_deg, yaw_deg, corners) for the
    target tag, or None. Also draws all detections on the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None

    aruco.drawDetectedMarkers(frame, corners, ids)
    width = frame.shape[1]
    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if tag_id != TARGET_TAG_ID:
            continue
        c = tag_corners[0]
        x, y = c.mean(axis=0)
        edges = sorted(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
        side_px = (edges[2] + edges[3]) / 2
        yaw = tag_yaw_deg(c, width)
        return float(x), float(y), distance_cm(side_px, width), bearing_deg(x, width), yaw, c
    return None


class VisibleArea:
    """Finds the real picture inside the frame, ignoring black letterbox /
    pillarbox bars. Watches the first few frames (the bars stay put) and then
    locks the result in."""

    WATCH_FRAMES = 15

    def __init__(self):
        self.seen = 0
        self.row_max = None
        self.col_max = None
        self.box = None

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
        if len(rows) < height // 2 or len(cols) < width // 2:
            return (0, 0, width - 1, height - 1)
        return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))

    def get(self, frame):
        if self.box is not None:
            return self.box
        return (0, 0, frame.shape[1] - 1, frame.shape[0] - 1)


def min_fit_distance_cm(corners, dist, area):
    """Closest distance at which the whole tag would still sit inside `area`
    (left, top, right, bottom), keeping FIT_MARGIN_PX from each edge."""
    left, top, right, bottom = area
    left, top = left + FIT_MARGIN_PX, top + FIT_MARGIN_PX
    right, bottom = right - FIT_MARGIN_PX, bottom - FIT_MARGIN_PX
    cx, cy = (left + right) / 2, (top + bottom) / 2

    xs, ys = corners[:, 0], corners[:, 1]
    growth = math.inf
    for extreme, limit, center in (
        (xs.min(), left, cx), (xs.max(), right, cx),
        (ys.min(), top, cy), (ys.max(), bottom, cy),
    ):
        offset, room = extreme - center, limit - center
        if offset * room > 0:
            growth = min(growth, room / offset)
    return 0.0 if math.isinf(growth) else dist / growth


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

    update() is called once per new frame. It returns (action, status):
      None  -> inside the stop zone, nothing to do
      0     -> hold still (settling or measuring)
      +1/-1 -> pulse now, correcting an error of that sign
    It learns how fast a pulse really moves things and sizes the next one to
    match."""

    def __init__(self, label, unit, stop_zone, start_zone, initial_rate, flip_hint):
        self.label = label
        self.unit = unit
        self.stop_zone = stop_zone
        self.start_zone = start_zone
        self.rate = initial_rate
        self.flip_hint = flip_hint
        self.reset()

    def reset(self):
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

    def update(self, error, now):
        self.history.append(error)

        if self.state == "pulse":
            if now < self.pulse_end:
                return self.pulse_sign, f"{self.label} pulse ({error:+.1f}{self.unit})"
            self.state = "settle"
            self.settle_until = now + SETTLE_SECONDS

        if self.state == "settle":
            if now < self.settle_until:
                self.history.clear()  # readings taken while it's still moving are useless
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
        moved = (start - error_after) * math.copysign(1.0, start)
        print(f"{self.label} pulse {self.pulse_seconds:.2f}s: {start:+.1f} -> {error_after:+.1f}{self.unit}")
        if moved < -self.stop_zone:
            print(f"  {self.label} got WORSE after moving - try flipping {self.flip_hint}")
        elif moved > 0:
            self.rate = 0.5 * self.rate + 0.5 * max(self.rate / 20, min(self.rate * 20, moved / self.pulse_seconds))


class Drive:
    """Sends wheel commands only when they change, and cancels any motion
    command that has gone stale (stream stalled, program hung, ...)."""

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


class MountDrive:
    """Same idea as Drive, but for the tag mount's single motor: one signed
    speed instead of a (left, right) pair."""

    def __init__(self, sm):
        self.sm = sm
        self.last = 0
        self.sent_at = 0.0

    def send(self, speed, status=""):
        speed = int(speed)
        if speed == self.last:
            return
        if speed == 0:
            self.sm.motor_stop()
        else:
            self.sm.motor_run(speed=speed, blocking=False)
        self.last = speed
        self.sent_at = time.time()
        print(f"{status} -> mount {speed}%")

    def check_watchdog(self):
        if self.last != 0 and time.time() - self.sent_at > MOTOR_WATCHDOG_SECONDS:
            self.send(0, "watchdog")

    def stop(self):
        self.sm.motor_stop()
        self.last = 0


def wheels_for(axis, action):
    """Wheel speeds (left, right) for a pulse on `axis` correcting an error of sign `action`."""
    facing = 1 if CAMERA_FACES_FORWARD else -1
    if axis == "bearing":
        # Tag to the right (+) -> spin right. Spinning left is left wheel back, right wheel forward.
        turn = -action * facing * TURN_SPEED
        return (-turn, turn)
    # Too far (+) -> drive toward the tag.
    speed = action * facing * DRIVE_SPEED
    return (speed, speed)


def mount_speed_for(action):
    """Signed speed for the tag mount's single motor correcting a yaw of sign `action`."""
    return action * MOUNT_TURN_SIGN * MOUNT_SPEED


CONNECT_TIMEOUT_SECONDS = 15  # give up and say so, rather than hang silently forever


def connect_with_timeout(label, device, card_serial, card_color):
    """device.connect(...) can hang forever with NO error if it never finds a
    matching device (the underlying library scans with no time limit) -- most
    often because that device still thinks it's connected from an earlier,
    uncleanly-ended session and has stopped advertising. Run the real connect
    call on a background thread and give up loudly after a bounded time
    instead of sitting there in silence."""
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
            "disconnect cleanly and has stopped advertising. Power-cycle this device (off, then on) "
            "and try again. The connect attempt is still running in the background and can't be "
            "cancelled -- restart this script after power-cycling."
        )
    if result["error"] is not None:
        raise result["error"]
    if not device.connected:
        raise ConnectionError(f"{label}: connect() returned without connecting (card {card_serial}/{card_color}).")


def main():
    grabber = FrameGrabber(CAMERA_SOURCE)
    dm = None
    sm = None
    try:
        print("Connecting to robot's Double Motor...")
        dm = doubleMotor()
        connect_with_timeout("Double Motor", dm, ROBOT_CARD_SERIAL, ROBOT_CARD_COLOR)
        print("Connected!")

        print("Connecting to tag mount's Single Motor...")
        sm = singleMotor()
        connect_with_timeout("Single Motor", sm, MOUNT_CARD_SERIAL, MOUNT_CARD_COLOR)
        print("Connected!")

        print(f"Looking for tag ID {TARGET_TAG_ID}; robot will stop {ARRIVE_DISTANCE_CM:.0f} cm away. Press 'q' to quit.")

        drive = Drive(dm)
        mount = MountDrive(sm)
        bearing_ctl = PulseController("turn", " deg", BEARING_STOP_DEG, BEARING_START_DEG,
                                       INITIAL_TURN_RATE, flip_hint="CAMERA_FACES_FORWARD")
        range_ctl = PulseController("drive", " cm", RANGE_STOP_CM, RANGE_START_CM,
                                     INITIAL_DRIVE_RATE, flip_hint="CAMERA_FACES_FORWARD")
        yaw_ctl = PulseController("mount square", " deg", YAW_STOP_DEG, YAW_START_DEG,
                                   INITIAL_MOUNT_RATE, flip_hint="MOUNT_TURN_SIGN")

        visible = VisibleArea()
        last_count = 0
        last_frame_time = time.time()
        stalled = False
        frames_lost = 0
        status = "waiting for video"
        readout = ""

        while True:
            frame, count = grabber.latest()
            now = time.time()

            if frame is None or count == last_count:
                if now - last_frame_time > STREAM_STALL_SECONDS and not stalled:
                    stalled = True
                    print("Video stream stalled - motors will stop.")
                drive.check_watchdog()
                mount.check_watchdog()
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            last_count = count
            last_frame_time = now
            stalled = False
            if MIRRORED:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            visible.update(frame)
            area = visible.get(frame)

            target = find_target(frame)
            tag_xy = None

            if target is None:
                frames_lost += 1
                readout = "tag --"
                if frames_lost >= LOST_FRAMES_BEFORE_STOP:
                    drive.send((0, 0), "lost")
                    mount.send(0, "lost")
                    bearing_ctl.reset()
                    range_ctl.reset()
                    yaw_ctl.reset()
                    status = "lost"
            else:
                frames_lost = 0
                x, y, dist, bearing, yaw, corners = target
                tag_xy = (int(x), int(y))
                fit_dist = min_fit_distance_cm(corners, dist, area)
                goal = max(ARRIVE_DISTANCE_CM, min(fit_dist, MAX_FIT_DISTANCE_CM))
                range_error = dist - goal
                readout = f"bearing {bearing:+.1f}  yaw {yaw:+.1f}  dist {dist:.0f}/{goal:.0f}cm"

                # The mount's squaring is fully independent of the robot's
                # driving -- neither loop resets the other.
                mount_action, mount_status = yaw_ctl.update(yaw, now)
                if mount_action is None:
                    mount.send(0, "mount square: SQUARE")
                elif mount_action == 0:
                    mount.send(0, mount_status)
                else:
                    mount.send(mount_speed_for(mount_action), mount_status)

                # Robot: finish any drive pulse first; otherwise centre, then close the distance.
                if range_ctl.busy:
                    bearing_ctl.reset()
                    axis = "range"
                    action, status = range_ctl.update(range_error, now)
                else:
                    axis = "bearing"
                    action, status = bearing_ctl.update(bearing, now)
                    if action is None:
                        axis = "range"
                        action, status = range_ctl.update(range_error, now)
                    else:
                        range_ctl.reset()

                if action is None:
                    drive.send((0, 0), "ARRIVED")
                    status = "ARRIVED"
                elif action == 0:
                    drive.send((0, 0), status)
                else:
                    drive.send(wheels_for(axis, action), status)

            drive.check_watchdog()
            mount.check_watchdog()

            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 0, 255), 1)
            m = FIT_MARGIN_PX
            cv2.rectangle(frame, (area[0] + m, area[1] + m), (area[2] - m, area[3] - m), (0, 255, 255), 1)
            if tag_xy is not None:
                cv2.circle(frame, tag_xy, 6, (0, 255, 0), -1)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"{readout}   L {drive.last[0]}%  R {drive.last[1]}%  mount {mount.last}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("AprilTag Approach (self-squaring tag)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if dm is not None:
            dm.movement_stop()
            dm.disconnect()
        if sm is not None:
            sm.motor_stop()
            sm.disconnect()
        grabber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
