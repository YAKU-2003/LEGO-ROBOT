"""Drive the LEGO Double Motor toward an AprilTag using an iPhone camera
mounted ON the robot.

The phone streams video to this computer. When it sees the target tag, the
robot:

  1. turns in place until the tag is centered in the phone's view, then
  2. drives forward / backward until the tag is TARGET_DISTANCE_CM away.

Both steps use short pulses (move, stop, wait for the lagging stream to catch
up, measure, repeat) instead of continuous motion. A phone stream lags well
behind the real robot, so continuous control would overshoot and hunt.

If the tag is lost, or the stream stalls, the robot stops.

Getting the iPhone video onto this computer -- set CAMERA_SOURCE below:
  * A "phone as webcam" app (Camo, EpocCam, iVCam, DroidCam) makes it show up
    as a normal webcam: use its index (0, 1, 2 ... whichever it is).
  * An IP-camera app that serves the video over Wi-Fi: use its URL as a string,
    e.g. "http://192.168.1.23:8080/video". Phone and computer must be on the
    same network.
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
from lelib import doubleMotor  # noqa: E402

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 1  # webcam index, or a stream URL string (see the top of this file). 1 = DroidCam here; 0 is the laptop webcam
MIRRORED = False  # set True if the app mirrors the picture (tags then can't be read)
CAMERA_HFOV_DEG = 65  # rough horizontal field of view of the phone's rear camera
STREAM_STALL_SECONDS = 1.0  # warn if no new frame arrives for this long

# ── Tag ──────────────────────────────────────────────────────────────────────
TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = 0  # the tag ID to go to; change to match your physical tag
TAG_SIZE_CM = 10.0  # side of the tag's BLACK square, measured edge to edge

# ── Double Motor pairing (same card the other scripts use) ───────────────────
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0994"

# ── Where to stop ────────────────────────────────────────────────────────────
TARGET_DISTANCE_CM = 20.0  # how far from the tag to stop, at the closest

# The whole tag, all four edges, must stay inside the picture. The robot can
# only move forward / backward (not up / down), so if the tag sits high or low
# in the view it may not fit at TARGET_DISTANCE_CM; the robot then stops further
# back, at the closest distance where the whole tag still fits.
FIT_MARGIN_PX = 20  # keep the tag at least this far inside the edges of the picture
MAX_FIT_DISTANCE_CM = 150.0  # never back off further than this to make it fit
# A row / column counts as picture (not a black bar) if anything in it is
# brighter than this. Phone apps often letterbox with black bars.
BAR_BRIGHTNESS = 24

# True if the phone looks the way the robot drives (forward). If it looks
# backward, set False: both the turning and the driving flip.
CAMERA_FACES_FORWARD = True

# Once close and roughly centered, the robot also spins to square the tag's
# face up to the camera (its left and right edges become equal length), so it
# doesn't just arrive close to the tag -- it arrives facing it head-on.
#
# NOTE: a robot that can only spin in place or drive straight cannot reach a
# perfectly perpendicular approach from every starting position without
# circling the tag first -- it can only square up the last stretch. Start it
# roughly in front of the tag for the best result.
SQUARE_UP = True
TURN_LEFT_FIXES_POSITIVE_YAW = True  # flip if the console says yaw "got WORSE" while turning

# ── Pulse control (shared by turning and driving) ────────────────────────────
PULSE_CORRECTION = 0.5  # each pulse tries to remove this fraction of the error
MIN_PULSE_SECONDS = 0.05  # about a camera frame or two; the shortest nudge
MAX_PULSE_SECONDS = 0.8
# Stopped time after each pulse. Must be longer than the total lag (Bluetooth +
# stream + coasting), or it measures while the robot is still moving. Phone
# streams often lag 0.3 - 1 s: raise this if it keeps overshooting.
SETTLE_SECONDS = 1.0
SAMPLE_FRAMES = 5  # readings (median) used for each decision

# Turning (tag left / right of center). Error is the tag's bearing in degrees.
BEARING_STOP_DEG = 2.0  # centered when within this
BEARING_START_DEG = 5.0  # once centered, only turn again past this
TURN_SPEED = 15  # percent
INITIAL_TURN_RATE = 30.0  # deg/s guess at TURN_SPEED; re-measured after each pulse

# Driving (tag too far / too close). Error is distance minus TARGET_DISTANCE_CM.
RANGE_STOP_CM = 5.0
RANGE_START_CM = 10.0
DRIVE_SPEED = 30  # percent
INITIAL_DRIVE_RATE = 10.0  # cm/s guess at DRIVE_SPEED; re-measured after each pulse

# Squaring up (tag's face angled away from the camera). Error is the tag's
# yaw in degrees, measured the same way as bearing / range: pulse, settle, remeasure.
YAW_STOP_DEG = 3.0
YAW_START_DEG = 6.0
# Only try to square up once this close; too far away, the yaw reading is too
# noisy (a couple of pixels of edge error is many degrees of yaw) to trust.
SQUARE_UP_RANGE_CM = TARGET_DISTANCE_CM * 2.0

LOST_FRAMES_BEFORE_STOP = 3  # tolerate a few missed detections before stopping
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
    """Estimate how far the tag is turned away from facing the camera, in
    degrees. Positive = its right edge is closer to the camera than its left.

    "Left" and "right" are taken in the IMAGE, not the tag's own frame: the
    corners come back in the tag's own order (TL, TR, BR, BL), so a tag stuck
    on sideways or upside-down would otherwise have its top / bottom edges
    measured instead. Works for any mount at a multiple of 90 degrees.

    A tag parallel to the image plane has equal left and right edge heights;
    turning it makes the near edge look taller. With r = (right-left)/(right+left),
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
        brightest = frame.max(axis=2)  # brightest channel of each pixel
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
        """The visible box, or the whole frame while still measuring."""
        if self.box is not None:
            return self.box
        return (0, 0, frame.shape[1] - 1, frame.shape[0] - 1)


def min_fit_distance_cm(corners, dist, area):
    """Closest distance at which the whole tag would still sit inside `area`
    (left, top, right, bottom), keeping FIT_MARGIN_PX from each edge.

    Driving straight toward the tag scales everything in the picture away from
    the image center, so each edge of the tag can grow by a limited factor
    before it hits the border."""
    left, top, right, bottom = area
    left, top = left + FIT_MARGIN_PX, top + FIT_MARGIN_PX
    right, bottom = right - FIT_MARGIN_PX, bottom - FIT_MARGIN_PX
    cx, cy = (left + right) / 2, (top + bottom) / 2

    xs, ys = corners[:, 0], corners[:, 1]
    growth = math.inf  # how many times bigger the tag can get and still fit
    for extreme, limit, center in (
        (xs.min(), left, cx), (xs.max(), right, cx),
        (ys.min(), top, cy), (ys.max(), bottom, cy),
    ):
        offset, room = extreme - center, limit - center
        if offset * room > 0:  # this extreme is on the same side of center as its border
            growth = min(growth, room / offset)
    return 0.0 if math.isinf(growth) else dist / growth


def find_target(frame):
    """Return (x, y, distance_cm, bearing_deg, yaw_deg, corners) for the
    target tag, or None. corners is a 4x2 array of the tag's corner pixels.
    Also draws all detections on the frame."""
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
        # Average the two longest edges: turning the tag foreshortens some
        # edges, but never all of them, so the longest ones stay true.
        edges = sorted(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
        side_px = (edges[2] + edges[3]) / 2
        yaw = tag_yaw_deg(c, width)
        return float(x), float(y), distance_cm(side_px, width), bearing_deg(x, width), yaw, c
    return None


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
        """(newest frame or None, number of frames received so far)."""
        with self._lock:
            return self._frame, self._count

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class PulseController:
    """Drives one error toward zero with pulse / settle / measure cycles.

    update() is called once per new frame with the current error. It returns
    (action, status):
      None -> within the stop zone; nothing to do
      0    -> hold still (settling or measuring)
      +1/-1 -> pulse now, correcting an error of that sign
    It learns how fast a pulse really moves things and sizes the next pulse to
    match."""

    def __init__(self, label, unit, stop_zone, start_zone, initial_rate, flip_hint="CAMERA_FACES_FORWARD"):
        self.label = label
        self.unit = unit
        self.stop_zone = stop_zone
        self.start_zone = start_zone
        self.rate = initial_rate  # error units per second of pulse
        self.flip_hint = flip_hint  # setting name to suggest flipping if it moves the wrong way
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
            MAX_PULSE_SECONDS,
            max(MIN_PULSE_SECONDS, PULSE_CORRECTION * size / self.rate),
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


def wheels_for(axis, action):
    """Wheel speeds (left, right) for a pulse on `axis` correcting an error of sign `action`."""
    facing = 1 if CAMERA_FACES_FORWARD else -1
    if axis == "bearing":
        # Tag to the right (+) -> spin right. Spinning left is left wheel back, right wheel forward.
        turn = -action * facing * TURN_SPEED
        return (-turn, turn)
    if axis == "yaw":
        # Independent flip flag: squaring up is a different physical motion
        # (aiming at the tag's own tilt, not its screen position), so its
        # correct turn direction isn't guaranteed to match "bearing"'s.
        yaw_sign = 1 if TURN_LEFT_FIXES_POSITIVE_YAW else -1
        turn = -action * facing * yaw_sign * TURN_SPEED
        return (-turn, turn)
    # Too far (+) -> drive toward the tag.
    speed = action * facing * DRIVE_SPEED
    return (speed, speed)


def main():
    grabber = FrameGrabber(CAMERA_SOURCE)
    dm = None
    try:
        print("Connecting to Double Motor...")
        dm = doubleMotor()
        dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        print("Connected!")
        print(f"Looking for tag ID {TARGET_TAG_ID}; will hold {TARGET_DISTANCE_CM:.0f} cm away. Press 'q' to quit.")

        drive = Drive(dm)
        bearing_ctl = PulseController("turn", " deg", BEARING_STOP_DEG, BEARING_START_DEG, INITIAL_TURN_RATE)
        yaw_ctl = PulseController("square", " deg", YAW_STOP_DEG, YAW_START_DEG, INITIAL_TURN_RATE,
                                   flip_hint="TURN_LEFT_FIXES_POSITIVE_YAW")
        range_ctl = PulseController("drive", " cm", RANGE_STOP_CM, RANGE_START_CM, INITIAL_DRIVE_RATE)

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
                # No new picture yet: keep the watchdog and window alive.
                if now - last_frame_time > STREAM_STALL_SECONDS and not stalled:
                    stalled = True
                    print("Video stream stalled - motors will stop.")
                drive.check_watchdog()
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
                    bearing_ctl.reset()
                    yaw_ctl.reset()
                    range_ctl.reset()
                    status = "lost"
            else:
                frames_lost = 0
                x, y, dist, bearing, yaw, corners = target
                tag_xy = (int(x), int(y))
                # Stop at the target distance, or further back if the whole tag wouldn't fit there.
                fit_dist = min_fit_distance_cm(corners, dist, area)
                goal = max(TARGET_DISTANCE_CM, min(fit_dist, MAX_FIT_DISTANCE_CM))
                readout = f"bearing {bearing:+.1f}  yaw {yaw:+.1f}  dist {dist:.0f}cm  goal {goal:.0f}cm"
                range_error = dist - goal
                may_square = SQUARE_UP and dist <= SQUARE_UP_RANGE_CM

                # Aim (bearing), then square up (yaw, once close), then drive
                # (range) -- but finish whichever pulse is already in flight
                # first, and reset whatever comes later so it starts from a
                # fresh measurement once it's actually its turn.
                order = [("bearing", bearing_ctl, bearing, True),
                         ("yaw", yaw_ctl, yaw, may_square),
                         ("range", range_ctl, range_error, True)]
                busy = next((o for o in order if o[1].busy), None)
                if busy is not None:
                    axis, ctl, err, _ = busy
                    for _, other, _, _ in order:
                        if other is not ctl:
                            other.reset()
                    action, status = ctl.update(err, now)
                else:
                    axis, action, status = "range", None, ""
                    for i, (name, ctl, err, applicable) in enumerate(order):
                        if not applicable:
                            ctl.reset()
                            continue
                        action, status = ctl.update(err, now)
                        if action is None:
                            continue  # this axis is satisfied; check the next one
                        for _, later, _, _ in order[i + 1:]:
                            later.reset()
                        axis = name
                        break

                if action is None:
                    drive.send((0, 0), "ON TARGET")
                    status = "ON TARGET"
                elif action == 0:
                    drive.send((0, 0), status)
                else:
                    drive.send(wheels_for(axis, action), status)

            drive.check_watchdog()

            # Overlay: crosshair at the image center, tag position, readouts.
            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 0, 255), 1)
            m = FIT_MARGIN_PX  # yellow box = where the whole tag has to stay
            cv2.rectangle(frame, (area[0] + m, area[1] + m), (area[2] - m, area[3] - m), (0, 255, 255), 1)
            if tag_xy is not None:
                cv2.circle(frame, tag_xy, 6, (0, 255, 0), -1)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"{readout}   L {drive.last[0]}%  R {drive.last[1]}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("AprilTag Follow (iPhone)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always stop the robot and release everything, even on error.
        if dm is not None:
            dm.movement_stop()
            dm.disconnect()
        grabber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
