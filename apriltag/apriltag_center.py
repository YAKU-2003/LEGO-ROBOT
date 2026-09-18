"""Drive the LEGO Double Motor so an AprilTag on the robot ends up dead center
in a fixed webcam's view AND facing the camera straight on.

Two things are corrected, in this priority order:

  1. Heading: the tag's yaw is estimated from how much taller its near
     vertical edge looks than its far one. The robot spins in place until the
     tag squares up to the camera.
  2. Position: the robot drives forward / backward until the tag is centered
     horizontally in the frame.

Both use proportional speeds (fast when far off, slow when close) and
hysteresis (stop inside a tight zone, only restart outside a wider one), so
the robot eases in instead of overshooting or twitching.

If the tag is tilted so far that it drops out of detection, the robot keeps
turning back the way it last saw the tag tilt for a few seconds to find it
again. If the tag is lost any other way, it stops (never drive blind).

Mount the tag facing the camera, squared to the image (any multiple of 90
degrees of roll is fine).
"""

import math
import os
import statistics
import sys
import time
from collections import deque

import cv2
import legoeducation as le
import numpy as np
from cv2 import aruco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "poserace"))
from lelib import doubleMotor  # noqa: E402

TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = 0  # the tag ID on the robot; change to match your physical tag

# Double Motor pairing (same card the other scripts use).
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0994"

# ── Position (drive forward / backward) ──────────────────────────────────────
# Errors are normalized: 0 = tag exactly at center, +/-1 = tag at the frame edge.
STOP_ZONE = 0.015  # stop when the tag is within this of center (~5px on a 640px frame)
START_ZONE = 0.04  # once stopped, only restart when the tag is farther out than this

MIN_SPEED = 15  # percent; lowest speed that still gets the motors turning
MAX_SPEED = 40  # percent; speed used when the tag is far from center
SPEED_GAIN = 100  # percent of speed per unit of normalized error (before clamping)

# Which way the robot has to drive to move the tag toward the right of the
# image. Depends on how the robot faces the camera: if it drives the wrong way
# (runs away from center), flip this.
FORWARD_MOVES_TAG_RIGHT = True

# ── Heading (spin in place) ──────────────────────────────────────────────────
YAW_STOP_DEG = 3.0  # squared up when the tag's yaw is within this many degrees
YAW_START_DEG = 6.0  # once squared up, only start turning again past this

# The robot keeps coasting after a stop command and the camera / Bluetooth lag
# a few frames behind it, so spinning continuously overshoots badly. Instead it
# turns in short pulses: pulse, stop, wait for everything to settle, take the
# median yaw over several frames, and repeat.
TURN_SPEED = 15  # percent; wheel speed during a pulse
PULSE_CORRECTION = 0.5  # each pulse tries to remove this fraction of the measured yaw
MIN_PULSE_SECONDS = 0.05  # about two camera frames; the shortest nudge we can make
MAX_PULSE_SECONDS = 0.6
# Stopped time after each pulse. Must be longer than the total lag (Bluetooth +
# camera + coasting) or it measures while the robot is still moving.
SETTLE_SECONDS = 0.8
SAMPLE_FRAMES = 5  # yaw readings (median) used for each decision
# Starting guess for how fast the robot turns at TURN_SPEED. It re-measures this
# after every pulse, so it only matters for the first one or two.
INITIAL_TURN_RATE_DEG_PER_SEC = 30.0

# Turn direction needed to straighten a tag that has positive yaw (its right
# edge looks closer to the camera). Turning left (counter-clockwise from above)
# is correct for a normal, un-mirrored camera; if the robot spins the wrong way
# and the yaw number grows, flip this.
TURN_LEFT_FIXES_POSITIVE_YAW = True

# Rough horizontal field of view of the webcam. Only scales the yaw estimate
# (the zero point, "straight on", doesn't depend on it), so it needn't be exact.
CAMERA_HFOV_DEG = 65

# ── Lost-tag handling ────────────────────────────────────────────────────────
LOST_FRAMES_BEFORE_STOP = 3  # tolerate a few missed detections before reacting
RECOVER_MIN_YAW_DEG = 10  # only search by turning if the tag was last seen at least this tilted
RECOVER_TURN_SPEED = 25  # percent; search speed, kept separate so slowing the fine turning doesn't stall it
RECOVER_SECONDS = 6.0  # give up and stop after searching this long

SPEED_STEP = 5  # round speeds to this step so we don't send a new BLE command every frame

detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # sub-pixel tag position
# A little more tolerant of tilted (foreshortened) tags than the defaults.
detector_params.adaptiveThreshWinSizeMax = 53
detector_params.polygonalApproxAccuracyRate = 0.05
detector_params.errorCorrectionRate = 0.8
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(TAG_FAMILY), detector_params)


def tag_yaw_deg(tag_corners, frame_width):
    """Estimate how far the tag is turned away from facing the camera, in
    degrees. Positive = its right edge is closer to the camera than its left.

    A tag parallel to the image plane has equal left and right edge heights
    wherever it sits in the frame. Turning it makes the near edge look taller;
    with r = (right - left) / (right + left), sin(yaw) = 2 * f * r / height.

    "Left" and "right" are taken in the IMAGE, not the tag's own frame: the
    corners come back in the tag's own order (TL, TR, BR, BL), so a tag stuck
    on sideways or upside-down would otherwise have its top / bottom edges
    measured instead. Works for any mount at a multiple of 90 degrees."""
    c = tag_corners[0]
    # The two ways to split the square into a pair of opposite edges.
    pairs = (((0, 3), (1, 2)), ((0, 1), (3, 2)))
    # Use the pair that runs most up-and-down in the image.
    edge_a, edge_b = max(pairs, key=lambda p: sum(abs(c[i][1] - c[j][1]) for i, j in p))
    mean_x = lambda e: (c[e[0]][0] + c[e[1]][0]) / 2
    left_edge, right_edge = (edge_a, edge_b) if mean_x(edge_a) < mean_x(edge_b) else (edge_b, edge_a)

    left = np.linalg.norm(c[left_edge[0]] - c[left_edge[1]])
    right = np.linalg.norm(c[right_edge[0]] - c[right_edge[1]])
    focal_px = (frame_width / 2) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2)
    ratio = (right - left) / (right + left)
    sin_yaw = 2 * focal_px * ratio / ((left + right) / 2)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_yaw))))


def find_target(frame):
    """Return (center x-pixel, center y-pixel, yaw in degrees) for the target
    tag, or None. Also draws all detections on the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None

    aruco.drawDetectedMarkers(frame, corners, ids)
    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if tag_id == TARGET_TAG_ID:
            x, y = tag_corners[0].mean(axis=0)
            return float(x), float(y), tag_yaw_deg(tag_corners, frame.shape[1])
    return None


def quantize(magnitude):
    return int(round(magnitude / SPEED_STEP) * SPEED_STEP)


def spin_direction(yaw):
    """+1 to spin left, -1 to spin right, whichever reduces this yaw."""
    return 1 if (yaw > 0) == TURN_LEFT_FIXES_POSITIVE_YAW else -1


def spin_cmd(direction, speed):
    """Wheel speeds (left, right) for an in-place spin; + direction is left."""
    turn = direction * speed
    return (-turn, turn)


class HeadingController:
    """Squares the tag up to the camera using pulse / settle / measure cycles.

    update() is called once per frame the tag is visible. It returns
    (wheel command, status). A command of None means the tag is squared up and
    the caller is free to do something else (drive to center)."""

    def __init__(self):
        self.turn_rate = INITIAL_TURN_RATE_DEG_PER_SEC
        self.reset()

    def reset(self):
        """Forget everything measured so far (call when the tag is lost)."""
        self.state = "idle"  # idle -> pulse -> settle -> idle
        self.aligned = False
        self.history = deque(maxlen=SAMPLE_FRAMES)
        self.calibrating = False
        self.pulse_cmd = (0, 0)
        self.pulse_end = 0.0
        self.pulse_seconds = 0.0
        self.pulse_start_yaw = 0.0
        self.settle_until = 0.0

    def update(self, yaw, now):
        self.history.append(yaw)

        if self.state == "pulse":
            if now < self.pulse_end:
                return self.pulse_cmd, f"turning (yaw {yaw:+.1f} deg)"
            self.state = "settle"
            self.settle_until = now + SETTLE_SECONDS

        if self.state == "settle":
            if now < self.settle_until:
                self.history.clear()  # readings taken while it's still coasting are useless
            if now < self.settle_until or len(self.history) < SAMPLE_FRAMES:
                return (0, 0), "settling"
            self.state = "idle"

        if len(self.history) < SAMPLE_FRAMES:
            return (0, 0), "measuring"

        measured = statistics.median(self.history)
        if self.calibrating:
            self._learn_turn_rate(measured)
        size = abs(measured)
        self.aligned = size <= YAW_STOP_DEG or (self.aligned and size <= YAW_START_DEG)
        if self.aligned:
            return None, ""

        # Start a pulse that should remove PULSE_CORRECTION of the error.
        self.pulse_seconds = min(
            MAX_PULSE_SECONDS,
            max(MIN_PULSE_SECONDS, PULSE_CORRECTION * size / self.turn_rate),
        )
        self.pulse_cmd = spin_cmd(spin_direction(measured), TURN_SPEED)
        self.pulse_start_yaw = measured
        self.pulse_end = now + self.pulse_seconds
        self.calibrating = True
        self.state = "pulse"
        return self.pulse_cmd, f"turning (yaw {measured:+.1f} deg)"

    def _learn_turn_rate(self, yaw_after):
        """Compare where the last pulse started and ended to learn how many
        degrees per second it really turns (including coasting)."""
        self.calibrating = False
        start = self.pulse_start_yaw
        moved = (start - yaw_after) * math.copysign(1.0, start)  # + = moved toward / past zero
        print(f"pulse {self.pulse_seconds:.2f}s: yaw {start:+.1f} -> {yaw_after:+.1f}")
        if moved < -YAW_STOP_DEG:
            print("  yaw got WORSE after turning - try flipping TURN_LEFT_FIXES_POSITIVE_YAW")
        elif moved > 0:
            rate = max(5.0, min(300.0, moved / self.pulse_seconds))
            self.turn_rate = 0.5 * self.turn_rate + 0.5 * rate


def speed_for(error, centered):
    """Signed drive speed (+ forward, - backward) for a normalized position
    error, and whether the tag now counts as centered.

    error > 0 means the tag is right of center, so it has to move left."""
    size = abs(error)
    if size <= STOP_ZONE or (centered and size <= START_ZONE):
        return 0, True

    magnitude = quantize(min(MAX_SPEED, max(MIN_SPEED, size * SPEED_GAIN)))
    move_tag_right = error < 0
    forward = move_tag_right == FORWARD_MOVES_TAG_RIGHT
    return (magnitude if forward else -magnitude), False


def set_drive(dm, left, right):
    if left == 0 and right == 0:
        dm.movement_stop()
    else:
        dm.movement_move_tank(left, right, blocking=False)


print("Connecting to Double Motor...")
dm = doubleMotor()
dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
print("Connected!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    dm.disconnect()
    raise RuntimeError("Could not open webcam (index 0).")

print(f"Tracking tag ID {TARGET_TAG_ID}. Press 'q' to quit.")

last_cmd = (0, 0)  # (left, right) wheel speeds; only talk to the motors on change
heading = HeadingController()
centered = False
frames_lost = 0
lost_since = 0.0
last_yaw = None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        height, width = frame.shape[:2]
        half_width = width / 2
        target = find_target(frame)
        tag_x = tag_y = None
        yaw = None

        if target is None:
            if frames_lost == 0:
                lost_since = time.time()
            frames_lost += 1

            if frames_lost < LOST_FRAMES_BEFORE_STOP:
                cmd, status = last_cmd, "lost (waiting)"
            elif (
                last_yaw is not None
                and abs(last_yaw) >= RECOVER_MIN_YAW_DEG
                and time.time() - lost_since < RECOVER_SECONDS
            ):
                # Tilted so far the tag is unreadable (edge-on): keep turning
                # back the way it was last seen tilting until it reappears.
                cmd = spin_cmd(spin_direction(last_yaw), RECOVER_TURN_SPEED)
                status = "lost - searching"
            else:
                cmd, status = (0, 0), "lost"

            if frames_lost >= LOST_FRAMES_BEFORE_STOP:
                heading.reset()  # old yaw readings are stale once the tag has been gone a while
        else:
            frames_lost = 0
            tag_x, tag_y, yaw = target
            last_yaw = yaw
            error = (tag_x - half_width) / half_width

            cmd, status = heading.update(yaw, time.time())
            if cmd is None:  # squared up: now drive to center
                speed, centered = speed_for(error, centered)
                cmd = (speed, speed)
                status = "CENTERED" if centered else f"driving (err {error * half_width:+.0f}px)"

        if cmd != last_cmd:
            set_drive(dm, *cmd)
            last_cmd = cmd
            print(f"{status} -> L {cmd[0]}%  R {cmd[1]}%")

        # Overlay: exact center line, stop zone, and tag position.
        cx = int(half_width)
        stop_px = int(STOP_ZONE * half_width)
        cv2.line(frame, (cx, 0), (cx, height), (0, 0, 255), 1)
        cv2.rectangle(frame, (cx - stop_px, 0), (cx + stop_px, height), (255, 0, 0), 1)
        if tag_x is not None:
            cv2.circle(frame, (int(tag_x), int(tag_y)), 6, (0, 255, 0), -1)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if yaw is not None:
            readout = f"x err {tag_x - half_width:+.0f}px   yaw {yaw:+.1f} deg"
        else:
            readout = "x err --   yaw --"
        cv2.putText(frame, f"{readout}   L {last_cmd[0]}%  R {last_cmd[1]}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("AprilTag Centering", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Always stop the robot and release the BLE link, even on error.
    dm.movement_stop()
    dm.disconnect()
    cap.release()
    cv2.destroyAllWindows()
