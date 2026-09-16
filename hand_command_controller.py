"""Convert MediaPipe hand landmarks into discrete robot commands.

Unlike hand_gesture_controller.py (which maps wrist height to a smoothed,
continuous wheel speed), this module recognizes a fixed set of hand
*shapes* -- fist, open palm, one finger, two fingers -- and which hand(s)
are raised, and turns that combination into one of a fixed set of
commands each frame. There's no smoothing: these are on/off gestures, not
an analog control input.

Gesture scheme:
    both hands raised, open palms  -> drive forward
    both hands raised, fists       -> drive backward
    only the left hand raised      -> turn left (pivot on the right wheel)
    only the right hand raised     -> turn right (pivot on the left wheel)
    two fingers up (either hand)   -> spin the auxiliary single motor one way
    one finger up (either hand)    -> spin the auxiliary single motor the
                                       other way
    anything else / no hands       -> stop
"""

import math
from dataclasses import dataclass
from enum import Enum, auto

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

# A wrist above this fraction of the frame height (0 = top, 1 = bottom)
# counts as "raised". There's no body/shoulder reference available from
# hand landmarks alone, so this is a fixed frame position rather than
# something scaled to the user's build.
RAISED_Y_THRESHOLD = 0.5

# A fingertip must be at least this many times farther from the wrist than
# its pip joint to count as "extended". Thumb is excluded from finger
# counting: its pip/mcp geometry points sideways rather than toward the
# fingertip the way the other four fingers' joints do, so the same
# tip-farther-than-pip test doesn't reliably tell curled from extended.
FINGER_EXTENDED_MARGIN = 1.1

# Minimum handedness classification confidence before a detected hand is
# trusted, so a low-confidence guess doesn't drive the wrong wheel.
HANDEDNESS_CONFIDENCE_THRESHOLD = 0.5

DRIVE_SPEED = 60  # wheel speed percent for forward/backward
TURN_SPEED = 60  # wheel speed percent for a pivot turn
SINGLE_MOTOR_SPEED = 70  # speed percent for the auxiliary single motor

_FINGER_JOINTS = (
    (HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP),
)


class HandShape(Enum):
    FIST = auto()
    OPEN_PALM = auto()
    ONE_FINGER = auto()
    TWO_FINGERS = auto()
    OTHER = auto()


@dataclass
class RobotCommand:
    name: str = "STOP"
    left_speed: float = 0.0
    right_speed: float = 0.0
    single_motor_speed: float = 0.0


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _count_extended_fingers(landmarks) -> int:
    wrist = landmarks[HandLandmark.WRIST]
    count = 0
    for tip_idx, pip_idx in _FINGER_JOINTS:
        tip_dist = _distance(landmarks[tip_idx], wrist)
        pip_dist = _distance(landmarks[pip_idx], wrist)
        if tip_dist > pip_dist * FINGER_EXTENDED_MARGIN:
            count += 1
    return count


def _classify_shape(landmarks) -> HandShape:
    extended = _count_extended_fingers(landmarks)
    if extended == 0:
        return HandShape.FIST
    if extended == 1:
        return HandShape.ONE_FINGER
    if extended == 2:
        return HandShape.TWO_FINGERS
    return HandShape.OPEN_PALM


def _is_raised(landmarks) -> bool:
    return landmarks[HandLandmark.WRIST].y < RAISED_Y_THRESHOLD


class HandCommandController:
    """Classifies one frame's hand landmarks into a single RobotCommand."""

    def update(self, hands_result, mirrored: bool) -> RobotCommand:
        """Compute this frame's command from a HandLandmarkerResult.

        Parameters:
            hands_result: A HandLandmarkerResult with `hand_landmarks` and
                `handedness`, aligned by index. Zero, one, or two hands may
                be present.
            mirrored: Whether the source frame was flipped horizontally.
                MediaPipe's Left/Right labels describe the image as
                captured, so a mirrored frame needs the labels swapped for
                a raised right hand to mean the person's actual right hand.
        """
        left_label, right_label = ("Right", "Left") if mirrored else ("Left", "Right")

        left_landmarks = None
        right_landmarks = None
        for landmarks, classifications in zip(hands_result.hand_landmarks, hands_result.handedness):
            best = classifications[0]
            if best.score < HANDEDNESS_CONFIDENCE_THRESHOLD:
                continue
            if best.category_name == left_label and left_landmarks is None:
                left_landmarks = landmarks
            elif best.category_name == right_label and right_landmarks is None:
                right_landmarks = landmarks

        left_shape = _classify_shape(left_landmarks) if left_landmarks is not None else None
        right_shape = _classify_shape(right_landmarks) if right_landmarks is not None else None

        # Finger-count gestures take priority over the drive/turn gestures
        # below: they control a separate, auxiliary motor and can be shown
        # with either hand regardless of how high it's raised.
        for shape in (left_shape, right_shape):
            if shape is HandShape.TWO_FINGERS:
                return RobotCommand(name="SINGLE_MOTOR_FORWARD", single_motor_speed=SINGLE_MOTOR_SPEED)
            if shape is HandShape.ONE_FINGER:
                return RobotCommand(name="SINGLE_MOTOR_REVERSE", single_motor_speed=-SINGLE_MOTOR_SPEED)

        left_raised = left_landmarks is not None and _is_raised(left_landmarks)
        right_raised = right_landmarks is not None and _is_raised(right_landmarks)

        if left_raised and right_raised:
            if left_shape is HandShape.OPEN_PALM and right_shape is HandShape.OPEN_PALM:
                return RobotCommand(name="FORWARD", left_speed=DRIVE_SPEED, right_speed=DRIVE_SPEED)
            if left_shape is HandShape.FIST and right_shape is HandShape.FIST:
                return RobotCommand(name="BACKWARD", left_speed=-DRIVE_SPEED, right_speed=-DRIVE_SPEED)
        elif left_raised:
            return RobotCommand(name="TURN_LEFT", left_speed=TURN_SPEED, right_speed=0.0)
        elif right_raised:
            return RobotCommand(name="TURN_RIGHT", left_speed=0.0, right_speed=TURN_SPEED)

        return RobotCommand(name="STOP")
