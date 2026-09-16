"""Convert MediaPipe hand landmarks into per-wheel motor speeds.

Gesture scheme: each hand's wrist height relative to a fixed neutral
height (the vertical center of the frame) controls that side's wheel
speed, normalized by the hand's own size so it works regardless of how
far the user's hand is from the camera. Hand size (wrist-to-middle-
knuckle distance) is used as the scale reference because -- unlike a
body's shoulder width -- a hand has no separate fixed anchor point to
measure against; the hand's own dimensions are the only thing available
that scale the same way distance-from-camera does.
"""

import math
from dataclasses import dataclass

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

# A wrist within this many hand-heights of the neutral height stays in
# this band and produces zero speed, so a hand held near-neutral doesn't
# cause drift.
DEAD_ZONE = 0.2

# Wrist this many hand-heights above/below the neutral height maps to full
# speed. Separate forward/backward ranges because the two directions have
# different usable ranges of motion: a hand can be raised well above the
# neutral height for forward, but lowering it too far tends to push the
# hand out of frame (or below VISIBILITY through occlusion by the body)
# before it can travel as far -- so full reverse needs a shorter,
# easier-to-reach range.
FULL_SPEED_RANGE_FORWARD = 2.5
FULL_SPEED_RANGE_BACKWARD = 1.2

# Vertical position (0 = top of frame, 1 = bottom) treated as a hand's
# "resting" height, i.e. the center of the frame. Assumes the user keeps
# their hands roughly centered vertically when not gesturing.
NEUTRAL_HEIGHT = 0.5

MAX_SPEED_PERCENT = 100

# Exponential moving average factor for smoothing landmark jitter frame to frame.
# Lower = smoother but laggier; higher = snappier but noisier.
SMOOTHING_ALPHA = 0.4

# Minimum handedness classification confidence before a detected hand is
# trusted. Below this, MediaPipe's Left/Right label for the hand is
# unreliable, so we drop the hand for this frame rather than risk driving
# the wrong wheel.
HANDEDNESS_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class WheelSpeeds:
    left: float
    right: float


def _hand_scale(landmarks) -> float:
    """Distance from wrist to middle-knuckle, used as this hand's size reference."""
    wrist = landmarks[HandLandmark.WRIST]
    middle_mcp = landmarks[HandLandmark.MIDDLE_FINGER_MCP]
    return math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)


def _hand_speed(landmarks) -> float:
    """Map one hand's wrist height to a speed percentage in [-100, 100].

    Image y-coordinates increase downward, so a raised hand has a smaller
    y than the neutral height. Returns 0 if the hand isn't present or its
    size can't be measured (e.g. landmarks are degenerate).
    """
    if landmarks is None:
        return 0.0

    scale = _hand_scale(landmarks)
    if scale <= 1e-6:
        return 0.0

    wrist = landmarks[HandLandmark.WRIST]
    raise_amount = (NEUTRAL_HEIGHT - wrist.y) / scale

    full_speed_range = FULL_SPEED_RANGE_FORWARD if raise_amount > 0 else FULL_SPEED_RANGE_BACKWARD
    magnitude = (abs(raise_amount) - DEAD_ZONE) / (full_speed_range - DEAD_ZONE)
    magnitude = max(0.0, min(1.0, magnitude))

    sign = 1.0 if raise_amount > 0 else -1.0
    return sign * magnitude * MAX_SPEED_PERCENT


class GestureController:
    """Tracks smoothed left/right wheel speeds across frames."""

    def __init__(self):
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0

    def update(self, hands_result, mirrored: bool) -> WheelSpeeds:
        """Compute new wheel speeds from one frame's hand landmarks.

        Parameters:
            hands_result: A HandLandmarkerResult, with `hand_landmarks`
                (one landmark list per detected hand) and `handedness`
                (one classification list per detected hand, aligned by
                index with `hand_landmarks`). Zero, one, or two hands may
                be present in a given frame.
            mirrored: Whether the source frame was flipped horizontally for
                a natural "mirror" display. MediaPipe's Left/Right labels
                describe the image as captured, so when the frame is
                mirrored, the person's actual right hand is reported as
                MediaPipe's "Left". Swap here so a raised right hand
                always drives the right wheel.
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

        target_left = _hand_speed(left_landmarks)
        target_right = _hand_speed(right_landmarks)

        self._smoothed_left += SMOOTHING_ALPHA * (target_left - self._smoothed_left)
        self._smoothed_right += SMOOTHING_ALPHA * (target_right - self._smoothed_right)

        return WheelSpeeds(left=self._smoothed_left, right=self._smoothed_right)

    def reset(self):
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0
