"""Convert MediaPipe pose landmarks into per-wheel motor speeds.

Gesture scheme: each wrist's height relative to its own shoulder controls
that side's wheel speed, normalized by shoulder width so it works
regardless of how far the user stands from the camera. Shoulder width
(rather than shoulder-to-hip torso height) is the scale reference because
it stays in frame even when a webcam only frames head-to-waist, which
cuts the hips out.
"""

from dataclasses import dataclass

from mediapipe.tasks.python import vision

PoseLandmark = vision.PoseLandmark

# Wrist at shoulder height stays inside this band and produces zero speed,
# so a resting arm doesn't cause drift.
DEAD_ZONE = 0.2

# Wrist this many shoulder-widths above/below the shoulder maps to full
# speed. Separate forward/backward ranges because the two directions have
# different usable ranges of motion: an arm can reach well above the head
# for forward, but lowering an arm past the waist tends to drop the wrist
# below the camera's view (or below VISIBILITY_THRESHOLD) before it can
# travel as far -- so full reverse needs a shorter, easier-to-reach range.
FULL_SPEED_RANGE_FORWARD = 2.5
FULL_SPEED_RANGE_BACKWARD = 1.2

MAX_SPEED_PERCENT = 100

# Exponential moving average factor for smoothing landmark jitter frame to frame.
# Lower = smoother but laggier; higher = snappier but noisier.
SMOOTHING_ALPHA = 0.4

# MediaPipe still reports a guessed x/y for landmarks it can't actually see
# (e.g. a wrist out of frame or behind the body), just with a low confidence
# score. Below this threshold we treat the landmark as untracked rather than
# trust its (often wrong) position.
VISIBILITY_THRESHOLD = 0.5


@dataclass
class WheelSpeeds:
    left: float
    right: float


def _is_tracked(landmark) -> bool:
    return landmark.visibility >= VISIBILITY_THRESHOLD and landmark.presence >= VISIBILITY_THRESHOLD


def _arm_speed(shoulder, wrist, other_shoulder) -> float:
    """Map one arm's wrist height to a speed percentage in [-100, 100].

    Image y-coordinates increase downward, so a raised wrist has a smaller
    y than the shoulder. Returns 0 if the shoulder or wrist aren't
    confidently tracked (e.g. the wrist is out of frame).
    """
    if not (_is_tracked(shoulder) and _is_tracked(wrist)):
        return 0.0

    shoulder_width = abs(other_shoulder.x - shoulder.x)
    if shoulder_width <= 1e-6:
        return 0.0

    raise_amount = (shoulder.y - wrist.y) / shoulder_width

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

    def update(self, landmarks, mirrored: bool) -> WheelSpeeds:
        """Compute new wheel speeds from one frame's pose landmarks.

        Parameters:
            landmarks: List of normalized landmarks from PoseLandmarker,
                indexed by PoseLandmark enum values.
            mirrored: Whether the source frame was flipped horizontally for
                a natural "mirror" display. MediaPipe's LEFT_*/RIGHT_*
                labels describe the image as captured, so when the frame is
                mirrored, the person's actual right arm is reported as
                MediaPipe's "left" side. Swap here so a raised right arm
                always drives the right wheel.
        """
        left_landmarks = (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_WRIST)
        right_landmarks = (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_WRIST)
        if mirrored:
            left_landmarks, right_landmarks = right_landmarks, left_landmarks

        left_shoulder, left_wrist = (landmarks[i] for i in left_landmarks)
        right_shoulder, right_wrist = (landmarks[i] for i in right_landmarks)

        target_left = _arm_speed(left_shoulder, left_wrist, right_shoulder)
        target_right = _arm_speed(right_shoulder, right_wrist, left_shoulder)

        self._smoothed_left += SMOOTHING_ALPHA * (target_left - self._smoothed_left)
        self._smoothed_right += SMOOTHING_ALPHA * (target_right - self._smoothed_right)

        return WheelSpeeds(left=self._smoothed_left, right=self._smoothed_right)

    def reset(self):
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0
