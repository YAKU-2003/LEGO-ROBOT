# LEGO-ROBOT

Codes and stuff for the ME193: AI in Mobile Robotics.

Control a LEGO robot's motors with hand and arm gestures seen through a webcam. This repo has two different gesture-recognition approaches, both working end-to-end on live video, plus one script that talks to real LEGO hardware. The pieces aren't wired together yet -- see [Status](#status) below.

## Gesture schemes

### Continuous: wrist height -> wheel speed

[`pose_gesture_controller.py`](pose_gesture_controller.py) (full-body pose) and [`hand_gesture_controller.py`](hand_gesture_controller.py) (hands only) both map an arm's height to a continuous wheel speed from -100% (full reverse) to +100% (full forward):

- Wrist at the neutral height -> that wheel is stopped (a small dead zone so a resting arm doesn't drift).
- Raise a hand/arm -> that side's wheel drives forward, faster the higher you raise it.
- Lower it below neutral -> that side reverses.
- Left controls the left wheel, right controls the right wheel. Raise one side and lower the other to spin in place.

The two versions differ only in what "neutral" and "scale" mean, because a hand alone has no shoulder to measure against:

| | `pose_gesture_controller.py` | `hand_gesture_controller.py` |
|---|---|---|
| Neutral height | that arm's own shoulder | fixed frame center |
| Scale reference | shoulder width | wrist-to-middle-knuckle span |
| Sees | full body (shoulders, wrists, face, ...) | hands only |

Shoulder/hand width is used as the scale reference (instead of, say, shoulder-to-hip torso length) because it stays in frame even when a webcam only frames head-to-waist. Reaching full speed takes a bigger raise going forward than going backward, since a lowered arm tends to leave the camera's view sooner than a raised one. Both controllers smooth the output with an exponential moving average to reduce frame-to-frame jitter.

[`run_pose_gesture.py`](run_pose_gesture.py) additionally watches for a `Closed_Fist` or `Open_Palm` gesture (via [`pose_camera.py`](pose_camera.py)'s built-in `GestureRecognizer`) as an emergency-stop override on top of the arm control.

### Discrete: hand shape -> one-shot command

[`hand_command_controller.py`](hand_command_controller.py) takes a different approach: instead of a continuous speed, it recognizes a fixed hand shape each frame and outputs one command:

- Both hands raised, palms open -> drive forward.
- Both hands raised, fists -> drive backward.
- Only the left hand raised -> turn left (pivot on the right wheel).
- Only the right hand raised -> turn right (pivot on the left wheel).
- Two fingers up (either hand) -> spin an auxiliary single motor one way.
- One finger up (either hand) -> spin it the other way.
- Anything else, or no hands visible -> stop.

Finger count is estimated by comparing each fingertip's distance from the wrist to its pip joint's distance (farther = extended); the thumb is excluded because its joint geometry doesn't fit that test as reliably as the other four fingers. "Raised" means the wrist is above the vertical center of the frame -- there's no shoulder reference available from hand landmarks alone, so it's a fixed frame position rather than something scaled to the user.

## Setup

```
python -m venv my_env
my_env\Scripts\activate        # Windows
pip install opencv-python mediapipe numpy
```

## Run

```
python run_hand_commands.py    # discrete hand-shape commands (hands only, no face)
python run_hand_gesture.py     # continuous wrist-height control (hands only)
python run_pose_gesture.py     # continuous arm-height control + fist/palm stop gesture (full body)
```

Press `q` in the video window to quit. `run_hand_commands.py` and `run_hand_gesture.py` need [`hand_landmarker.task`](hand_landmarker.task) (already checked into the repo). `run_pose_gesture.py` downloads its two models (pose + gesture, ~14MB total) automatically into `models/` on first run.

There are also two scripts unrelated to gesture control:
- [`test_single_motor.py`](test_single_motor.py) -- connects to a real LEGO Education Single Motor over Bluetooth and spins it, as a hardware smoke test.
- [`grayscale_image.py`](grayscale_image.py) -- a standalone OpenCV playground (grayscale -> threshold -> erode/dilate -> boundary extraction -> convolution-kernel presets) used to prototype image-processing ideas, not currently connected to the gesture or motor code.

## Status

None of the gesture scripts drive a motor yet -- they compute and display wheel speeds/commands on screen, but stop there. The only script that talks to real hardware is `test_single_motor.py`. Wiring a `GestureController`/`HandCommandController` output into an actual `SingleMotor` (or a two-motor drivetrain) each frame is the next step.

## How does Python talk to the LEGO hardware?

`test_single_motor.py` connects over Bluetooth Low Energy via the [`legoeducation`](https://pypi.org/project/legoeducation/) package, which wraps [`bleak`](https://pypi.org/project/bleak/) (cross-platform BLE) and speaks LEGO's binary RPC protocol to the hub. Calls like `motor_run_for_time(...)` and `motor_run_for_degrees(...)` take a `speed` and `direction`.

Internally, `legoeducation` runs all I/O through a single asyncio event loop (`background_worker.py`) that owns the BLE connection. The public API is synchronous by default -- each call takes a `blocking` argument, and a blocking-looking call is bridged onto the async loop under the hood. Once motor control is wired into the gesture loop, calls would need `blocking=False` so sending a new command every video frame doesn't stall the webcam loop waiting for a BLE acknowledgment.

## How did you train it, and what are its limitations?

Nothing here is trained from scratch. All three controllers sit on top of pretrained MediaPipe models -- Pose Landmarker, Hand Landmarker, and Gesture Recognizer -- downloaded automatically as `.task` files. Our own code is hand-written, rule-based logic (arm/wrist height -> speed, fingertip-distance -> finger count) applied to those models' landmark output; no custom classifier was trained.

Limitations:
- Requires decent, even lighting and the relevant landmarks (shoulders/wrists, or hands) visible in frame; occlusion or being partially off-screen breaks tracking.
- Tracks one person (pose) or up to two hands at a time; a crowded frame can confuse detection.
- The built-in `GestureRecognizer` only knows its fixed vocabulary of canned gestures (e.g. `Closed_Fist`, `Open_Palm`) -- there's no way to add a custom gesture without training a new classifier, which we didn't do.
- Raw landmark coordinates are jittery frame-to-frame; the continuous controllers smooth speed with an exponential moving average, trading a little responsiveness for stability. The discrete `HandCommandController` does not smooth, so a borderline hand shape can flicker between commands.
- Dead zones, full-speed ranges, the "raised" height threshold, the finger-extended margin, and confidence thresholds are fixed constants tuned by hand-testing, not per-user calibrated -- they may feel off for very different body proportions or camera angles.
- No motor is actually being driven yet (see [Status](#status)), so real-world control latency and behavior are still unverified.
