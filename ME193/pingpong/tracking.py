"""Low-latency webcam + pose pipeline for the ping-pong game.

Why this exists (the racket used to lag behind your hand):
    - cap.read() blocks until the next frame, and the old loop did
      read -> pose -> draw one after another, so everything waited on everything.
    - MediaPipe's VIDEO mode smooths landmarks over time, which makes fast
      swings trail behind your real hand.

What this does instead:
    CameraThread  grabs frames non-stop and keeps only the newest one.
    PoseThread    runs pose detection on the newest frame in the background
                  (IMAGE mode = no built-in smoothing) and stamps each result
                  with the time its frame was captured.
    HandTracker   One-Euro filter per landmark (removes jitter when you're
                  still, almost no lag when you move fast), predicts forward to
                  the moment the on-screen frame was captured (the pose is
                  always ~1 frame older than the picture), and holds the last
                  position for a moment if the wrist blurs out mid-swing.
"""

import math
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# MediaPipe pose landmark indices
POINTS = {"ls": 11, "rs": 12, "le": 13, "re": 14, "lw": 15, "rw": 16}

LEAD_S = 0.01         # tiny extra lead beyond the displayed frame's capture time
MAX_PREDICT_S = 0.10  # never extrapolate further than this
HOLD_S = 0.30        # keep showing a landmark this long after it was lost


# ── camera ───────────────────────────────────────────────────────────────────

class CameraThread:
    """Reads the webcam continuously; latest() returns the newest mirrored frame."""

    def __init__(self, index=0, width=1280, height=720):
        cap = cv2.VideoCapture(index)  # default backend (DirectShow crashes on some laptop webcams)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam {index} (try --camera 1)")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam opened but returned no frame")
        self.cap = cap
        self.shape = frame.shape
        self._frame, self._t, self._id = cv2.flip(frame, 1), time.monotonic(), 0
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self.running = True
        self.fps = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        n, t0 = 0, time.monotonic()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.monotonic()
            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            with self._new:
                self._frame, self._t, self._id = frame, t, self._id + 1
                self._new.notify_all()
            n += 1
            if t - t0 >= 1.0:
                self.fps, n, t0 = n / (t - t0), 0, t

    def latest(self):
        """(frame, capture_time, frame_id). The frame is shared: copy before drawing on it."""
        with self._lock:
            return self._frame, self._t, self._id

    def wait_newer(self, frame_id, timeout=0.1):
        """Block until a frame newer than frame_id exists; return latest()."""
        with self._new:
            if self._id <= frame_id:
                self._new.wait(timeout)
            return self._frame, self._t, self._id

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ── pose ─────────────────────────────────────────────────────────────────────

def create_landmarker(model_path, smooth=False):
    """IMAGE mode = no temporal smoothing (lowest lag). smooth=True = VIDEO mode (old behaviour)."""
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path,
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO if smooth else vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


class PoseThread:
    """Runs pose detection on the newest camera frame, in the background."""

    def __init__(self, camera, model_path, smooth=False):
        self.camera = camera
        self.smooth = smooth
        self.landmarker = create_landmarker(model_path, smooth)
        self._lock = threading.Lock()
        self._pose, self._t, self._id = None, 0.0, -1
        self.fps = 0.0
        self.ms = 0.0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_id, last_ts = -1, -1
        n, t0 = 0, time.monotonic()
        while self.running:
            frame, t_cap, fid = self.camera.wait_newer(last_id)
            if fid == last_id:
                continue
            last_id = fid
            a = time.monotonic()
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.smooth:
                ts = max(last_ts + 1, int(t_cap * 1000))
                last_ts = ts
                result = self.landmarker.detect_for_video(image, ts)
            else:
                result = self.landmarker.detect(image)
            pose = None
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                pose = {}
                for key, i in POINTS.items():
                    p = lm[i]
                    vis = 1.0 if p.visibility is None else p.visibility
                    pose[key] = (p.x * w, p.y * h, vis)
            with self._lock:
                self._pose, self._t, self._id = pose, t_cap, fid
            self.ms = 0.9 * self.ms + 0.1 * (time.monotonic() - a) * 1000
            n += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now

    def latest(self):
        """(pose dict or None, capture time of its frame, frame id)."""
        with self._lock:
            return self._pose, self._t, self._id

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.landmarker.close()


# ── filtering + prediction ───────────────────────────────────────────────────

class OneEuro:
    """One-Euro filter (Casiez et al. 2012): cutoff rises with speed, so slow
    motion is smoothed hard and fast motion passes through with little lag."""

    def __init__(self, min_cutoff=1.2, beta=0.008, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None or t <= self.t:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class HandTracker:
    """Smoothed + slightly predicted landmark positions for drawing the racket."""

    def __init__(self):
        self.filters = {k: (OneEuro(), OneEuro()) for k in POINTS}
        self.state = {}  # key -> (x, y, vx, vy, t_seen)
        self.last_id = -1

    def update(self, pose, t, frame_id):
        if frame_id == self.last_id or pose is None:
            return
        self.last_id = frame_id
        for key, (x, y, vis) in pose.items():
            if vis < 0.4:
                continue
            fx, fy = self.filters[key]
            sx, sy = fx(x, t), fy(y, t)
            self.state[key] = (sx, sy, fx.dx or 0.0, fy.dx or 0.0, t)

    def get(self, t_target):
        """{key: (x, y, vis)} predicted for time t_target (use the capture time of
        the frame being displayed so the racket lines up with your hand in it).
        vis drops to 0 after HOLD_S without a sighting."""
        out = {}
        for key in POINTS:
            s = self.state.get(key)
            if s is None:
                out[key] = (0.0, 0.0, 0.0)
                continue
            x, y, vx, vy, t = s
            age = t_target - t
            if age > HOLD_S:
                out[key] = (x, y, 0.0)
                continue
            ahead = min(MAX_PREDICT_S, max(0.0, age + LEAD_S))
            speed = math.hypot(vx, vy)
            if speed > 4000:  # px/s; ignore crazy spikes
                vx, vy = vx * 4000 / speed, vy * 4000 / speed
            out[key] = (x + vx * ahead, y + vy * ahead, 1.0)
        return out
