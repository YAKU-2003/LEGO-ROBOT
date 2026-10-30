"""Shared pieces for the gait wearable: MQTT link to the UNO Q, CSV loading,
windowing, and the features for both models.

  Model 1 ("peak")  -- ONE number per window: the biggest acceleration in one
                       direction on one axis (e.g. max of +ay). A 1-D decision
                       tree turns it into 3 thresholds.
  Model 2 ("full")  -- many numbers per window from all six IMU outputs (vector
                       lengths, rotation axes, cadence...). Random forest.

No neural nets anywhere.
"""

import csv
import glob
import os
import socket
import subprocess
import threading
import time
import pickle
from collections import Counter, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")

ACTIVITIES = ["standing", "walking", "running", "biking"]
ACTIVITY_ID = {"standing": 1, "walking": 2, "running": 3, "biking": 4}  # letter ids in sketch.ino (0 = ?)

FS = 50.0           # Hz, matches SAMPLE_MS = 20 in sketch.ino
WINDOW_S = 3.0      # long enough for >= 2 strides even when walking slowly / pedalling
STEP_S = 0.5        # windows overlap; a new one every half second
SEGMENT_S = 20.0    # recordings are cut into 20 s chunks for cross-validation groups
MAX_GAP_S = 0.25    # windows with a Wi-Fi dropout longer than this are skipped
G = 9.81

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
PEAK_CANDIDATES = [f"{s}{a}" for a in ("ax", "ay", "az") for s in ("+", "-")]  # "+ay" = max of ay, "-ay" = max of -ay


# ── MQTT link to the board ───────────────────────────────────────────────────

MOSQUITTO_EXE = r"C:\Program Files\mosquitto\mosquitto.exe"
MOSQUITTO_CONF = os.path.join(HERE, "mosquitto.conf")
IMU_TOPIC = "gait/imu"                # board -> laptop: "t,ax,ay,az,gx,gy,gz"  (m/s^2, rad/s)
PREDICTION_TOPIC = "gait/prediction"  # laptop -> board: "m1,m2" activity ids
STATUS_TOPIC = "gait/status"          # board -> laptop: text messages


def _port_open(host, port, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_broker(host="localhost", port=1883):
    """Start Mosquitto (listening on the network) if nothing is on the MQTT port yet."""
    if _port_open(host, port):
        return
    if not os.path.exists(MOSQUITTO_EXE):
        raise ConnectionError(f"Mosquitto not found at {MOSQUITTO_EXE}; install it or start a broker yourself")
    log = open(os.path.join(HERE, "mosquitto.log"), "w")
    subprocess.Popen([MOSQUITTO_EXE, "-c", MOSQUITTO_CONF, "-v"], stdout=log, stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(30):
        if _port_open(host, port):
            print("[gait] started Mosquitto broker (log: gait_identifier/laptop/mosquitto.log)")
            return
        time.sleep(0.1)
    raise ConnectionError("Started Mosquitto but it isn't accepting connections (see mosquitto.log)")


class IMUStream:
    """Receives the board's readings. Keeps the last few seconds in a buffer
    (for live.py) and optionally hands every sample to a callback (record.py)."""

    def __init__(self, host="localhost", port=1883, on_sample=None, keep_s=10.0):
        self.host, self.port = host, port
        self.on_sample = on_sample
        self.buffer = deque(maxlen=int(keep_s * FS * 1.5))
        self.lock = threading.Lock()
        self._last_rx = 0.0
        self._rx = 0
        self.client = None

    def connect(self):
        import paho.mqtt.client as mqtt

        ensure_broker(self.host, self.port)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"gait-laptop-{os.getpid()}")
        self.client.on_connect = lambda c, u, f, rc, p=None: c.subscribe([(IMU_TOPIC, 0), (STATUS_TOPIC, 0)])
        self.client.on_message = self._on_message
        self.client.connect(self.host, self.port, keepalive=10)
        self.client.loop_start()
        return self

    def _on_message(self, client, userdata, msg):
        text = msg.payload.decode(errors="replace")
        if msg.topic == STATUS_TOPIC:
            print(f"[board] {text}")
            return
        try:
            sample = tuple(float(v) for v in text.split(","))
        except ValueError:
            return
        if len(sample) != 7:
            return
        self._last_rx = time.monotonic()
        self._rx += 1
        with self.lock:
            self.buffer.append(sample)
        if self.on_sample:
            self.on_sample(sample)

    @property
    def connected(self):
        return time.monotonic() - self._last_rx < 1.0

    def take_rate(self, dt):
        """Samples/second since the last call (call once per dt)."""
        n, self._rx = self._rx, 0
        return n / dt

    def snapshot(self):
        with self.lock:
            return np.array(self.buffer) if self.buffer else np.zeros((0, 7))

    def send_prediction(self, m1, m2):
        if self.client:
            self.client.publish(PREDICTION_TOPIC, f"{ACTIVITY_ID.get(m1, 0)},{ACTIVITY_ID.get(m2, 0)}")

    def close(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# ── recordings on disk ───────────────────────────────────────────────────────

def save_csv(path, samples):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + CHANNELS)
        for s in samples:
            w.writerow([f"{s[0]:.4f}"] + [f"{v:.3f}" for v in s[1:]])


def load_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def label_of(path):
    """data/walking_strapA_20261027-153000.csv -> 'walking'"""
    return os.path.basename(path).split("_")[0]


def session_of(path):
    """data/walking_strapA_20261027-153000.csv -> 'strapA' (untagged files -> 'untagged').
    A session = one time the wearable was strapped on; train.py tests on unseen sessions."""
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    return parts[1] if len(parts) >= 3 else "untagged"


def recordings(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    return [(f, label_of(f)) for f in files if label_of(f) in ACTIVITIES]


# ── windowing ────────────────────────────────────────────────────────────────

def resample(data, n=int(WINDOW_S * FS)):
    """Board timestamps are slightly uneven; put the samples on an exact 50 Hz grid
    of n points starting at the first sample. Returns None if the data doesn't
    cover n points or has a gap (dropped Wi-Fi packets) longer than MAX_GAP_S."""
    t = data[:, 0]
    if len(t) < 2 or np.max(np.diff(t)) > MAX_GAP_S:
        return None
    grid = t[0] + np.arange(n) / FS
    if grid[-1] > t[-1] + 1.5 / FS:
        return None
    return np.column_stack([np.interp(grid, t, data[:, i]) for i in range(1, 7)])


def windows(data, window_s=WINDOW_S, step_s=STEP_S):
    """Yield (start_time, Nx6 array) for each sliding window of one recording."""
    t = data[:, 0]
    start = t[0]
    while start + window_s <= t[-1]:
        mask = (t >= start) & (t < start + window_s + 0.1)
        w = resample(data[mask], int(window_s * FS))
        if w is not None:
            yield start, w
        start += step_s


def latest_window(data, window_s=WINDOW_S):
    """The most recent window_s of a live buffer (Nx7 with times), or None."""
    if len(data) < 2:
        return None
    return resample(data[data[:, 0] >= data[-1, 0] - window_s - 1.0 / FS][-int(window_s * FS) - 2:],
                    int(window_s * FS))


def dataset(recs, extract):
    """All windows of all recordings -> X, y, meta.
    meta["recording"] / meta["session"] / meta["chunk"] say where each window came
    from, so cross-validation can hold out whole recordings or whole strap-ons."""
    X, y = [], []
    meta = {"recording": [], "session": [], "chunk": []}
    for path, label in recs:
        data = load_csv(path)
        if len(data) < 2:
            continue
        t0 = data[0, 0]
        name = os.path.basename(path)
        for start, w in windows(data):
            chunk = int((start - t0) // SEGMENT_S)
            X.append(extract(w))
            y.append(label)
            meta["recording"].append(name)
            meta["session"].append(session_of(path))
            # windows straddling a chunk boundary get their own group so no two groups share samples
            straddles = int((start + WINDOW_S - t0) // SEGMENT_S) != chunk
            meta["chunk"].append(f"{name}#{chunk}" + ("+" if straddles else ""))
    return np.array(X), np.array(y), {k: np.array(v) for k, v in meta.items()}


def is_still(data, still_s=0.4):
    """Live shortcut: True when the last still_s of a live buffer (Nx7 with times)
    has basically no motion -- |a| varies < 0.08 g and the leg rotates < 15 deg/s.
    Lets live.py say 'standing' at once instead of waiting for a full window."""
    if len(data) < 2 or data[-1, 0] - data[0, 0] < still_s:
        return False
    recent = data[data[:, 0] >= data[-1, 0] - still_s]
    amag = np.linalg.norm(recent[:, 1:4], axis=1) / G
    gmag = np.linalg.norm(recent[:, 4:7], axis=1)
    return amag.max() - amag.min() < 0.08 and gmag.max() < np.radians(15)


# ── Model 1 feature: peak acceleration in one direction ──────────────────────

def peak_feature(w, direction):
    """direction like '+ay' or '-az'. Returns the window's peak in g, measured from
    the window's median so gravity drops out. A raw axis carries whatever share of
    gravity the strap angle puts on it (a few degrees of tilt = 0.1 g), which made
    a still leg look like biking whenever the sensor sat a little differently."""
    sign = 1.0 if direction[0] == "+" else -1.0
    col = w[:, CHANNELS.index(direction[1:])]
    return float(np.max(sign * (col - np.median(col))) / G)


# ── Model 2 features ─────────────────────────────────────────────────────────
#
# Two feature sets, both built from all six IMU outputs. train.py tests both and
# keeps the one that does better on recordings / strap-ons it never saw.
#   "tiltfree" -- only things that don't change when the sensor is strapped on at
#                 a different angle: lengths of the accel/gyro vectors, the
#                 rotation's principal axes, cadence. (Preferred.)
#   "peraxis"  -- per-axis stats + gravity direction. Can win if the strap angle
#                 never changes, but tends to learn the strap angle instead of the motion.

def _corr(u, v):
    if u.std() < 1e-6 or v.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1])


def _autocorr_cadence(x, min_s=0.25, max_s=2.0):
    """Repeat rate (Hz) from the autocorrelation's highest peak, plus how regular it
    is (0..1). Finer than the FFT, which only resolves 1/WINDOW_S = 0.33 Hz steps --
    that's what separates a fast walk from a slow jog."""
    x = x - x.mean()
    if x.std() < 1e-6:
        return 0.0, 0.0
    n = len(x)
    ac = np.correlate(x, x, "full")[n - 1:]
    ac = ac / (ac[0] * (n - np.arange(n)) / n)  # unbiased, ac[0] = 1
    lo, hi = int(min_s * FS), min(int(max_s * FS), n - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(FS / lag), float(np.clip(ac[lag], 0, 1))


def _dominant_freq(x):
    """Strongest frequency in 0.3-6 Hz (step / pedal cadence) and how much of the power it holds."""
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / FS)
    band = (freqs >= 0.3) & (freqs <= 6.0)
    if not np.any(band) or spec[band].sum() == 0:
        return 0.0, 0.0
    i = np.argmax(spec[band])
    return float(freqs[band][i]), float(spec[band][i] / (spec[1:].sum() + 1e-9))


def peraxis_feature_names():
    names = []
    for ch in CHANNELS + ["amag", "gmag"]:
        names += [f"{ch}_{s}" for s in ("mean", "std", "min", "max", "p10", "p90", "range")]
    for ch in ("amag", "gmag", "gx", "gy", "gz"):
        names += [f"{ch}_freq", f"{ch}_freqpow"]
    names += ["corr_ax_ay", "corr_ax_az", "corr_ay_az", "corr_gx_gy", "corr_gx_gz", "corr_gy_gz"]
    names += ["grav_x", "grav_y", "grav_z", "jerk_mean", "jerk_max", "amag_peaks_per_s"]
    return names


def peraxis_features(w):
    a = w[:, 0:3] / G   # g
    g = w[:, 3:6]       # rad/s
    amag = np.linalg.norm(a, axis=1)
    gmag = np.linalg.norm(g, axis=1)
    cols = [a[:, 0], a[:, 1], a[:, 2], g[:, 0], g[:, 1], g[:, 2], amag, gmag]

    f = []
    for x in cols:
        p10, p90 = np.percentile(x, [10, 90])
        f += [x.mean(), x.std(), x.min(), x.max(), p10, p90, x.max() - x.min()]
    for x in (amag, gmag, g[:, 0], g[:, 1], g[:, 2]):
        f += list(_dominant_freq(x))

    f += [_corr(a[:, 0], a[:, 1]), _corr(a[:, 0], a[:, 2]), _corr(a[:, 1], a[:, 2]),
          _corr(g[:, 0], g[:, 1]), _corr(g[:, 0], g[:, 2]), _corr(g[:, 1], g[:, 2])]

    # Which way gravity points (= how the leg is oriented: upright vs lying on your back)
    grav = a.mean(axis=0)
    grav = grav / (np.linalg.norm(grav) + 1e-9)
    jerk = np.abs(np.diff(amag)) * FS
    # impacts per second: |a| crossing up through 1.5 g
    peaks = np.sum((amag[1:] > 1.5) & (amag[:-1] <= 1.5)) / (len(amag) / FS)
    f += list(grav) + [jerk.mean(), jerk.max(), peaks]
    return np.array(f, dtype=float)


def tiltfree_feature_names():
    names = []
    for ch in ("amag", "gmag"):
        names += [f"{ch}_{s}" for s in ("mean", "std", "min", "max", "p10", "p90", "range")]
    names += ["amag_freq", "amag_freqpow", "gmag_freq", "gmag_freqpow", "gmain_freq", "gmain_freqpow",
              "gmain_cadence", "gmain_regularity", "amag_cadence", "amag_regularity",
              "gyro_spread1", "gyro_spread2", "gyro_spread3", "gyro_planarity",
              "accdyn_spread1", "accdyn_spread2", "accdyn_spread3",
              "corr_amag_gmag", "jerk_mean", "jerk_max", "amag_peaks_per_s"]
    return names


def tiltfree_features(w):
    a = w[:, 0:3] / G   # g
    g = w[:, 3:6]       # rad/s
    amag = np.linalg.norm(a, axis=1)
    gmag = np.linalg.norm(g, axis=1)

    # |a| is measured relative to the window's own median: the sensor's idea of
    # "1 g" shifts a little between sessions (it read 1.00 g, then 1.01 g after
    # re-wiring), and the absolute level let the model tell sessions apart instead
    # of activities.
    f = []
    for x in (amag - np.median(amag), gmag):
        p10, p90 = np.percentile(x, [10, 90])
        f += [x.mean(), x.std(), x.min(), x.max(), p10, p90, x.max() - x.min()]

    # The leg swings mostly about one axis. Find it from the data (principal axis of
    # the gyro readings) instead of trusting the sensor's x/y/z, so it doesn't matter
    # how the sensor is turned on the strap. The eigenvalues say how much rotation
    # there is about each principal axis -- the same whatever the strap angle.
    gc = g - g.mean(axis=0)
    gvals, gvecs = np.linalg.eigh(np.cov(gc.T) + 1e-12 * np.eye(3))
    gmain = gc @ gvecs[:, -1]
    gspread = np.sqrt(np.clip(gvals[::-1], 0, None))
    adyn = a - a.mean(axis=0)  # motion only, gravity removed
    aspread = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov(adyn.T) + 1e-12 * np.eye(3))[::-1], 0, None))

    f += list(_dominant_freq(amag)) + list(_dominant_freq(gmag)) + list(_dominant_freq(gmain))
    f += list(_autocorr_cadence(gmain)) + list(_autocorr_cadence(amag))
    f += list(gspread) + [gspread[0] / (gspread.sum() + 1e-9)]
    f += list(aspread)

    jerk = np.abs(np.diff(amag)) * FS
    peaks = np.sum((amag[1:] > 1.5) & (amag[:-1] <= 1.5)) / (len(amag) / FS)
    f += [_corr(amag, gmag), jerk.mean(), jerk.max(), peaks]
    return np.array(f, dtype=float)


FEATURE_SETS = {
    "tiltfree": (tiltfree_feature_names, tiltfree_features),
    "peraxis": (peraxis_feature_names, peraxis_features),
}


# ── saving a recording (record.py and app.py) ────────────────────────────────

def save_recording(activity, session, samples, trim_s=2.0):
    """Trim trim_s off each end and save to data/<activity>_<session>_<time>.csv.
    Returns (path, n_samples, seconds); raises ValueError if too short."""
    if not samples:
        raise ValueError("no samples received")
    t0, t1 = samples[0][0] + trim_s, samples[-1][0] - trim_s
    kept = [s for s in samples if t0 <= s[0] <= t1]
    if len(kept) < WINDOW_S * FS:
        raise ValueError(f"only {len(kept)} samples after trimming -- too short")
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{activity}_{session}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    save_csv(path, kept)
    return path, len(kept), kept[-1][0] - kept[0][0]


# ── live classification with both models (live.py and app.py) ───────────────

class LiveClassifier:
    """Call update(buffer) every TICK_S. Every STEP_S it classifies the newest
    window with both models; stillness is checked every tick (model 2 only --
    it uses the gyro, which model 1 isn't allowed to see) so 'standing' shows up
    the moment the leg stops instead of ~2 s later."""

    TICK_S = 0.1

    def __init__(self, smooth=3):
        self.smooth = smooth
        self.peak = self.full = None
        self.reload()

    def reload(self):
        """(Re)load the trained models. Returns False if there aren't any yet."""
        try:
            with open(os.path.join(MODEL_DIR, "peak_model.pkl"), "rb") as f:
                peak = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "full_model.pkl"), "rb") as f:
                full = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return False
        self.peak, self.full = peak, full
        self.full_features = FEATURE_SETS[full.get("feature_set", "peraxis")][1]
        self.hist1, self.hist2 = deque(maxlen=self.smooth), deque(maxlen=self.smooth)
        self._tick, self._was_still = 0, False
        self.result = None
        return True

    @property
    def ready(self):
        return self.peak is not None

    def update(self, data):
        """data = live buffer (Nx7 with times). Returns the latest result dict, or
        None when there's no full window yet. 'fresh' is True when it just changed."""
        if not self.ready:
            return None
        self._tick += 1
        still = is_still(data)
        if still and not self._was_still:
            self.hist2.clear()
            self.hist2.extend(["standing"] * self.smooth)
            self._tick = 0  # update right now
        self._was_still = still
        if self._tick % round(STEP_S / self.TICK_S):
            if self.result:
                self.result["fresh"] = False
            return self.result

        w = latest_window(data)
        if w is None:
            self.result = None
            return None
        p = peak_feature(w, self.peak["direction"])
        self.hist1.append(self.peak["model"].predict([[p]])[0])
        x2 = self.full_features(w)
        if still:
            self.hist2.append("standing")
            probs = {"standing": 1.0}
        else:
            pr = self.full["model"].predict_proba([x2])[0]
            probs = {c: float(v) for c, v in zip(self.full["model"].classes_, pr)}
            self.hist2.append(max(probs, key=probs.get))
        self.result = {
            "fresh": True,
            "model1": Counter(self.hist1).most_common(1)[0][0],
            "model2": Counter(self.hist2).most_common(1)[0][0],
            "raw1": list(self.hist1), "raw2": list(self.hist2),
            "peak": p, "direction": self.peak["direction"],
            "probs": probs, "still": bool(still),
        }
        return self.result


# ── sensor sanity check ──────────────────────────────────────────────────────

GYRO_OFFSET_LIMIT = 60.0  # deg/s


def gyro_offset(data):
    """Biggest average rotation rate on any gyro axis (deg/s) of an Nx7 array.
    A leg swings back and forth, so over a few seconds every axis should average
    out near 0. A big average means the gyro is stuck with an offset (seen on
    this MPU6050 after a hard knock: gz jumped to -373 deg/s and stayed there).
    Fix: unplug the board for a few seconds so the sensor resets."""
    if len(data) == 0:
        return 0.0
    return float(np.max(np.abs(np.degrees(data[:, 4:7].mean(axis=0)))))


def mean_accel_g(data):
    """Average |a| in g. Gravity alone is 1 g, so ~0 means the sensor isn't really being read."""
    if len(data) == 0:
        return 1.0
    return float(np.linalg.norm(data[:, 1:4], axis=1).mean() / G)
