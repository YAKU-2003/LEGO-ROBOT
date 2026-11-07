"""Shared pieces for the step counter: recordings with hand-counted steps, the
step labeler, the neural net, the live counter and two classical baselines.

Same hardware and data link as the gait identifier (UNO Q + MPU6050 on the
ankle, 50 Hz over MQTT) -- the MQTT/CSV code is imported from its gaitlib.py.

How the net counts steps
------------------------
StepNet is a small 1-D convolutional network. It reads all six IMU outputs
(plus |a| and |w|) and outputs a "step density" for every sample: about 0
between steps and a small bump around each step whose area is 1 step. The
count is the sum of the density. Because it's fully convolutional it runs on
any length of signal: a 6 s live window or a whole recording.

Labels
------
You count your steps (both feet) out loud while recording and type the total
when you stop -- that number is the ground truth. label_events() places the
steps in time from the gyro (one swing of the strapped leg = 2 steps) and is
only trusted when its total agrees with your count; otherwise the recording
still trains the net through its total alone.
"""

import glob
import json
import os
import re
import sys
import time

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "gait_identifier", "laptop"))
sys.path.insert(0, GAIT_DIR)
import gaitlib  # noqa: E402  (MQTT link, CSV helpers, broker start-up)

FS = int(gaitlib.FS)
G = gaitlib.G
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stepnet.pt")
STEPS_TOPIC = "gait/steps"  # laptop -> board: running step count for the LED matrix

# What you can record. Zero-step kinds teach the net what is NOT a step.
KINDS = {
    "walk": "normal walk",
    "slowwalk": "slow walk",
    "fastwalk": "fast walk",
    "run": "jog / run",
    "stairs": "stairs up or down",
    "stopgo": "walk, stop, turn, walk again",
    "still": "standing still (0 steps)",
    "fidget": "seated: swing / tap the leg (0 steps)",
    "bike": "biking / air-biking (0 steps)",
}
ZERO_KINDS = {"still", "fidget", "bike"}
GAIT_ZERO = {"standing": "still", "biking": "bike"}  # gait_identifier recordings reused as 0-step data

NAME_RE = re.compile(r"^(?P<kind>[a-z]+)_(?P<steps>\d+)steps_(?P<session>[A-Za-z0-9-]+)_(?P<time>[0-9-]+)\.csv$")

WINDOW_S = 6.0     # live: the net sees the last 6 s ...
HOP_S = 0.5        # ... every 0.5 s ...
CONTEXT_S = 1.0    # ... and counts the half second that ends 1 s ago (so every step has context on both sides)
SIGMA_S = 0.06     # width of the bump drawn around each labelled step
MAX_GAP_S = 0.5    # longer Wi-Fi gaps are skipped, not interpolated over


# ── recordings ───────────────────────────────────────────────────────────────

def recording_name(kind, steps, session):
    return f"{kind}_{int(steps)}steps_{session}_{time.strftime('%Y%m%d-%H%M%S')}.csv"


def recordings(include_gait=True):
    """All recordings with a known step count: step_counter/data/<kind>_<N>steps_<session>_<time>.csv,
    plus the gait identifier's standing / biking recordings (0 steps)."""
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        m = NAME_RE.match(os.path.basename(p))
        if m:
            out.append({"path": p, "name": os.path.basename(p), "kind": m["kind"], "steps": int(m["steps"]),
                        "session": m["session"], "source": "steps"})
    if include_gait:
        for p, label in gaitlib.recordings():
            if label in GAIT_ZERO:
                out.append({"path": p, "name": os.path.basename(p), "kind": GAIT_ZERO[label], "steps": 0,
                            "session": "gait-" + gaitlib.session_of(p), "source": "gait"})
    return out


def save_recording(kind, steps, session, samples):
    """No trimming here (unlike the gait recorder): every counted step has to stay in."""
    if len(samples) < 3 * FS:
        raise ValueError(f"only {len(samples)} samples -- too short")
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, recording_name(kind, steps, session))
    gaitlib.save_csv(path, samples)
    return path


def to_grid(data):
    """Board samples (t, ax..gz) -> exact 50 Hz grid. Returns (t, X[n,6], ok[n]) where ok
    is False inside Wi-Fi gaps longer than MAX_GAP_S."""
    t = data[:, 0]
    grid = np.arange(np.ceil(t[0] * FS), np.floor(t[-1] * FS) + 1) / FS
    X = np.column_stack([np.interp(grid, t, data[:, i]) for i in range(1, 7)])
    ok = np.ones(len(grid), bool)
    for i in np.nonzero(np.diff(t) > MAX_GAP_S)[0]:
        ok[(grid > t[i]) & (grid < t[i + 1])] = False
    return grid, X, ok


def net_inputs(X):
    """X[n,6] in m/s^2 and rad/s -> the net's 8 input channels, roughly unit-scaled:
    a (g), w / 4 (rad/s), |a| - 1, |w| / 4."""
    a = X[:, 0:3] / G
    w = X[:, 3:6] / 4.0
    return np.column_stack([a, w, np.linalg.norm(a, axis=1) - 1.0, np.linalg.norm(w, axis=1)]).astype(np.float32)


CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz", "|a|", "|w|"]


# ── step labeler (places your counted steps in time) ─────────────────────────

def swing_signal(X):
    """Rotation rate about the leg's main swing axis (PCA of the gyro), signed so the
    swing peaks are positive. Independent of how the sensor sits on the strap."""
    g = X[:, 3:6] - X[:, 3:6].mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(g.T) + 1e-9 * np.eye(3))
    s = g @ vecs[:, -1]
    if abs(np.percentile(s, 0.5)) > abs(np.percentile(s, 99.5)):
        s = -s
    return np.convolve(s, np.ones(5) / 5, mode="same")


def strapped_strikes(X):
    """Heel strikes of the strapped foot: one per swing of that leg. The strike is the
    |a| spike in the 0.05-0.4 s after the swing peak."""
    s = swing_signal(X)
    if np.percentile(np.abs(s), 99) < 1.0:  # leg barely rotating: no strides
        return np.array([], int)
    peaks, _ = find_peaks(s, height=max(1.0, 0.35 * np.percentile(s, 99)), distance=int(0.4 * FS),
                          prominence=0.8)
    amag = np.linalg.norm(X[:, 0:3], axis=1)
    strikes = []
    for p in peaks:
        lo, hi = p + int(0.05 * FS), min(len(amag), p + int(0.4 * FS))
        strikes.append(lo + int(np.argmax(amag[lo:hi])) if hi > lo else p)
    return np.array(strikes, int)


def label_events(X, n_steps):
    """Sample indices of all n_steps steps (both feet), or None if the gyro can't
    account for your count. Returns (events or None, info)."""
    if n_steps == 0:
        return np.array([], int), {"auto": 0, "matched": True}
    k = strapped_strikes(X)
    info = {"strides": len(k)}
    if len(k) == 0:
        info.update(auto=0, matched=False)
        return None, info
    # Split into walking bouts at pauses; inside a bout the other foot lands halfway between strikes.
    bouts = np.split(k, np.nonzero(np.diff(k) > 1.8 * FS)[0] + 1)
    events, ends = [], []
    for b in bouts:
        events += list(b)
        events += [int((x + y) / 2) for x, y in zip(b[:-1], b[1:])]
        half = int(np.median(np.diff(b)) / 2) if len(b) > 1 else int(0.55 * FS)
        ends += [b[-1] + half, b[0] - half]  # other foot finishing a bout / starting it
    base = len(events)
    # Add the bout-boundary steps of the other foot until the total matches your count.
    extra = [e for e in ends if 0 <= e < len(X)][:max(0, n_steps - base)]
    events = np.sort(np.array(events + extra, int))
    tol = max(2, round(0.04 * n_steps))
    info.update(auto=len(events), matched=abs(len(events) - n_steps) <= tol)
    return (events if info["matched"] else None), info


def density_target(n, events):
    """Sum of Gaussian bumps (each with area 1) at the step times: what the net should output."""
    t = np.arange(n)
    y = np.zeros(n, np.float32)
    sig = SIGMA_S * FS
    for e in events:
        bump = np.exp(-0.5 * ((t - e) / sig) ** 2)
        y += bump / bump.sum()
    return y


# ── the neural net ───────────────────────────────────────────────────────────

def build_net(dropout=0.1):
    import torch.nn as nn

    class StepNet(nn.Module):
        """Dilated 1-D conv net. Each layer looks 2x further apart than the one below,
        so 4 layers of width 9 see ~2.4 s -- about two full strides -- around each sample."""

        def __init__(self, ch=len(CHANNEL_NAMES), width=32):
            super().__init__()
            layers, c_in = [], ch
            for d in (1, 2, 4, 8):
                # replicate padding: past the ends of the signal it looks like the sensor holding still.
                # (Zero padding looks like gravity vanishing -> a fake jolt -> fake steps at the start.)
                layers += [nn.Conv1d(c_in, width, 9, padding=4 * d, dilation=d, padding_mode="replicate"),
                           nn.BatchNorm1d(width), nn.ReLU(),
                           nn.Dropout(dropout)]
                c_in = width
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(width, 1, 1)
            self.out = nn.Softplus()

        def forward(self, x):            # x: [batch, 8 channels, time]
            return self.out(self.head(self.body(x)))[:, 0]  # [batch, time] step density

    return StepNet()


def load_model(path=None):
    import torch

    path = path or MODEL_PATH
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck.get("metrics", {})


def density(net, inputs):
    """Run the net on [n, 8] inputs; returns the [n] step density."""
    import torch

    with torch.no_grad():
        return net(torch.from_numpy(inputs.T[None]))[0].numpy()


# ── live / streaming counter (also used to score the net offline) ────────────

class StreamCounter:
    """Feed it the growing buffer of board samples every HOP_S. Every call it runs the
    net on the last WINDOW_S and adds the density of the samples that are now
    CONTEXT_S old -- each sample is counted exactly once, with context on both sides."""

    def __init__(self, net, from_start=False):
        self.net = net
        self.from_start = from_start  # offline scoring: also count the first seconds of the data
        self.reset()

    def reset(self):
        self.count = 0.0
        self.done_until = None      # grid time up to which density has been counted
        self.steps = []             # times of counted steps (density peaks), for the page
        self.trace = []             # (t, density) of counted samples, for the page

    def update(self, data, flush=False):
        """flush=True (end of a recording) also counts the last CONTEXT_S."""
        if self.net is None or len(data) < 2:
            return
        data = data[data[:, 0] >= data[-1, 0] - WINDOW_S - 0.5]
        t, X, ok = to_grid(data)
        if len(t) < WINDOW_S * FS * 0.5 and not self.from_start:
            return
        d = density(self.net, net_inputs(X))
        end = t[-1] if flush else t[-1] - CONTEXT_S
        if self.done_until is not None:
            start = self.done_until
        else:
            start = t[0] - 1.0 if self.from_start else end - HOP_S
        sel = (t > start + 1e-6) & (t <= end + 1e-6)
        if not sel.any():
            return
        dd = np.where(ok, d, 0.0)
        self.count += float(dd[sel].sum())
        self.trace += [(float(a), float(b)) for a, b in zip(t[sel], dd[sel])]
        self.trace = self.trace[-int(30 * FS):]
        pk, _ = find_peaks(dd, height=0.08, distance=int(0.2 * FS))
        self.steps += [float(t[p]) for p in pk if sel[p]]
        self.steps = self.steps[-400:]
        self.done_until = end


def stream_count(net, data):
    """Count a whole recording exactly the way the live counter would."""
    c = StreamCounter(net, from_start=True)
    t0, t1 = data[0, 0], data[-1, 0]
    for t_now in np.arange(t0 + 2.0, t1, HOP_S):
        c.update(data[data[:, 0] <= t_now])
    c.update(data, flush=True)
    return c.count


# ── classical baselines (what the net has to beat) ───────────────────────────

def baseline_accel_peaks(X):
    """Textbook pedometer: count |a| peaks above 1.25 g at least 0.3 s apart."""
    amag = np.convolve(np.linalg.norm(X[:, 0:3], axis=1) / G, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(amag, height=1.25, distance=int(0.3 * FS))
    return len(peaks)


def baseline_gyro_strides(X):
    """2 x the swings of the strapped leg (the same gyro detector the labeler uses)."""
    return 2 * len(strapped_strikes(X))


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
