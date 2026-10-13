"""Sound effects, synthesized with numpy (no downloads) and played with Windows'
built-in winsound. Files are generated once into pingpong/sounds/.

    sfx = Sounds()          # Sounds(enabled=False) to mute
    sfx.play("hit")

winsound plays one sound at a time (a new sound cuts off the previous one),
which is fine for short game effects.
"""

import os
import wave

import numpy as np

try:
    import winsound
except ImportError:  # not on Windows: sounds are silently skipped
    winsound = None

RATE = 22050
SOUND_DIR = os.path.join(os.path.dirname(__file__), "sounds")


def _tone(freq, dur, vol=0.5, decay=8.0, shape="sine"):
    t = np.arange(int(RATE * dur)) / RATE
    if np.ndim(freq) == 0:
        phase = 2 * np.pi * freq * t
    else:  # frequency sweep
        phase = 2 * np.pi * np.cumsum(np.linspace(freq[0], freq[1], len(t))) / RATE
    w = np.sin(phase) if shape == "sine" else np.sign(np.sin(phase)) * 0.6
    return vol * w * np.exp(-decay * t)


def _noise(dur, vol=0.3, decay=30.0):
    t = np.arange(int(RATE * dur)) / RATE
    return vol * np.random.uniform(-1, 1, len(t)) * np.exp(-decay * t)


def _mix(*parts):
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:len(p)] += p
    return out


def _seq(*parts, gap=0.0):
    pad = np.zeros(int(RATE * gap))
    out = []
    for p in parts:
        out += [p, pad]
    return np.concatenate(out)


def _notes(freqs, dur, vol=0.35, decay=4.0, shape="sine"):
    return _seq(*[_tone(f, dur, vol, decay, shape) for f in freqs])


def _make_all():
    np.random.seed(1)
    return {
        # paddle "tock": short pitched click + a puff of noise
        "hit": _mix(_tone(1250, 0.09, 0.7, 45), _tone(620, 0.09, 0.4, 35), _noise(0.03, 0.35, 120)),
        "robot_hit": _mix(_tone(1050, 0.08, 0.5, 45), _tone(520, 0.08, 0.3, 35), _noise(0.025, 0.3, 120)),
        "bounce": _mix(_tone(1700, 0.05, 0.4, 70), _noise(0.02, 0.2, 150)),
        "smash": _mix(_tone(900, 0.2, 0.8, 18), _tone((400, 120), 0.25, 0.5, 10), _noise(0.12, 0.6, 25)),
        "miss": _tone((420, 140), 0.45, 0.5, 3, "square"),
        "point": _notes([880, 1320], 0.12, 0.4, 10),
        "golden": _notes([1047, 1319, 1568, 2093], 0.07, 0.35, 12),
        "golden_point": _notes([784, 988, 1175, 1568, 2093], 0.09, 0.4, 8),
        "beep": _tone(660, 0.12, 0.4, 12),
        "go": _tone(990, 0.3, 0.45, 6),
        "select": _notes([660, 880], 0.08, 0.35, 12),
        "win": _notes([523, 659, 784, 1047, 784, 1047], 0.13, 0.4, 5),
        "lose": _seq(_tone((392, 370), 0.3, 0.4, 2, "square"), _tone((370, 349), 0.3, 0.4, 2, "square"),
                     _tone((349, 260), 0.7, 0.4, 2, "square")),
    }


def _write(path, samples):
    data = (np.clip(samples, -1, 1) * 32767 * 0.8).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(data.tobytes())


class Sounds:
    def __init__(self, enabled=True):
        self.enabled = enabled and winsound is not None
        self.paths = {}
        if not self.enabled:
            return
        os.makedirs(SOUND_DIR, exist_ok=True)
        for name, samples in _make_all().items():
            path = os.path.join(SOUND_DIR, f"{name}.wav")
            if not os.path.exists(path):
                _write(path, samples)
            self.paths[name] = path

    def play(self, name):
        if not self.enabled or name not in self.paths:
            return
        try:
            winsound.PlaySound(self.paths[name], winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError:
            pass
