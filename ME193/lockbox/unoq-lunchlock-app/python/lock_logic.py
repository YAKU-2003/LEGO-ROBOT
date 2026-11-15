"""
The lock policy, with no camera or window code, so the same logic runs in
lock_pc.py (OpenCV window), app.py (web dashboard) and later on the UNO Q.

Feed it one CNN guess per frame with  engine.update(k, conf, now)  and press the
button with  engine.press_button(now).  Read engine.state / engine.reader.digits etc.
"""

import json
import time
from pathlib import Path

# ------------------------------- lock policy -------------------------------
CONF_MIN = 0.85        # CNN must be at least this sure about a frame
HOLD_S = 0.7           # ...and say the same digit for this long before it is accepted
RELEASE_S = 0.3        # fist / unsure for this long "releases" the last digit (lets you repeat one)
ENTRY_TIMEOUT_S = 20   # time allowed to enter the whole code after pressing the button
OPEN_S = 6             # how long the safe stays unlocked
WRONG_S = 2            # how long "wrong code" shows
MAX_TRIES = 3          # wrong codes in a row before a lock-out
LOCKOUT_S = 30
DEFAULT_CODE = (3, 1, 4)
# ---------------------------------------------------------------------------

CONFIG_FILE = Path(__file__).with_name("lock_config.json")


def load_code():
    try:
        code = tuple(int(c) for c in json.loads(CONFIG_FILE.read_text())["code"])
        if len(code) == 3 and all(1 <= c <= 5 for c in code):
            return code
    except Exception:
        pass
    return DEFAULT_CODE


def save_code(code):
    CONFIG_FILE.write_text(json.dumps({"code": list(code)}))


class CodeReader:
    """
    Turns a noisy stream of per-frame CNN guesses into clean digit entries.

    A digit is ENTERED when the CNN reports the same count (1-5), with confidence >= CONF_MIN,
    continuously for HOLD_S seconds.  After that the reader is "latched" on that digit and will
    not enter it again until either a DIFFERENT digit is held, or the hand is released
    (fist / no confident guess for RELEASE_S) - that is how you enter a repeated digit.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.digits = []
        self.cand, self.cand_since = None, 0.0      # digit currently being held
        self.latched = None                         # last digit entered
        self.release_since = None
        self.progress = 0.0

    def update(self, k, conf, now):
        """Feed one frame. Returns the digit if one was just entered, else None."""
        self.progress = 0.0
        steady = conf >= CONF_MIN and 1 <= k <= 5
        if not steady:                               # fist (0) or unsure -> counts as releasing
            self.cand = None
            if self.release_since is None:
                self.release_since = now
            if now - self.release_since >= RELEASE_S:
                self.latched = None
            return None
        self.release_since = None

        if k == self.latched:                        # still showing the digit we just entered
            return None
        if k != self.cand:                           # new candidate - start its hold timer
            self.cand, self.cand_since = k, now
        self.progress = min(1.0, (now - self.cand_since) / HOLD_S)
        if self.progress >= 1.0:
            self.digits.append(k)
            self.latched, self.cand = k, None
            return k
        return None


class LockEngine:
    """
    States:  IDLE -> (button) -> ENTER -> 3 digits -> OPEN | WRONG | LOCKOUT -> IDLE
                                    \\-> timeout -> IDLE
    Hardware hooks (on_leds, on_unlock, on_lock) are plain callbacks so the UNO Q can
    plug its real LEDs and latch in.
    """

    def __init__(self, code=None, on_leds=None, on_unlock=None, on_lock=None, on_event=None):
        self.code = tuple(code) if code else load_code()
        self.reader = CodeReader()
        self.state, self.t_state = "IDLE", time.time()
        self.wrong = 0
        self.on_leds = on_leds or (lambda on: None)
        self.on_unlock = on_unlock or (lambda: None)
        self.on_lock = on_lock or (lambda: None)
        self.on_event = on_event or (lambda kind, text: None)

    def _go(self, state, now):
        self.state, self.t_state = state, now

    def time_left(self, now):
        limit = {"ENTER": ENTRY_TIMEOUT_S, "OPEN": OPEN_S, "WRONG": WRONG_S, "LOCKOUT": LOCKOUT_S}.get(self.state)
        return None if limit is None else max(0.0, limit - (now - self.t_state))

    def press_button(self, now=None):
        now = now or time.time()
        if self.state != "IDLE":
            return False
        self.reader.reset()
        self.on_leds(True)
        self._go("ENTER", now)
        self.on_event("button", "Button pressed - LEDs on, show your code")
        return True

    def cancel(self, now=None):
        if self.state == "ENTER":
            self.on_leds(False)
            self._go("IDLE", now or time.time())

    def update(self, k, conf, now=None):
        now = now or time.time()
        left = self.time_left(now)

        if self.state == "ENTER":
            d = self.reader.update(k, conf, now)
            if d is not None:
                self.on_event("digit", f"Digit {len(self.reader.digits)} entered")
            if len(self.reader.digits) == 3:
                self.on_leds(False)
                if tuple(self.reader.digits) == self.code:
                    self.wrong = 0
                    self.on_unlock()
                    self.on_event("open", "Correct code - UNLOCKED")
                    self._go("OPEN", now)
                else:
                    self.wrong += 1
                    self.on_event("wrong", f"Wrong code ({self.wrong}/{MAX_TRIES})")
                    self._go("LOCKOUT" if self.wrong >= MAX_TRIES else "WRONG", now)
                    if self.state == "LOCKOUT":
                        self.on_event("lockout", f"Too many tries - locked out for {LOCKOUT_S} s")
            elif left <= 0:
                self.on_leds(False)
                self.on_event("timeout", "Too slow - timed out")
                self._go("IDLE", now)

        elif self.state == "OPEN" and left <= 0:
            self.on_lock()
            self.on_event("lock", "Re-locked")
            self._go("IDLE", now)
        elif self.state == "WRONG" and left <= 0:
            self._go("IDLE", now)
        elif self.state == "LOCKOUT" and left <= 0:
            self.wrong = 0
            self._go("IDLE", now)
