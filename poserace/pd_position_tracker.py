"""
PD position tracker: turn the Single Motor by hand and watch the Double
Motor drive to match its angle. Kp and Kd sliders let you see, live, how
proportional and derivative gain change the response -- rise time,
overshoot, oscillation, settling time.

Install first:
    pip install legoeducation matplotlib
Then copy lelib.py from the "useful libraries" folder into this project's
folder (already done here).

Controller: at every loop tick,
    error   = target_position - actual_position
    speed   = Kp * error - Kd * d(actual_position)/dt
speed (clamped to +-MAX_SPEED) is the only thing being commanded -- there is
no separate position-move command, so all of the tracking behavior you see
comes from the P and D terms alone. The D term damps the follower's own
velocity (not the error's), so it damps overshoot instead of reacting to
how fast the dial is being turned by hand -- see the comment in
control_loop() for why.
"""

import threading
import time
from collections import deque

import legoeducation as le
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from lelib import singleMotor, doubleMotor

# --- Bluetooth card info for your hardware ---------------------------------
# Both devices pair with the same LEGO connection card.
# Valid colors: le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE,
# _MAGENTA, _AZURE, _ORANGE.
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0994"

# Which side of the Double Motor tracks the target. The other side is left
# unpowered -- this is a one-degree-of-freedom demo, matching the Single
# Motor's one shaft.
DOUBLE_MOTOR_SIDE = le.MOTOR_LEFT

MAX_SPEED = 100          # speed cap sent to the Double Motor, in percent
LOOP_HZ = 50             # control loop rate -- as fast as the BLE link can
                          # sustain while still giving a clean d(error)/dt
HISTORY_SECONDS = 15     # how much history the plot keeps on screen

KP_MAX = 3.0
KD_MAX = 1.0
KP_INIT = 1.0
KD_INIT = 0.0

# --- Shared state between the control thread and the plot thread -----------
# Everything below is read/written from both threads under `lock`.

lock = threading.Lock()
kp = KP_INIT
kd = KD_INIT
running = True

t_hist = deque()
target_hist = deque()
actual_hist = deque()


def control_loop(sm, dm):
    """Runs in a background thread: reads both motors' positions, computes
    the PD speed command, and sends it -- as fast as the loop allows."""
    dt_target = 1.0 / LOOP_HZ
    prev_actual = None
    prev_time = time.monotonic()
    start_time = prev_time
    tick = 0

    while running:
        now = time.monotonic()
        dt = now - prev_time
        if dt <= 0:
            continue

        target = float(sm.motor.position)
        actual = float(dm.motor[DOUBLE_MOTOR_SIDE].position)
        error = target - actual
        # Derivative on measurement (actual's own velocity), not on error.
        # Differentiating the error would include d(target)/dt, so every
        # time the dial gets turned by hand the D term would spike in the
        # direction of that motion -- "derivative kick" -- which pushes the
        # follower harder instead of damping it. Damping should only react
        # to how fast the follower itself is moving.
        d_actual = 0.0 if prev_actual is None else (actual - prev_actual) / dt

        with lock:
            speed = kp * error - kd * d_actual

        speed = max(-MAX_SPEED, min(MAX_SPEED, speed))
        # motor_run()'s speed accepts a signed value directly -- confirmed on
        # hardware that a negative speed simply reverses whichever direction
        # is passed (direction is left at its CLOCKWISE default here), so
        # there's no need to compute a direction and pass abs(speed).
        dm.motor_run(motor=DOUBLE_MOTOR_SIDE, speed=int(round(speed)), blocking=False)

        t = now - start_time
        with lock:
            t_hist.append(t)
            target_hist.append(target)
            actual_hist.append(actual)
            while t_hist and t - t_hist[0] > HISTORY_SECONDS:
                t_hist.popleft()
                target_hist.popleft()
                actual_hist.popleft()

        prev_actual = actual
        prev_time = now
        tick += 1
        if tick % (LOOP_HZ * 2) == 0:
            print(f"loop rate: {1.0 / dt:5.1f} Hz   error: {error:7.1f} deg   speed: {speed:6.1f}%")

        sleep_left = dt_target - (time.monotonic() - now)
        if sleep_left > 0:
            time.sleep(sleep_left)

    dm.motor_stop(motor=DOUBLE_MOTOR_SIDE)


def main():
    sm = singleMotor()
    print("Connecting to Single Motor...")
    sm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)

    dm = doubleMotor()
    print("Connecting to Double Motor...")
    dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)

    print("Connected. Zeroing both positions...")
    sm.motor_reset_relative_position()
    dm.motor_reset_relative_position(motor=DOUBLE_MOTOR_SIDE, position=0)

    control_thread = threading.Thread(target=control_loop, args=(sm, dm), daemon=True)
    control_thread.start()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(bottom=0.28)
    target_line, = ax.plot([], [], label="Target (Single Motor)", linewidth=2)
    actual_line, = ax.plot([], [], label="Actual (Double Motor)", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (degrees)")
    ax.set_title("PD position tracking")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax_kp = fig.add_axes([0.15, 0.13, 0.7, 0.03])
    ax_kd = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    s_kp = Slider(ax_kp, "Kp", 0.0, KP_MAX, valinit=KP_INIT)
    s_kd = Slider(ax_kd, "Kd", 0.0, KD_MAX, valinit=KD_INIT)

    def on_kp_changed(val):
        global kp
        with lock:
            kp = val

    def on_kd_changed(val):
        global kd
        with lock:
            kd = val

    s_kp.on_changed(on_kp_changed)
    s_kd.on_changed(on_kd_changed)

    def update(_frame):
        with lock:
            t = list(t_hist)
            tgt = list(target_hist)
            act = list(actual_hist)
        if t:
            target_line.set_data(t, tgt)
            actual_line.set_data(t, act)
            ax.set_xlim(max(0.0, t[-1] - HISTORY_SECONDS), max(HISTORY_SECONDS, t[-1]))
            all_vals = tgt + act
            pad = max(5.0, 0.1 * (max(all_vals) - min(all_vals) + 1e-6))
            ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
        return target_line, actual_line

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)

    def on_close(_event):
        global running
        running = False

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        plt.show()
    finally:
        global running
        running = False
        control_thread.join(timeout=2)
        dm.motor_stop(motor=DOUBLE_MOTOR_SIDE)
        sm.disconnect()
        dm.disconnect()


if __name__ == "__main__":
    main()
