"""Record labelled training data from the ankle wearable.

    python record.py walking              # 60 s of walking (default)
    python record.py running --seconds 45
    python record.py biking               # on a bike, or lying on your back air-biking
    python record.py standing
    python record.py walking --session strap2   # after taking it off and putting it back on

Each run saves data/<activity>_<session>_<date-time>.csv. A "session" is one
time the wearable is strapped on: use a new --session name every time you
re-strap it (strap1, strap2, ...). train.py then tests on sessions it never saw,
which is the honest check that it still works after re-strapping. Record each activity several
times (different days, speeds, shoes, a friend wearing it...) -- the more
variety, the better the models do on game day. The first and last couple of
seconds are trimmed off so the "getting started / stopping" bits aren't in it.
"""

import argparse
import sys
import time

import gaitlib

ap = argparse.ArgumentParser()
ap.add_argument("activity", nargs="?", choices=gaitlib.ACTIVITIES, help="asked for if left out")
ap.add_argument("--session", default="strap1", help="name for this strap-on (no underscores)")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--countdown", type=float, default=5.0, help="seconds to get going before recording")
ap.add_argument("--trim", type=float, default=2.0, help="seconds dropped from each end")
ap.add_argument("--host", default="localhost")
args = ap.parse_args()
# Run with no arguments (e.g. VS Code's Run button) -> ask instead.
while args.activity not in gaitlib.ACTIVITIES:
    args.activity = input(f"Activity to record ({' / '.join(gaitlib.ACTIVITIES)}): ").strip().lower()
if len(sys.argv) == 1:
    args.session = input(f"Strap-on session -- change ONLY after re-strapping the sensor (e.g. strap2) "
                         f"[Enter = {args.session}]: ").strip().replace(" ", "") or args.session
if "_" in args.session or not args.session:
    sys.exit("--session can't be empty or contain '_'")

samples = []
recording = False


def on_sample(s):
    if recording:
        samples.append(s)


stream = gaitlib.IMUStream(args.host, on_sample=on_sample).connect()
print("Waiting for the board...", end="", flush=True)
while not stream.connected:
    time.sleep(0.1)
print(" connected.")

for k in range(int(args.countdown), 0, -1):
    print(f"  start {args.activity} -- recording in {k}", flush=True)
    time.sleep(1)
print(f"RECORDING {args.activity} for {args.seconds:.0f} s (Ctrl+C stops early and still saves)")

recording = True
stream.take_rate(1)
t_end = time.monotonic() + args.seconds + 2 * args.trim
try:
    while time.monotonic() < t_end:
        time.sleep(1)
        rate = stream.take_rate(1)
        recent = stream.snapshot()[-50:]
        peak = (abs(recent[:, 1:4]).max() / gaitlib.G) if len(recent) else 0
        warn = "  <-- LOW RATE, check Wi-Fi" if rate < 40 else ""
        print(f"  {t_end - time.monotonic():5.0f} s left  {rate:3.0f} samples/s  peak axis {peak:4.1f} g{warn}")
except KeyboardInterrupt:
    pass
recording = False
stream.close()

try:
    path, n, secs = gaitlib.save_recording(args.activity, args.session, samples, args.trim)
except ValueError as e:
    sys.exit(f"Nothing saved: {e}")
print(f"Saved {n} samples ({secs:.0f} s) to {path}")
offset = gaitlib.gyro_offset(gaitlib.load_csv(path))
if offset > gaitlib.GYRO_OFFSET_LIMIT:
    print(f"WARNING: the gyro averaged {offset:.0f} deg/s on one axis -- it's stuck with an offset. "
          "Unplug the board for a few seconds, then record this one again (and delete this file).")
