"""Live activity recognition with both models side by side.

    python live.py

Every half second it classifies the last 3 s from the ankle wearable with
model 1 (peak in one direction) and model 2 (all IMU outputs), prints both,
and sends them to the board, which shows them on the LED matrix as two
letters: left = model 1, right = model 2 (S/W/R/B).

Run train.py first.
"""

import argparse
import sys
import time

import gaitlib

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--smooth", type=int, default=3, help="majority vote over this many updates (1 = off)")
args = ap.parse_args()

clf = gaitlib.LiveClassifier(args.smooth)
if not clf.ready:
    sys.exit("No trained models -- record each activity (record.py or app.py), then run train.py.")

stream = gaitlib.IMUStream(args.host).connect()
print(f"Model 1 = peak of {clf.peak['direction']}   Model 2 = all IMU outputs ({clf.full.get('feature_set')})."
      "  Ctrl+C to stop.")
print("Waiting for the board...")

try:
    while True:
        time.sleep(clf.TICK_S)
        r = clf.update(stream.snapshot())
        if r is None:
            if clf._tick % 10 == 0:
                print("  ...waiting for 3 s of data" if stream.connected else "  ...no data from the board")
                stream.send_prediction(None, None)
            continue
        if not r["fresh"]:
            continue
        stream.send_prediction(r["model1"], r["model2"])
        agree = "" if r["model1"] == r["model2"] else "   (disagree)"
        print(f"  peak {r['direction']}={r['peak']:5.2f} g   MODEL 1: {r['model1']:<9}  MODEL 2: {r['model2']:<9}"
              f"{'  still' if r['still'] else ''}{agree}")
except KeyboardInterrupt:
    pass
finally:
    stream.send_prediction(None, None)
    time.sleep(0.2)
    stream.close()
