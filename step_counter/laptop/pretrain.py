"""Pre-train StepNet on public gait data (transfer learning, step 1 of 2).

    python pretrain.py               # validate on held-out people, then pre-train -> models/pretrained.pt
    python pretrain.py --no-val      # skip the validation (fast)

The public data (publicdata.py, IPOL 2023: 19 people, 110 walks, 3866 hand-checked steps)
teaches the net what a step looks like from thousands of exactly-timed footfalls.
train_steps.py then starts from these weights and fine-tunes them on YOUR counted walks,
and measures accuracy only on your sensor.

Validation: the 19 people are split into 4 groups; each group is counted by a net that
never saw those people. That says how well the net generalises to new walkers -- before
it has seen a single recording from our ankle sensor.
"""

import argparse
import os
import time

import numpy as np
import torch

import publicdata
import steplib as sl
import trainlib as tl

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--no-val", action="store_true")
args = ap.parse_args()

PRETRAINED_PATH = os.path.join(sl.MODEL_DIR, "pretrained.pt")
os.makedirs(sl.MODEL_DIR, exist_ok=True)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


if not os.path.isdir(publicdata.IPOL_DIR):
    raise SystemExit(f"Public data not found in {publicdata.IPOL_DIR}\n"
                     "Download https://www.ipol.im/pub/art/2023/497/497-main.zip and unzip it into "
                     "public_data/ipol2023/")

recs = publicdata.ipol_recordings()
subjects = sorted({r["subject"] for r in recs})
say(f"Public data: {len(recs)} foot-sensor recordings, {len(recs) // 2} walks, {len(subjects)} people, "
    f"{sum(r['steps'] for r in recs) // 2} hand-checked steps, {sum(r['minutes'] for r in recs):.0f} min")
say(f"  {publicdata.CITATION}")
say(f"Training on {tl.DEV}" + (f" ({torch.cuda.get_device_name(0)})" if tl.DEV == "cuda" else ""))
say()

# How good is our own gyro step-labeler (used to time your counted steps)? The public data has the truth.
hits = total = n_auto = 0
for r in recs:
    ev, info = sl.label_events(r["X"], r["steps"])
    n_auto += abs(info.get("auto", 0) - r["steps"])
    if ev is not None:
        d = np.abs(ev[:, None] - r["events"][None, :]).min(1)
        hits += int((d <= 0.15 * sl.FS).sum())
        total += len(ev)
say(f"Our gyro labeler vs the hand-checked steps: off by {n_auto / len(recs):.1f} steps per walk on average; "
    f"{100 * hits / max(1, total):.0f}% of the steps it places are within 0.15 s of a real heel strike")
say()

metrics = {}
if not args.no_val:
    say("VALIDATION -- 4 groups of people; each group counted by a net that never saw them")
    rows = []
    t0 = time.time()
    for k in range(4):
        test_s = subjects[k::4]
        net = tl.train_net([r for r in recs if r["subject"] not in test_s], iters=args.iters, batch=args.batch, seed=k)
        for r in recs:
            if r["subject"] in test_s:
                rows.append({"true": r["steps"], "net": sl.stream_count(net, r["data"]),
                             "accel": sl.baseline_accel_peaks(r["X"]), "gyro": sl.baseline_gyro_strides(r["X"])})
        say(f"  group {k + 1}/4 (people {test_s}) done  ({time.time() - t0:.0f} s)")
    for m, label in (("net", "Neural net"), ("accel", "|a| peak counter"), ("gyro", "2 x gyro swings")):
        s = tl.summarize([w[m] - w["true"] for w in rows], [w["true"] for w in rows])
        if m == "net":
            s["mape_ci"] = tl.bootstrap_ci([w[m] - w["true"] for w in rows], [w["true"] for w in rows])
        metrics[m] = s
        say(f"  {label:18} accuracy {s['accuracy']:5.1f}%   mean |err| {s['mae']:4.1f} steps of "
            f"{np.mean([w['true'] for w in rows]):.0f}   bias {s['bias']:+5.1f}%   sd {s['sd']:4.1f}%")
    say()

say(f"Pre-training on all {len(recs)} recordings ({args.iters} steps)...")
t0 = time.time()
net = tl.train_net(recs, iters=args.iters, batch=args.batch, seed=1234)
say(f"  done in {time.time() - t0:.0f} s")

# A first look at our own sensor, before any fine-tuning. No hand counts exist for these, so this
# is only a sanity check against the two classical counters -- not an accuracy number.
walks = [(p, lab) for p, lab in sl.gaitlib.recordings() if lab in ("walking", "running", "standing", "biking")]
if walks:
    say()
    say("Our ankle sensor, zero-shot (gait identifier recordings, NOT hand-counted -- sanity check only):")
    for p, lab in walks:
        X = sl.to_grid(sl.gaitlib.load_csv(p))[1]
        say(f"  {os.path.basename(p)[:44]:44} net {sl.stream_count(net, sl.gaitlib.load_csv(p)):6.1f}   "
            f"|a| peaks {sl.baseline_accel_peaks(X):4d}   gyro {sl.baseline_gyro_strides(X):4d}")

torch.save({"state_dict": net.state_dict(), "validation": metrics, "source": publicdata.CITATION,
            "trained_at": time.time(), "iters": args.iters}, PRETRAINED_PATH)
with open(os.path.join(sl.MODEL_DIR, "pretrain_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
say()
say(f"Saved {PRETRAINED_PATH} -- train_steps.py fine-tunes from it automatically")
