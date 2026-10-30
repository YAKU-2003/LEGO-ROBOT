"""Train and test both gait models on everything in data/.

    python train.py                 # model 1 picks its best axis direction itself
    python train.py --axis +ay      # force model 1 to use the peak of +ay

Model 1 (peak):  one number per window -- the peak acceleration in ONE direction
                 on ONE axis -- into a tiny decision tree (at most 3 thresholds).
Model 2 (full):  features from all six IMU outputs into a random forest. Two
                 feature sets are tried (see gaitlib.py): "tiltfree" (vector
                 lengths, rotation axes, cadence -- doesn't care how the sensor
                 is strapped on) and "peraxis" (per-axis stats + gravity direction).

How accuracy is measured (windows overlap, so testing on random windows would cheat):
  1. Leave-one-recording-out: train on every recording but one, test on that
     one, repeat for each. Needs >= 2 recordings of every activity.
  2. Leave-one-session-out (if recordings are tagged with --session in record.py):
     train on every strap-on but one, test on that one. This is the real test of
     "will it work after I take it off and put it back on?"

Writes models/peak_model.pkl, models/full_model.pkl (used by live.py),
models/report.txt and a few plots for the writeup.
"""

import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

import gaitlib

ap = argparse.ArgumentParser()
ap.add_argument("--axis", choices=gaitlib.PEAK_CANDIDATES, help="model 1's direction (default: pick the best)")
ap.add_argument("--features", choices=list(gaitlib.FEATURE_SETS), help="model 2's feature set (default: pick the best)")
ap.add_argument("--data", default=gaitlib.DATA_DIR)
ap.add_argument("--no-plots", action="store_true")
args = ap.parse_args()

recs = gaitlib.recordings(args.data)
if not recs:
    sys.exit(f"No recordings in {args.data} -- run record.py first.")
os.makedirs(gaitlib.MODEL_DIR, exist_ok=True)
report_lines = []


def say(line=""):
    print(line)
    report_lines.append(line)


def show_results(name, y, pred, labels):
    acc = accuracy_score(y, pred)
    say(f"{name}: {CV_NAME} accuracy {acc * 100:.1f}%")
    cm = confusion_matrix(y, pred, labels=labels)
    say("  confusion matrix (rows = truth, columns = predicted):")
    say("  " + " " * 10 + "".join(f"{l[:8]:>10}" for l in labels))
    for l, row in zip(labels, cm):
        say(f"  {l:>10}" + "".join(f"{v:>10}" for v in row))
    say(classification_report(y, pred, labels=labels, zero_division=0))
    return acc, cm


# ── load ─────────────────────────────────────────────────────────────────────

say(f"{len(recs)} recordings:")
for act in gaitlib.ACTIVITIES:
    n = sum(1 for _, l in recs if l == act)
    say(f"  {act:>9}: {n} recording(s)")
labels = [a for a in gaitlib.ACTIVITIES if any(l == a for _, l in recs)]
if len(labels) < 2:
    sys.exit("Need at least two different activities recorded.")

windows_raw, y, meta = gaitlib.dataset(recs, lambda w: w)
say(f"{len(y)} windows of {gaitlib.WINDOW_S:.0f} s (new one every {gaitlib.STEP_S} s)")

# ── how to test ──────────────────────────────────────────────────────────────

recs_per_class = min(len(set(meta["recording"][y == c])) for c in labels)
if recs_per_class >= 2:
    CV_NAME, CV, CV_GROUPS = "leave-one-recording-out", LeaveOneGroupOut(), meta["recording"]
else:
    chunks_per_class = min(len(set(meta["chunk"][y == c])) for c in labels)
    if chunks_per_class < 2:
        sys.exit("Every activity needs at least ~45 s of data -- record more.")
    say("WARNING: some activity has only 1 recording, so testing holds out 20 s chunks instead of whole "
        "recordings. Record each activity at least twice for an honest score.")
    CV_NAME = "held-out-chunk"
    CV, CV_GROUPS = StratifiedGroupKFold(n_splits=min(5, chunks_per_class), shuffle=True, random_state=0), meta["chunk"]

sessions = sorted(set(meta["session"]))
# A session can only be held out if the other sessions still cover every activity
testable_sessions = [s for s in sessions
                     if set(y[meta["session"] != s]) >= set(y[meta["session"] == s])]
if len(testable_sessions) >= 1 and len(sessions) >= 2:
    say(f"{len(sessions)} strap-on sessions: {', '.join(sessions)} -- will also test on unseen sessions")
else:
    testable_sessions = []
    say("Only one strap-on session -- tag recordings with record.py --session to test re-strapping")
say(f"Testing: {CV_NAME}")
say()


def cv_predict(make_model, X):
    return cross_val_predict(make_model(), X, y, groups=CV_GROUPS, cv=CV)


def session_accuracy(make_model, X, override=None):
    """Accuracy on sessions the model never saw, or None if there's nothing to test.
    override: optional per-window labels that replace the model's answer where set."""
    if not testable_sessions:
        return None
    correct = total = 0
    for s in testable_sessions:
        test = meta["session"] == s
        pred = make_model().fit(X[~test], y[~test]).predict(X[test])
        if override is not None:
            pred = np.where(override[test] != "", override[test], pred)
        correct += np.sum(pred == y[test])
        total += test.sum()
    return correct / total


def fmt(acc):
    return "   n/a" if acc is None else f"{acc * 100:5.1f}%"


# ── model 1: peak in one direction ───────────────────────────────────────────

say("=" * 70)
say("MODEL 1 -- peak acceleration in one direction only")
say("=" * 70)


def peak_tree():
    return DecisionTreeClassifier(max_leaf_nodes=len(labels), min_samples_leaf=5,
                                  class_weight="balanced", random_state=0)


candidates = [args.axis] if args.axis else gaitlib.PEAK_CANDIDATES
scores, sess_scores = {}, {}
say(f"  direction   {CV_NAME:>24}   unseen-session")
for d in candidates:
    X1 = np.array([[gaitlib.peak_feature(w, d)] for w in windows_raw])
    scores[d] = accuracy_score(y, cv_predict(peak_tree, X1))
    sess_scores[d] = session_accuracy(peak_tree, X1)
    say(f"  peak {d}   {fmt(scores[d]):>24}   {fmt(sess_scores[d]):>14}")
# Judge by unseen sessions if we can (the real game-day situation), else by unseen recordings.
direction = max(scores, key=lambda d: sess_scores[d] if testable_sessions else scores[d])
say(f"-> using the peak of {direction}")
say()

X1 = np.array([[gaitlib.peak_feature(w, direction)] for w in windows_raw])
pred1 = cv_predict(peak_tree, X1)
acc1, cm1 = show_results("Model 1", y, pred1, labels)
sess1 = session_accuracy(peak_tree, X1)
model1 = peak_tree().fit(X1, y)

# The tree on one feature is just a few thresholds -- print them as rules.
cuts = sorted(t for t, f in zip(model1.tree_.threshold, model1.tree_.feature) if f >= 0)
edges = [-np.inf] + cuts + [np.inf]
rules = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (hi - 1 if np.isfinite(hi) else lo + 1)
    cls = model1.predict([[mid]])[0]
    if rules and rules[-1][2] == cls:
        rules[-1] = (rules[-1][0], hi, cls)
    else:
        rules.append((lo, hi, cls))
say(f"Model 1 rules (peak of {direction}, in g):")
for lo, hi, cls in rules:
    lo_s = f"{lo:5.2f} <" if np.isfinite(lo) else "       "
    hi_s = f"<= {hi:5.2f}" if np.isfinite(hi) else "        "
    say(f"  {lo_s} peak {hi_s}  ->  {cls}")
missing = [a for a in labels if a not in {r[2] for r in rules}]
if missing:
    say(f"  ({', '.join(missing)} never gets a band -- its peaks overlap another activity's)")
say()
for act in labels:
    v = X1[y == act, 0]
    say(f"  {act:>9}: peak median {np.median(v):5.2f} g  (10-90%: {np.percentile(v, 10):5.2f} - {np.percentile(v, 90):5.2f})")
say()

with open(os.path.join(gaitlib.MODEL_DIR, "peak_model.pkl"), "wb") as f:
    pickle.dump({"direction": direction, "model": model1, "rules": rules,
                 "metrics": {"acc": acc1, "sess": sess1, "cm": cm1.tolist(), "labels": labels, "cv": CV_NAME,
                             "n_windows": len(y), "n_recordings": len(recs),
                             "peaks": {a: [float(v) for v in np.percentile(X1[y == a, 0], [10, 50, 90])]
                                       for a in labels},
                             "directions": {d: [scores[d], sess_scores[d]] for d in scores}}}, f)

# ── model 2: all IMU outputs ─────────────────────────────────────────────────

say("=" * 70)
say("MODEL 2 -- all IMU outputs (accel + gyro), random forest")
say("=" * 70)


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  random_state=0, n_jobs=-1)


# live.py / app.py say "standing" the moment the leg is still (last 0.4 s barely
# moving), before model 2 even runs. Score model 2 the same way so the accuracy
# is the accuracy of what's actually deployed.
_t = np.arange(windows_raw.shape[1]) / gaitlib.FS
still = np.array(["standing" if gaitlib.is_still(np.column_stack([_t, w])) else "" for w in windows_raw])
say(f"  stillness rule fires on: " + ", ".join(f"{a} {np.mean(still[y == a] != ''):.0%}" for a in labels))

tried = {}
say(f"  feature set  {CV_NAME:>24}   unseen-session")
for fs in ([args.features] if args.features else list(gaitlib.FEATURE_SETS)):
    names_fn, extract = gaitlib.FEATURE_SETS[fs]
    X = np.array([extract(w) for w in windows_raw])
    pred = cv_predict(forest, X)
    pred = np.where(still != "", still, pred)
    acc = accuracy_score(y, pred)
    sess = session_accuracy(forest, X, override=still)
    tried[fs] = {"X": X, "pred": pred, "acc": acc, "sess": sess, "names": names_fn()}
    say(f"  {fs:<11}  {fmt(acc):>24}   {fmt(sess):>14}   ({X.shape[1]} features)")

# Judge by unseen sessions if we can (the real game-day situation), else by unseen recordings.
# Tilt-free wins ties (within 1 point): it won't break if the strap angle changes.
key = "sess" if testable_sessions else "acc"
feature_set = max(tried, key=lambda fs: tried[fs][key] + (0.01 if fs == "tiltfree" else 0))
say(f"-> using '{feature_set}' (best on {'unseen sessions' if key == 'sess' else CV_NAME})")
say()

chosen = tried[feature_set]
X2, names = chosen["X"], chosen["names"]
acc2, cm2 = show_results("Model 2 (+ stillness rule)", y, chosen["pred"], labels)
sess2 = chosen["sess"]
model2 = forest().fit(X2, y)

order = np.argsort(model2.feature_importances_)[::-1]
say("Most useful features:")
for i in order[:12]:
    say(f"  {names[i]:>18}  {model2.feature_importances_[i]:.3f}")
say()

with open(os.path.join(gaitlib.MODEL_DIR, "full_model.pkl"), "wb") as f:
    pickle.dump({"model": model2, "feature_set": feature_set, "features": names,
                 "metrics": {"acc": acc2, "sess": sess2, "cm": cm2.tolist(), "labels": labels, "cv": CV_NAME,
                             "feature_sets": {fs: [t["acc"], t["sess"]] for fs, t in tried.items()}}}, f)

say(f"SUMMARY ({CV_NAME}):  model 1 (peak of {direction}) {acc1 * 100:.1f}%   "
    f"model 2 ({feature_set}) {acc2 * 100:.1f}%")
if testable_sessions:
    say(f"SUMMARY (unseen strap-on sessions):  model 1 {fmt(sess1)}   model 2 {fmt(sess2)}")
with open(os.path.join(gaitlib.MODEL_DIR, "report.txt"), "w") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"Saved models + report.txt to {gaitlib.MODEL_DIR}")

# ── plots for the writeup ────────────────────────────────────────────────────

if args.no_plots:
    sys.exit()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"standing": "#4c78a8", "walking": "#59a14f", "running": "#e15759", "biking": "#f28e2b"}

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(X1.min(), X1.max(), 50)
for act in labels:
    ax.hist(X1[y == act, 0], bins=bins, alpha=0.6, label=act, color=colors[act])
for c in cuts:
    ax.axvline(c, color="k", ls="--", lw=1)
ax.set_xlabel(f"peak of {direction} in a {gaitlib.WINDOW_S:.0f} s window (g)")
ax.set_ylabel("windows")
ax.set_title(f"Model 1: one feature, dashed = learned thresholds ({acc1 * 100:.0f}% {CV_NAME})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(gaitlib.MODEL_DIR, "model1_peak_histogram.png"), dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, cm, title in ((axes[0], cm1, f"Model 1: peak {direction} ({acc1 * 100:.0f}%)"),
                      (axes[1], cm2, f"Model 2: {feature_set} ({acc2 * 100:.0f}%)")):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("truth")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.suptitle(CV_NAME)
fig.tight_layout()
fig.savefig(os.path.join(gaitlib.MODEL_DIR, "confusion_matrices.png"), dpi=150)

fig, axes = plt.subplots(len(labels), 1, figsize=(9, 2.0 * len(labels)), sharex=True)
for ax, act in zip(np.atleast_1d(axes), labels):
    w = windows_raw[list(y).index(act)]
    t = np.arange(len(w)) / gaitlib.FS
    for i, ch in enumerate(("ax", "ay", "az")):
        ax.plot(t, w[:, i] / gaitlib.G, lw=1, label=ch)
    ax.set_ylabel(f"{act}\n(g)")
np.atleast_1d(axes)[0].legend(loc="upper right", ncol=3)
np.atleast_1d(axes)[-1].set_xlabel("time (s)")
fig.suptitle("Example window of each activity (ankle accelerometer)")
fig.tight_layout()
fig.savefig(os.path.join(gaitlib.MODEL_DIR, "example_windows.png"), dpi=150)
print("Saved plots: model1_peak_histogram.png, confusion_matrices.png, example_windows.png")
