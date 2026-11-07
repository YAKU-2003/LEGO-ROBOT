"""Train the step-counting neural net and measure how good it is.

    python train_steps.py               # cross-validate + train the final net
    python train_steps.py --iters 3000  # train longer

1. Loads every recording with a hand-counted step total (data/), plus the gait
   identifier's standing / biking recordings as 0-step examples.
2. Places the counted steps in time (steplib.label_events) where the gyro agrees
   with your count.
3. Uncertainty analysis: leave-one-recording-out. For each recording, a fresh net
   is trained on all the others and then counts the held-out one exactly like
   the live counter would. Compared with two classical pedometers.
4. Trains the final net on everything -> models/stepnet.pt, plus report.txt,
   plots and training_log.json (losses + gradients, for animate.py).
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import publicdata  # noqa: E402
import steplib as sl  # noqa: E402
import trainlib as tl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=None, help="training steps per net (1500; 800 with --transfer)")
ap.add_argument("--transfer", action="store_true",
                help="also use the public gait data: start from models/pretrained.pt and mix the public walks in")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--no-cv", action="store_true", help="skip the uncertainty analysis (fast)")
args = ap.parse_args()

DEV = tl.DEV
PRETRAINED_PATH = os.path.join(sl.MODEL_DIR, "pretrained.pt")
INIT = None
if args.transfer and os.path.exists(PRETRAINED_PATH):
    INIT = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)["state_dict"]
ITERS = args.iters or (800 if INIT is not None else 1500)
os.makedirs(sl.MODEL_DIR, exist_ok=True)
report_lines = []


def say(line=""):
    print(line, flush=True)
    report_lines.append(line)


# ── data ─────────────────────────────────────────────────────────────────────

recs = []
for r in sl.recordings():
    data = sl.gaitlib.load_csv(r["path"])
    t, X, ok = sl.to_grid(data)
    events, info = sl.label_events(X, r["steps"])
    r.update(data=data, X=X, ok=ok, inp=sl.net_inputs(X), events=events, label=info,
             target=None if events is None else sl.density_target(len(X), events), minutes=len(X) / sl.FS / 60)
    recs.append(r)

stepped = [r for r in recs if r["steps"] > 0]
say(f"{len(recs)} recordings ({sum(r['minutes'] for r in recs):.1f} min): "
    f"{len(stepped)} with steps ({sum(r['steps'] for r in stepped)} counted steps), "
    f"{len(recs) - len(stepped)} with 0 steps")
for r in recs:
    lab = r["label"]
    note = "" if r["steps"] == 0 else (
        f"  gyro places {lab['auto']} steps -> " + ("labels OK" if lab["matched"] else "doesn't match, count-only"))
    say(f"  {r['name'][:52]:52} {r['kind']:>8} {r['steps']:4d} steps{note}")
# The public walks (exact, hand-checked steps) stay in every training set while fine-tuning, so the
# net doesn't forget them. They're never scored here: every number below is on your sensor.
public = []
if args.transfer and os.path.isdir(publicdata.IPOL_DIR):
    public = publicdata.ipol_recordings()
    own = sum(len(r["X"]) for r in recs)
    for r in public:  # together the public walks get about as many training crops as your recordings
        r["weight"] = own / sum(len(q["X"]) for q in public)
    say(f"  + {len(public) // 2} public walks ({sum(r['steps'] for r in public) // 2} hand-checked steps) mixed "
        "into training, never scored")
# Your counted walks are required: fine-tuning on only 0-step recordings from your sensor teaches the net
# the shortcut "this sensor never steps" (tried it: walks counted as ~5 steps instead of ~95).
if not stepped:
    raise SystemExit("\nNo recordings with steps yet -- record some walks (counting your steps) first.")
if not any(r["target"] is not None for r in stepped + public):
    raise SystemExit("\nThe gyro couldn't place the steps in any recording -- check the sensor is on the ankle.")
say(f"Training on {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))
say("Starting from the net pre-trained on public gait data (models/pretrained.pt), fine-tuning "
    f"{ITERS} steps" if INIT is not None else f"Training on your recordings only, from scratch, {ITERS} steps")
say()


# ── training (the loop itself is in trainlib.py) ─────────────────────────────

def train_net(train_recs, seed=0, log=None, example=None):
    """Fine-tune the pre-trained net (or train from scratch with --scratch)."""
    return tl.train_net(train_recs + public, iters=ITERS, batch=args.batch, seed=seed, log=log, example=example,
                        init=INIT, max_lr=1e-3 if INIT is not None else 3e-3)


# ── uncertainty analysis: leave one recording out ────────────────────────────

summarize, bootstrap_ci = tl.summarize, tl.bootstrap_ci
METHODS = {"net": "Neural net (PyTorch)", "accel": "|a| peak counter", "gyro": "2 x gyro swings"}
COLORS = {"net": "#6366f1", "accel": "#f59e0b", "gyro": "#10b981", "pre": "#94a3b8"}
zero_shot = None
if INIT is not None:  # the pre-trained net before it has seen any of your data: shows what fine-tuning adds
    zero_shot = sl.build_net()
    zero_shot.load_state_dict(INIT)
    zero_shot.eval()
    METHODS["pre"] = "Pre-trained only"
metrics = {}
rows = []
if args.no_cv or len(stepped) < 2:
    say("Skipping the uncertainty analysis" + ("" if args.no_cv else " (needs at least 2 recordings with steps)"))
else:
    say("=" * 74)
    say("UNCERTAINTY ANALYSIS -- leave-one-recording-out")
    say("Each recording is counted by a net that never saw it, the same way as live.")
    say("=" * 74)
    t0 = time.time()
    for i, r in enumerate(recs):
        net = train_net([q for q in recs if q is not r], seed=i)
        rows.append({"name": r["name"], "kind": r["kind"], "true": r["steps"], "minutes": r["minutes"],
                     "net": sl.stream_count(net, r["data"]), "accel": sl.baseline_accel_peaks(r["X"]),
                     "gyro": sl.baseline_gyro_strides(r["X"])})
        if zero_shot is not None:
            rows[-1]["pre"] = sl.stream_count(zero_shot, r["data"])
        w = rows[-1]
        say(f"  {i + 1:2d}/{len(recs)} {r['name'][:44]:44} true {w['true']:4d}  net {w['net']:6.1f}  "
            f"|a| peaks {w['accel']:4d}  gyro {w['gyro']:4d}   ({time.time() - t0:4.0f} s)")
    say()
    walk = [w for w in rows if w["true"] > 0]
    zero = [w for w in rows if w["true"] == 0]
    say(f"Recordings with steps ({len(walk)}, {sum(w['true'] for w in walk)} steps):")
    say(f"  {'method':22} {'accuracy':>9} {'mean |err|':>11} {'mean |%err|':>12} {'bias':>7} {'spread (sd)':>12}")
    for m, label in METHODS.items():
        s = summarize([w[m] - w["true"] for w in walk], [w["true"] for w in walk])
        if m == "net" and len(walk) > 2:
            s["mape_ci"] = bootstrap_ci([w[m] - w["true"] for w in walk], [w["true"] for w in walk])
        if zero:
            mins = sum(w["minutes"] for w in zero)
            s["false_per_min"] = float(sum(w[m] for w in zero) / mins)
        metrics[m] = s
        say(f"  {label:22} {s['accuracy']:8.1f}% {s['mae']:8.1f} st {s['mape']:10.1f}% {s['bias']:+6.1f}% "
            f"{s['sd']:10.1f}%")
    if zero:
        say(f"0-step recordings ({len(zero)}, {sum(w['minutes'] for w in zero):.1f} min standing / fidgeting / "
            f"biking) -- false steps per minute:")
        say("  " + "   ".join(f"{METHODS[m]}: {metrics[m]['false_per_min']:.1f}" for m in METHODS))
    n = metrics["net"]
    say()
    say(f"SUMMARY  net: {n['accuracy']:.1f}% of steps right; a count of N steps is N x (1 {n['bias'] / 100:+.3f}) "
        f"+/- {1.96 * n['sd']:.1f}% (95%)" + (f"; mean error {n['mape']:.1f}% (95% CI {n['mape_ci'][0]:.1f}-"
                                              f"{n['mape_ci'][1]:.1f}%)" if "mape_ci" in n else ""))
    say()

    # plots
    walk_k = sorted({w["kind"] for w in walk})
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    hi = max(max(w[m] for m in ["true"] + list(METHODS)) for w in walk) * 1.08
    ax[0].plot([0, hi], [0, hi], "k--", lw=1, label="perfect")
    for m, c in [(m, COLORS[m]) for m in METHODS if m != "net"] + [("net", COLORS["net"])]:
        ax[0].scatter([w["true"] for w in walk], [w[m] for w in walk], s=45 if m == "net" else 25, c=c,
                      label=METHODS[m], alpha=0.9 if m == "net" else 0.6, zorder=3 if m == "net" else 2)
    ax[0].set(xlabel="steps you counted", ylabel="steps counted by the method", title="Held-out recordings")
    ax[0].legend()
    true = np.array([w["true"] for w in walk], float)
    rel = (np.array([w["net"] for w in walk]) - true) / true * 100
    ax[1].scatter(true, rel, c="#6366f1", s=45, zorder=3)
    for k in walk_k:
        sel = [j for j, w in enumerate(walk) if w["kind"] == k]
        for j in sel:
            ax[1].annotate(k, (true[j], rel[j]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax[1].axhline(n["bias"], color="#6366f1", lw=1.5, label=f"bias {n['bias']:+.1f}%")
    for sgn in (-1, 1):
        ax[1].axhline(n["bias"] + sgn * 1.96 * n["sd"], color="#6366f1", ls="--", lw=1)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="steps you counted", ylabel="net error (%)",
              title=f"Bland-Altman: 95% of counts within {n['bias']:+.1f} +/- {1.96 * n['sd']:.1f}%")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(sl.MODEL_DIR, "uncertainty.png"), dpi=130)
    plt.close(fig)

    if zero:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = np.arange(len(METHODS))
        ax.bar(xs, [metrics[m]["false_per_min"] for m in METHODS], color=[COLORS[m] for m in METHODS])
        ax.set_xticks(xs, list(METHODS.values()))
        ax.set(ylabel="false steps per minute", title="Steps counted while NOT walking (standing, fidgeting, biking)")
        fig.tight_layout()
        fig.savefig(os.path.join(sl.MODEL_DIR, "false_steps.png"), dpi=130)
        plt.close(fig)


# ── final net on everything ──────────────────────────────────────────────────

# A 10 s stretch of the longest placed walk: plotted below, and tracked during training for animate.py.
ex = max((r for r in recs if r["target"] is not None and r["steps"] > 0), key=lambda r: r["steps"],
         default=None) or max(public, key=lambda r: r["steps"])
s0 = max(0, int(np.argmax(ex["target"] > 0.01)) - int(sl.FS))
sl_ = slice(s0, min(len(ex["X"]), s0 + int(10 * sl.FS)))

say("Training the final net on all recordings...")
log = {"iter": [], "loss": [], "shape_loss": [], "count_loss": [], "grad_norm": [], "snapshots": [],
       "example": {"name": ex["name"], "inputs": ex["inp"][sl_].round(4).tolist(),
                   "target": ex["target"][sl_].round(5).tolist()}}
t0 = time.time()
net = train_net(recs, seed=1234, log=log, example=(ex["inp"][sl_], ex["target"][sl_]))
say(f"  {ITERS} steps in {time.time() - t0:.0f} s, final loss {log['loss'][-1]:.4f}")
log["layers"] = [n for n, p in net.named_parameters() if n.endswith("weight") and p.dim() == 3]
log["params"] = int(sum(p.numel() for p in net.parameters()))
say(f"  {log['params']} weights")

torch.save({"state_dict": net.state_dict(), "metrics": metrics, "trained_at": time.time(),
            "recordings": len(recs), "steps": sum(r["steps"] for r in recs),
            "pretrained": INIT is not None}, sl.MODEL_PATH)
sl.save_json(os.path.join(sl.MODEL_DIR, "training_log.json"), log)
sl.save_json(os.path.join(sl.MODEL_DIR, "cv_results.json"), rows)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(log["iter"], log["loss"], label="total loss", color="#6366f1")
ax.semilogy(log["iter"], log["shape_loss"], label="shape (density vs labelled bumps)", color="#10b981", lw=1)
ax.semilogy(log["iter"], log["count_loss"], label="count (sum vs counted steps)", color="#f59e0b", lw=1)
ax.set(xlabel="training step", ylabel="loss (log scale)", title="Training the final net")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(sl.MODEL_DIR, "training_curve.png"), dpi=130)
plt.close(fig)

# The example stretch through the final net: inputs, density, labelled steps.
d = sl.density(net, ex["inp"])
tt = np.arange(len(ex["X"]))[sl_] / sl.FS
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for i, c in zip(range(3), ("#ef4444", "#10b981", "#3b82f6")):
    ax[0].plot(tt, ex["inp"][sl_, i], c, lw=1, label=sl.CHANNEL_NAMES[i])
ax[0].set(ylabel="accel (g)")
ax[0].legend(loc="upper right", ncol=3)
for i, c in zip(range(3, 6), ("#ef4444", "#10b981", "#3b82f6")):
    ax[1].plot(tt, ex["inp"][sl_, i] * 4, c, lw=1, label=sl.CHANNEL_NAMES[i])
ax[1].set(ylabel="gyro (rad/s)")
ax[1].legend(loc="upper right", ncol=3)
ax[2].fill_between(tt, d[sl_], color="#6366f1", alpha=0.35, label="net's step density")
ax[2].plot(tt, ex["target"][sl_], "k", lw=0.8, label="labelled steps")
ax[2].set(ylabel="steps / sample", xlabel="time (s)",
          title=f"{ex['name']}: {d[sl_].sum():.1f} steps in this 10 s (labelled: {ex['target'][sl_].sum():.1f})")
ax[2].legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(sl.MODEL_DIR, "example_density.png"), dpi=130)
plt.close(fig)

say("Saved models/stepnet.pt, report.txt, training_log.json and plots")
with open(os.path.join(sl.MODEL_DIR, "report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
