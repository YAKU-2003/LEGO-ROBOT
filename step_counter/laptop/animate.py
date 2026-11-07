"""Render a video of how the whole step counter works -> models/how_it_works.mp4

    python animate.py            (after train_steps.py)

Everything shown is real: the example walk is one of your recordings, the
feature maps are the trained net's actual activations, the gradient sizes and
the loss curve come from training_log.json, and the "learning" panel replays
the net's output on the example as it was saved every 50 training steps.

Scenes: the system -> the inputs -> forward pass through the conv layers ->
density -> count -> the loss -> back-propagation -> training -> result.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import steplib as sl  # noqa: E402

FPS = 24
OUT = os.path.join(sl.MODEL_DIR, "how_it_works.mp4")
BG, PANEL, TEXT, MUTED = "#0a0c11", "#151923", "#e9ecf2", "#8b93a5"
BLUE, VIOLET, GREEN, AMBER, RED = "#7c9cff", "#a78bfa", "#3ecf8e", "#ffb547", "#ff5c7a"
CH_COLORS = [RED, GREEN, BLUE, RED, GREEN, BLUE, TEXT, AMBER]

try:
    import imageio_ffmpeg

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass  # falls back to an ffmpeg on PATH
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a3040"})

# ── load the real pieces ─────────────────────────────────────────────────────

with open(os.path.join(sl.MODEL_DIR, "training_log.json"), encoding="utf-8") as f:
    LOG = json.load(f)
net, METRICS = sl.load_model()
if net is None:
    raise SystemExit("No trained net -- run train_steps.py first.")
EX_IN = np.array(LOG["example"]["inputs"], np.float32)        # [n, 8]
EX_Y = np.array(LOG["example"]["target"], np.float32)         # [n]
N = len(EX_IN)
T = np.arange(N) / sl.FS

acts = []
hooks = [m.register_forward_hook(lambda m, i, o: acts.append(o[0].detach().numpy()))
         for m in net.body if isinstance(m, torch.nn.ReLU)]
with torch.no_grad():
    EX_D = net(torch.from_numpy(EX_IN.T[None]))[0].numpy()
for h in hooks:
    h.remove()
FMAPS = acts                                                     # 4 x [32, n]
STEP_T = T[[i for i in range(1, N - 1) if EX_Y[i] > EX_Y[i - 1] and EX_Y[i] >= EX_Y[i + 1] and EX_Y[i] > 0.05]]
SNAPS = LOG["snapshots"]
GRADS = np.array(LOG["grad_norm"])                               # [logged steps, 5 layers]
LAYER_NAMES = ["conv 1\n(dilation 1)", "conv 2\n(dilation 2)", "conv 3\n(dilation 4)", "conv 4\n(dilation 8)",
               "head\n(1x1 conv)"]


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def stage(fig):
    """Full-frame axes in 'diagram units': 16 wide x 9 tall."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return ax


def title(ax, text, sub=None):
    ax.text(0.6, 8.35, text, fontsize=24, fontweight="bold", va="center")
    if sub:
        ax.text(0.62, 7.75, sub, fontsize=13, color=MUTED, va="center")


def caption(ax, text):
    ax.text(8, 0.45, text, fontsize=13.5, color=TEXT, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec="#2a3040"))


def box(ax, x, y, w, h, label, color, alpha=1.0, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=PANEL, ec=color,
                                lw=2, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, alpha=alpha, linespacing=1.4)


def panel_axes(fig, rect):
    a = fig.add_axes(rect)
    a.set_facecolor(PANEL)
    for s in a.spines.values():
        s.set_color("#2a3040")
    return a


# ── scenes: each takes (fig, p) with p going 0 -> 1 ──────────────────────────

def s_title(fig, p):
    ax = stage(fig)
    a = ease(p * 2)
    ax.text(8, 5.3, "How the step counter works", fontsize=40, fontweight="bold", ha="center", alpha=a)
    ax.text(8, 4.3, "Ankle IMU  ->  UNO Q  ->  MQTT  ->  PyTorch neural net  ->  steps", fontsize=17,
            color=MUTED, ha="center", alpha=a)
    ax.text(8, 3.3, f"{LOG.get('params', 0):,} weights, trained by back-propagation on your own counted walks",
            fontsize=14, color=VIOLET, ha="center", alpha=ease(p * 2 - 0.6))


SYS = [(0.4, "MPU6050\non the ankle\naccel + gyro", AMBER), (3.5, "UNO Q\nmicrocontroller\nreads 50x / s", GREEN),
       (6.6, "UNO Q Linux\nPython", GREEN), (9.7, "Laptop\nPyTorch\nStepNet", VIOLET),
       (12.8, "LED matrix\nshows\nthe count", BLUE)]
BOX_W = 2.25
LINKS = ["I2C", "Bridge", "MQTT\nover Wi-Fi", "MQTT\ngait/steps"]


def s_system(fig, p):
    ax = stage(fig)
    title(ax, "1 - The system", "the same hardware as the gait identifier")
    for i, (x, lab, c) in enumerate(SYS):
        box(ax, x, 3.6, BOX_W, 2.2, lab, c, alpha=ease(p * 5 - i * 0.5))
    for i in range(4):
        x0, x1 = SYS[i][0] + BOX_W, SYS[i + 1][0]
        a = ease(p * 5 - i * 0.5 - 0.4)
        ax.add_patch(FancyArrowPatch((x0 + 0.05, 4.7), (x1 - 0.05, 4.7), arrowstyle="-|>", mutation_scale=18,
                                     color=MUTED, alpha=a, lw=1.6))
        ax.text((x0 + x1) / 2, 3.25, LINKS[i], ha="center", va="top", fontsize=10.5, color=MUTED, alpha=a)
        for k in range(3):  # packets flowing along the link
            q = (p * 6 + k / 3 + i * 0.2) % 1
            if p > 0.35:
                ax.add_patch(Circle((x0 + (x1 - x0) * q, 4.7), 0.08, color=SYS[i + 1][2], alpha=0.9))
    texts = ["Every 20 ms the board reads 3 accelerations + 3 rotation rates and sends them over Wi-Fi",
             "The laptop keeps the last 6 s and runs the neural net every 0.5 s",
             "The running count goes back to the board and lights up the LED matrix"]
    caption(ax, texts[min(2, int(p * 3))])


def s_inputs(fig, p):
    ax = stage(fig)
    title(ax, "2 - What the net reads", "8 channels x 50 samples per second")
    n = max(2, int(N * ease(p * 1.3)))
    for k in range(8):
        a = panel_axes(fig, [0.07, 0.17 + (7 - k) * 0.072, 0.86, 0.066])
        a.plot(T[:n], EX_IN[:n, k], color=CH_COLORS[k], lw=1.2)
        a.set_xlim(0, T[-1])
        lo, hi = EX_IN[:, k].min(), EX_IN[:, k].max()
        a.set_ylim(lo - 0.1 * (hi - lo + 1e-3), hi + 0.1 * (hi - lo + 1e-3))
        a.set_yticks([])
        a.set_xticks([] if k < 7 else np.arange(0, T[-1] + 0.1, 2))
        a.text(-0.012, 0.5, sl.CHANNEL_NAMES[k], transform=a.transAxes, ha="right", va="center", fontsize=12,
               color=CH_COLORS[k])
    caption(ax, "accel x y z (g)  -  gyro x y z (rad/s / 4)  -  |a| - 1  -  |w| / 4      (a 10 s stretch of a real walk)")


def fmap_img(m):
    m = m / (np.percentile(m, 99) + 1e-6)
    return np.clip(m, 0, 1)


def s_forward(fig, p):
    ax = stage(fig)
    title(ax, "3 - Forward pass", "1-D convolutions slide along time; each layer looks 2x further apart")
    rows = [("input (8 ch)", EX_IN.T, None)] + [(f"conv {i + 1}  (32 ch, dilation {2 ** i})", fmap_img(FMAPS[i]), None)
                                              for i in range(4)] + [("output: step density", None, EX_D)]
    y0, hrow = 0.14, 0.105
    reveal = p * 7.5
    for r, (name, img, curve) in enumerate(rows):
        a_alpha = ease(reveal - r)
        if a_alpha <= 0:
            continue
        a = panel_axes(fig, [0.2, y0 + r * (hrow + 0.012), 0.75, hrow])
        if img is not None:
            show = img if r else (img - img.min(1, keepdims=True)) / (np.ptp(img, 1, keepdims=True) + 1e-6)
            a.imshow(show, aspect="auto", cmap="magma" if r else "viridis", extent=[0, T[-1], 0, 1],
                     alpha=a_alpha, interpolation="nearest")
        else:
            a.fill_between(T, EX_D, color=VIOLET, alpha=0.5 * a_alpha)
            a.plot(T, EX_D, color=VIOLET, alpha=a_alpha)
            a.set_xlim(0, T[-1])
            a.set_ylim(0, max(0.16, EX_D.max() * 1.1))
        a.set_xticks([])
        a.set_yticks([])
        ax.text(3.05, (y0 + r * (hrow + 0.012) + hrow / 2) * 9, name, ha="right", va="center", fontsize=11.5,
                alpha=a_alpha)
    if p > 0.82:  # receptive field of one output sample, fanning down through the dilations
        a = ease((p - 0.82) / 0.15)
        i_out = int(N * 0.55)
        X = lambda i: 3.2 + 12.0 * i / N
        Y = lambda r: (y0 + r * (hrow + 0.012) + hrow / 2) * 9
        centers = [i_out]
        for lvl, d in zip(range(5, 0, -1), (8, 4, 2, 1, 1)):
            c = centers[-1]
            for k in range(-4, 5):
                ax.plot([X(c), X(c + k * d)], [Y(lvl), Y(lvl - 1)], color=AMBER, lw=1.6, alpha=0.9 * a)
            centers.append(c)
        half = 60  # receptive field: 1 + 8 x (1 + 2 + 4 + 8) = 121 samples
        ax.add_patch(FancyBboxPatch((X(i_out - half), Y(0) - hrow * 4.5), X(i_out + half) - X(i_out - half),
                                    hrow * 9, boxstyle="round,pad=0.02", fc="none", ec=AMBER, lw=2, alpha=a))
        ax.text(15.6, 8.35, "one output sample sees 2.4 s of signal", ha="right", va="center", fontsize=13,
                color=AMBER, alpha=a)
    caption(ax, "Each conv layer = 32 learned 9-tap filters + ReLU. Bright = filter firing. The output is >= 0 (softplus).")


def s_count(fig, p):
    ax = stage(fig)
    title(ax, "4 - From density to a count", "each step is a bump with area ~1 -> add it all up")
    a = panel_axes(fig, [0.07, 0.22, 0.6, 0.5])
    n = max(2, int(N * ease(p * 1.2)))
    a.fill_between(T[:n], EX_D[:n], color=VIOLET, alpha=0.45)
    a.plot(T, EX_D, color=VIOLET, alpha=0.25)
    a.plot(T[:n], EX_D[:n], color=VIOLET)
    for t in STEP_T:
        if t <= T[n - 1]:
            a.axvline(t, color=TEXT, lw=0.6, alpha=0.35)
    a.set_xlim(0, T[-1])
    a.set_ylim(0, max(0.16, EX_D.max() * 1.1))
    a.set_xlabel("time (s)")
    a.set_ylabel("steps per sample")
    cum = EX_D[:n].sum()
    ax.text(12.9, 5.3, f"{cum:5.1f}", fontsize=72, fontweight="bold", ha="center", va="center", color=TEXT)
    ax.text(12.9, 3.9, "steps so far", fontsize=15, ha="center", color=MUTED)
    ax.text(12.9, 3.2, f"you counted {EX_Y[:n].sum():.0f}", fontsize=13, ha="center", color=GREEN)
    caption(ax, "Live, the net re-reads the last 6 s every 0.5 s and adds the half second that ended 1 s ago")


def s_loss(fig, p):
    ax = stage(fig)
    title(ax, "5 - The loss: how wrong is it?", "compared with your counted steps, placed in time by the gyro")
    snap = next((s for s in SNAPS[1:] if sum(s["density"]) < 2 * EX_Y.sum()), SNAPS[-1])
    d0 = np.array(snap["density"])
    a = panel_axes(fig, [0.07, 0.2, 0.55, 0.52])
    a.plot(T, EX_Y, color=TEXT, lw=1.4, label="target (your steps)")
    a.plot(T, d0, color=VIOLET, lw=1.4, label=f"net at training step {snap['iter']}")
    if p > 0.25:
        a.fill_between(T, EX_Y, d0, color=RED, alpha=0.35 * ease((p - 0.25) * 4), label="difference")
    a.set_xlim(0, T[-1])
    a.set_ylim(0, max(EX_Y.max(), d0.max()) * 1.15)
    a.legend(loc="upper right", facecolor=PANEL, edgecolor="#2a3040", labelcolor=TEXT)
    a.set_xlabel("time (s)")
    shape = 50 * np.mean((d0 - EX_Y) ** 2)
    count = 0.1 * (d0.sum() - EX_Y.sum()) ** 2
    al = ease((p - 0.4) * 3)
    ax.text(10.6, 6.2, "shape loss", fontsize=15, color=RED, alpha=al)
    ax.text(10.6, 5.6, "= 50 x mean( (density - target)^2 )", fontsize=12.5, alpha=al)
    ax.text(10.6, 5.05, f"= {shape:.4f}", fontsize=13, color=MUTED, alpha=al)
    al2 = ease((p - 0.6) * 3)
    ax.text(10.6, 4.2, "count loss", fontsize=15, color=AMBER, alpha=al2)
    ax.text(10.6, 3.6, "= 0.1 x ( sum(density) - steps )^2", fontsize=12.5, alpha=al2)
    ax.text(10.6, 3.05, f"= 0.1 x ({d0.sum():.1f} - {EX_Y.sum():.0f})^2 = {count:.2f}", fontsize=13, color=MUTED,
            alpha=al2)
    caption(ax, "One number says how wrong the net is. Training = nudging all the weights to make it smaller.")


def s_backprop(fig, p):
    ax = stage(fig)
    title(ax, "6 - Back-propagation", "loss.backward(): the chain rule, from the loss back to every weight")
    names = ["input"] + LAYER_NAMES + ["loss"]
    xs = np.linspace(1.0, 13.2, len(names))
    w = 1.55
    mid = GRADS[len(GRADS) // 3]
    gmax = mid.max()
    fwd = ease(p * 2.2)                # forward wave, left -> right
    bwd = ease((p - 0.42) * 2.2)       # gradient wave, right -> left
    for i, (x, nm) in enumerate(zip(xs, names)):
        c = RED if nm == "loss" else (TEXT if nm == "input" else VIOLET)
        box(ax, x, 4.9, w, 1.4, nm, c, fs=11)
        if i < len(names) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 5.85), (xs[i + 1] - 0.03, 5.85), arrowstyle="-|>",
                                         mutation_scale=14, color=BLUE, alpha=0.3 + 0.7 * (fwd * len(names) > i + 1)))
            ax.add_patch(FancyArrowPatch((xs[i + 1] - 0.03, 5.35), (x + w + 0.03, 5.35), arrowstyle="-|>",
                                         mutation_scale=14, color=AMBER,
                                         alpha=0.15 + 0.85 * (bwd * len(names) > len(names) - 1 - i)))
    # forward / backward pulses
    if 0 < fwd < 1:
        ax.add_patch(Circle((xs[0] + w / 2 + (xs[-1] - xs[0]) * fwd, 6.55), 0.13, color=BLUE))
    if 0 < bwd < 1:
        ax.add_patch(Circle((xs[-1] + w / 2 - (xs[-1] - xs[0]) * bwd, 4.65), 0.13, color=AMBER))
    # real gradient sizes appear as the wave reaches each layer
    for li in range(5):
        x = xs[li + 1]
        reached = bwd * len(names) > len(names) - 2 - li
        if reached:
            h = 1.4 * np.sqrt(mid[li] / gmax)  # square root so the small ones stay visible
            ax.add_patch(FancyBboxPatch((x + 0.45, 4.35 - h), 0.65, h, boxstyle="round,pad=0.01", fc=AMBER, ec="none",
                                        alpha=0.85))
            ax.text(x + w / 2, 4.15 - h, f"|dL/dw| = {mid[li]:.3f}", ha="center", va="top", fontsize=9.5,
                    color=AMBER)
    ax.text(0.7, 7.2, "forward:  input -> layers -> density -> loss", color=BLUE, fontsize=13, alpha=ease(p * 4))
    ax.text(0.7, 6.8, "backward: dL/dw = dL/d(density) x d(density)/d(layer 4) x ... x d(layer)/dw",
            color=AMBER, fontsize=13, alpha=ease((p - 0.42) * 4))
    code = ["d = net(x)                    # forward", "loss = shape_loss + count_loss",
            "loss.backward()               # back-propagation", "optimizer.step()              # w <- w - lr x Adam(dL/dw)"]
    hl = 0 if p < 0.38 else (1 if p < 0.45 else (2 if p < 0.85 else 3))
    for i, line in enumerate(code):
        ax.text(0.7, 2.25 - i * 0.36, line, family="monospace", fontsize=12, color=TEXT if i == hl else MUTED,
                fontweight="bold" if i == hl else "normal")
    caption(ax, "Bar = size of the real gradient for that layer's weights, measured during training")


def s_training(fig, p):
    ax = stage(fig)
    title(ax, "7 - Training", f"{LOG['iter'][-1] + 1} steps of: random 8 s crops -> forward -> loss -> backward -> update")
    k = min(len(SNAPS) - 1, int(ease(p * 1.1) * (len(SNAPS) - 1)))
    it = SNAPS[k]["iter"]
    a1 = panel_axes(fig, [0.06, 0.2, 0.36, 0.5])
    li = np.array(LOG["iter"])
    sel = li <= it
    a1.semilogy(li, LOG["loss"], color=VIOLET, alpha=0.12)
    a1.semilogy(li[sel], np.array(LOG["loss"])[sel], color=VIOLET)
    a1.set_xlabel("training step")
    a1.set_ylabel("loss (log)")
    a1.set_xlim(0, li[-1])
    a2 = panel_axes(fig, [0.5, 0.2, 0.45, 0.5])
    d = np.array(SNAPS[k]["density"])
    a2.plot(T, EX_Y, color=TEXT, lw=1.2, label="target")
    a2.fill_between(T, d, color=VIOLET, alpha=0.4)
    a2.plot(T, d, color=VIOLET, lw=1.3, label="net's density")
    a2.set_xlim(0, T[-1])
    a2.set_ylim(0, max(EX_Y.max(), 0.16) * 1.3)
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper right", facecolor=PANEL, edgecolor="#2a3040", labelcolor=TEXT)
    ax.text(8.2, 6.9, f"step {it:5d}", fontsize=20, fontweight="bold", family="monospace")
    ax.text(15.2, 6.9, f"counts {d.sum():5.1f} of {EX_Y.sum():.0f}", fontsize=17, ha="right", family="monospace",
            color=GREEN if abs(d.sum() - EX_Y.sum()) < 1.5 else AMBER)
    caption(ax, "Augmented every step: random sensor tilt (+/-25 deg), signal scale, and walking speed (+/-12%)")


def s_result(fig, p):
    ax = stage(fig)
    title(ax, "8 - How good is it?", "tested on recordings the net never trained on")
    m = (METRICS or {}).get("net")
    if m:
        lines = [(f"{m['accuracy']:.1f}%", "of steps counted right", GREEN),
                 (f"+/- {1.96 * m['sd']:.1f}%", "95% uncertainty of one count", VIOLET)]
        if m.get("false_per_min") is not None:
            lines.append((f"{m['false_per_min']:.1f}", "false steps per minute when not walking", AMBER))
    else:
        lines = [("--", "run train_steps.py with cross-validation for the numbers", MUTED)]
    for i, (v, k, c) in enumerate(lines):
        a = ease(p * 4 - i * 0.6)
        ax.text(5.2, 6.0 - i * 1.5, v, fontsize=40, fontweight="bold", ha="right", va="center", color=c, alpha=a)
        ax.text(5.6, 6.0 - i * 1.5, k, fontsize=16, va="center", alpha=a)
    caption(ax, "Neural net and PyTorch on the laptop  -  counts back to the UNO Q's LED matrix")


SCENES = [(s_title, 4), (s_system, 10), (s_inputs, 7), (s_forward, 13), (s_count, 8), (s_loss, 9),
          (s_backprop, 13), (s_training, 12), (s_result, 6)]


def main():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    writer = FFMpegWriter(fps=FPS, bitrate=4000, metadata={"title": "How the step counter works"})
    total = sum(d for _, d in SCENES) * FPS
    done = 0
    with writer.saving(fig, OUT, dpi=100):
        for scene, dur in SCENES:
            n = int(dur * FPS)
            for f in range(n):
                fig.clf()
                fig.set_facecolor(BG)
                scene(fig, f / (n - 1))
                writer.grab_frame(facecolor=BG)
                done += 1
                if done % (FPS * 5) == 0:
                    print(f"  {done}/{total} frames", flush=True)
    print(f"Saved {OUT} ({total / FPS:.0f} s)")


if __name__ == "__main__":
    main()
