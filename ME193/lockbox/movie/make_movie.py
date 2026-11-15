"""
Animated explainer for LunchLock: how the finger-counting network works.

  python make_movie.py            -> LunchLock_explainer.mp4  (1280x720, 24 fps, silent - add your voice-over)
  python make_movie.py --stills   -> also saves one still per scene to stills/ (for checking / the Word doc)

Everything shown is REAL: your board-camera photo, the hand mask, the filters the network
learned, its actual feature maps at every layer, its prediction, the training curve from
model_report.json, and the V-data fits from vdata/fit_v.py.
Scene start times are written to scenes.json so the voice-over script lines up.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
LOCKBOX = HERE.parent
sys.path[:0] = [str(LOCKBOX), str(LOCKBOX / "vdata")]
import finger_model as fm  # noqa: E402

W, H, FPS = 1280, 720, 24
BG = (11, 13, 18)
FG = (233, 236, 242)
MUTED = (139, 147, 165)
ACCENT = (124, 156, 255)
OK = (62, 207, 142)
WARN = (255, 181, 71)
BAD = (255, 92, 122)
SAMPLE = fm.DATA_DIR / "3_760.jpg"          # a 3-finger photo through the UNO Q's camera


def font(size, bold=False):
    for name in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE, F_H, F_B, F_S, F_XS = font(64, True), font(38, True), font(26), font(21), font(16)
F_MONO = ImageFont.truetype("consola.ttf", 15) if Path("C:/Windows/Fonts/consola.ttf").exists() else F_XS


# ───────────────────────────────────────────── helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    """0 before a, 1 after b, eased in between."""
    return ease((t - a) / max(b - a, 1e-6))


def canvas():
    return Image.new("RGB", (W, H), BG)


def heat(arr, size, cmap="magma", vmin=None, vmax=None):
    a = np.asarray(arr, np.float32)
    lo = a.min() if vmin is None else vmin
    hi = a.max() if vmax is None else vmax
    n = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (matplotlib.colormaps[cmap](n)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.NEAREST)


def bgr_img(bgr, size, nearest=False):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(size, Image.NEAREST if nearest else Image.LANCZOS)


def paste_alpha(base, img, xy, alpha):
    if alpha <= 0:
        return
    if alpha >= 1:
        base.paste(img, xy)
        return
    region = base.crop((xy[0], xy[1], xy[0] + img.width, xy[1] + img.height))
    base.paste(Image.blend(region, img.convert("RGB"), alpha), xy)


def text(d, xy, s, f=F_B, fill=FG, alpha=1.0, anchor="la"):
    if alpha <= 0:
        return
    c = tuple(int(BG[i] + (fill[i] - BG[i]) * alpha) for i in range(3))
    d.text(xy, s, font=f, fill=c, anchor=anchor)


def header(d, title, sub, t):
    text(d, (60, 40), title, F_H, FG, seg(t, 0, 0.6))
    text(d, (60, 92), sub, F_S, MUTED, seg(t, 0.2, 0.9))


def arrow(d, x0, y, x1, color=MUTED, w=4):
    d.line((x0, y, x1, y), fill=color, width=w)
    d.polygon([(x1, y), (x1 - 14, y - 9), (x1 - 14, y + 9)], fill=color)


def box(d, xy, outline=MUTED, fill=None, w=2, r=10):
    d.rounded_rectangle(xy, r, outline=outline, fill=fill, width=w)


# ───────────────────────────────────────────── real data from the trained network
class Data:
    def __init__(self):
        self.model = fm.load_model()
        m = self.model
        bgr = cv2.imread(str(SAMPLE))
        self.photo = bgr
        x = fm.preprocess(bgr)
        self.rgb96 = x[0].permute(1, 2, 0).numpy()
        acts = {}
        hooks = [m.features[i].register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(i, out.detach()[0].numpy()))
                 for i in (1, 2, 3, 7, 11)]               # BN1 (pre-ReLU), ReLU1, pool1, pool2, pool3
        with torch.no_grad():
            self.probs = torch.softmax(m(x), 1)[0].numpy()
        for h in hooks:
            h.remove()
        self.pre1, self.relu1, self.pool1, self.pool2, self.pool3 = acts[1], acts[2], acts[3], acts[7], acts[11]
        mask, density, profile = fm.explain(bgr)
        self.mask, self.density, self.profile = mask, density, profile
        g = 0.299 * self.rgb96[..., 0] + 0.587 * self.rgb96[..., 1] + 0.114 * self.rgb96[..., 2]
        self.gray = g
        # most interesting first-layer filter = largest response spread on this photo
        self.fi = int(np.argmax(self.pre1.reshape(16, -1).std(1)))
        self.filters = m.features[0].weight.detach()[:, 0].numpy()        # 16 x 5 x 5 (photo channel)
        rep = json.loads((LOCKBOX / "model_report.json").read_text())
        self.hist = rep["history"]
        self.val_acc = rep["val_acc"]
        self._v_data()
        self._policy_trace()

    def _v_data(self):
        import fit_v
        x, y = fit_v.load()
        self.vx, self.vy = x, y
        # straight line: every step of the first 5 iterations + a few later
        m, b, snaps = 0.0, 0.0, []
        n = len(x)
        for ep in range(60):
            err = m * x + b - y
            snaps.append((m, b, float(np.mean(err ** 2))))
            m, b = m - 0.3 * 2 / n * np.sum(err * x), b - 0.3 * 2 / n * np.sum(err)
        self.line_snaps = snaps
        # two ReLU neurons: snapshots on a log-ish schedule
        w, bb, v, c = np.array([-5.0, 5.0]), np.array([3.0, -3.0]), np.array([10.0, 10.0]), 0.0
        keep = sorted(set([int(v_) for v_ in np.geomspace(1, 30000, 60)] + [0]))
        snaps, ep = [], 0
        for target in keep:
            while ep < target:
                _, dw, db, dv, dc = fit_v.grads(x, y, w, bb, v, c)
                w, bb, v, c = w - 0.005 * dw, bb - 0.005 * db, v - 0.005 * dv, c - 0.005 * dc
                ep += 1
            snaps.append((ep, w.copy(), bb.copy(), v.copy(), c, float(fit_v.grads(x, y, w, bb, v, c)[0])))
        self.relu_snaps = snaps

    def _policy_trace(self):
        """Real network confidences for a played-back hand: 3, change, 1, change, 4."""
        seq = []
        def frames(cls, start, n):
            return [cv2.imread(str(fm.DATA_DIR / f"{cls}_{start + j}.jpg")) for j in range(n)]
        a3, a1, a4 = frames(3, 700, 14), frames(1, 520, 14), frames(4, 700, 14)
        def blend(p, q, k):
            return [cv2.addWeighted(p[-1], 1 - (j + 1) / (k + 1), q[0], (j + 1) / (k + 1), 0) for j in range(k)]
        for f in a3 + blend(a3, a1, 4) + a1 + blend(a1, a4, 4) + a4:
            k, conf, p = fm.predict(self.model, f)
            seq.append((k, conf))
        self.trace = seq            # one entry per 0.1 s


# ───────────────────────────────────────────── scenes  (each: (seconds, function(img, t, T, D)))
def s_title(img, t, T, D):
    d = ImageDraw.Draw(img)
    a = seg(t, 0.2, 1.4)
    text(d, (W // 2, 290), "LunchLock", F_TITLE, FG, a, "mm")
    text(d, (W // 2, 360), "How a neural network counts your fingers", F_H, ACCENT, seg(t, 0.8, 2.0), "mm")
    text(d, (W // 2, 420), "a finger-code lunch safe  ·  PyTorch CNN  ·  Arduino UNO Q", F_S, MUTED, seg(t, 1.5, 2.6), "mm")


def s_lock(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "The idea", "A fridge lunch box that only opens for your finger code", t)
    xs = [140, 520, 900]
    labels = ["1  Press the button", "2  Show 3 · 1 · 4", "3  The lid opens"]
    for i, x in enumerate(xs):
        a = seg(t, 1 + i * 2.5, 2 + i * 2.5)
        box(d, (x, 200, x + 260, 560), outline=tuple(int(BG[j] + (MUTED[j] - BG[j]) * a) for j in range(3)))
        text(d, (x + 130, 600), labels[i], F_S, FG, a, "mm")
    # 1: button
    a = seg(t, 1, 2)
    pressed = 2.2 < t < 2.8
    if a > 0:
        cy = 380 + (6 if pressed else 0)
        d.ellipse((xs[0] + 80, cy - 50, xs[0] + 180, cy + 50), fill=(int(224 * a), int(52 * a), int(79 * a)))
    # 2: three code boxes fill
    for j, digit in enumerate("314"):
        filled = t > 4.5 + j * 1.2
        col = WARN if filled else MUTED
        bx = xs[1] + 30 + j * 70
        if t > 3.5:
            box(d, (bx, 340, bx + 60, 420), outline=col, fill=WARN if filled else None)
            if filled:
                text(d, (bx + 30, 380), digit, F_H, BG, 1, "mm")
    # 3: lid opens
    if t > 6:
        ang = 90 * seg(t, 8.5, 10)
        base = (xs[2] + 40, 470)
        d.rectangle((xs[2] + 40, 400, xs[2] + 220, 470), outline=FG, width=3)
        L = 180
        ex = base[0] + L * np.cos(np.radians(ang))
        ey = 400 - L * np.sin(np.radians(ang))
        d.line((base[0], 400, ex, ey), fill=OK if ang > 80 else FG, width=6)
        if ang > 80:
            text(d, (xs[2] + 130, 300), "OPEN", F_H, OK, seg(t, 9.6, 10.2), "mm")


def s_camera(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "What the network sees", "The UNO Q's webcam, cropped to the hand box, shrunk to 96 × 96 pixels", t)
    big = bgr_img(D.photo, (400, 400))
    paste_alpha(img, big, (90, 180), seg(t, 0.5, 1.5))
    text(d, (290, 600), "camera crop (360 × 360)", F_S, MUTED, seg(t, 0.5, 1.5), "mm")
    a = seg(t, 2.5, 3.5)
    arrow(d, 520, 380, 600, MUTED) if a > 0.5 else None
    small = Image.fromarray((D.rgb96 * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST)
    paste_alpha(img, small, (630, 180), a)
    text(d, (830, 600), "96 × 96 pixels", F_S, MUTED, a, "mm")
    # zoom into numbers
    z = seg(t, 5.5, 6.5)
    if z > 0:
        gx, gy = 40, 30
        patch = D.gray[gy:gy + 5, gx:gx + 5]
        box(d, (1045, 230, 1250, 400), outline=ACCENT)
        for r in range(5):
            for c in range(5):
                text(d, (1068 + c * 40, 252 + r * 32), f"{patch[r, c]:.2f}", F_MONO, FG, z, "mm")
        text(d, (1147, 430), "each pixel is", F_XS, MUTED, z, "mm")
        text(d, (1147, 452), "just a number (0-1)", F_XS, MUTED, z, "mm")
    text(d, (W // 2, 668), "96 × 96 × 3 colours = 27,648 numbers in — 6 scores out", F_B, ACCENT, seg(t, 7.5, 8.5), "mm")


def s_mask(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Step 1 · The hand mask (my 'grid' idea)", "Skin is redder than the wall: compare every pixel's redness to the background", t)
    S = 300
    paste_alpha(img, bgr_img(D.photo, (S, S)), (80, 200), seg(t, 0.4, 1.2))
    text(d, (230, 530), "photo", F_S, MUTED, seg(t, 0.4, 1.2), "mm")
    arrow(d, 400, 350, 470) if t > 2 else None
    paste_alpha(img, heat(D.mask, (S, S), "gray", 0, 1), (490, 200), seg(t, 2, 3.2))
    text(d, (640, 530), "hand mask", F_S, MUTED, seg(t, 2, 3.2), "mm")
    arrow(d, 810, 350, 880) if t > 5 else None
    # grid view (density cells + profile)
    a = seg(t, 5, 6.5)
    if a > 0:
        gv = heat(D.mask, (S, S), "gray", 0, 1).convert("RGB")
        gv = Image.blend(Image.new("RGB", (S, S), (0, 0, 0)), gv, 0.3)
        gd = ImageDraw.Draw(gv)
        cell = S / 12
        for r in range(12):
            for c in range(12):
                dv = float(D.density[r, c])
                if dv > 0.05:
                    gd.rectangle((c * cell, r * cell, (c + 1) * cell, (r + 1) * cell),
                                 fill=(int(124 * dv), int(156 * dv), int(255 * dv)))
        for i in range(1, 12):
            gd.line((i * cell, 0, i * cell, S), fill=(60, 60, 70))
            gd.line((0, i * cell, S, i * cell), fill=(60, 60, 70))
        n = int(len(D.profile) * seg(t, 7, 9))
        pts = [((i + 0.5) * S / 24, S * (1 - D.profile[i])) for i in range(len(D.profile))][:max(n, 2)]
        if n >= 2:
            gd.line(pts, fill=WARN, width=4)
        paste_alpha(img, gv, (900, 200), a)
        text(d, (1050, 530), "12×12 grid + finger heights", F_S, MUTED, a, "mm")
    text(d, (W // 2, 600), "Raised fingers = tall bumps in the yellow profile.", F_B, WARN, seg(t, 9, 10), "mm")
    text(d, (W // 2, 650), "The mask goes into the network next to the photo: 2 input channels (\"early fusion\")",
         F_S, ACCENT, seg(t, 11, 12), "mm")


def s_conv(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Step 2 · Convolution", "A 5×5 filter slides over the image: multiply, add up, write one number", t)
    S, X0, Y0 = 384, 70, 190
    inp = heat(D.gray, (S, S), "gray")
    img.paste(inp, (X0, Y0))
    text(d, (X0 + S // 2, Y0 + S + 30), "input (photo channel)", F_S, MUTED, 1, "mm")
    # filter weights
    f = D.filters[D.fi]
    lim = float(np.abs(f).max())
    fimg = heat(f, (175, 175), "coolwarm", -lim, lim)
    img.paste(fimg, (552, 250))
    for r in range(5):
        for c in range(5):
            text(d, (552 + c * 35 + 17, 250 + r * 35 + 17), f"{f[r, c]:+.1f}", F_XS, (20, 20, 20), 1, "mm")
    text(d, (640, 450), f"learned filter #{D.fi + 1}", F_S, MUTED, 1, "mm")
    text(d, (640, 480), "(16 of these in layer 1)", F_XS, MUTED, 1, "mm")
    # sweep: position along raster
    p = seg(t, 1.5, T - 4)
    total = 96 * 96
    k = int(p * (total - 1))
    rr, cc = divmod(k, 96)
    out = D.pre1[D.fi]
    lo, hi = float(out.min()), float(out.max())
    shown = np.full(out.shape, np.nan, np.float32)
    shown.reshape(-1)[:k + 1] = out.reshape(-1)[:k + 1]
    oimg = heat(np.nan_to_num(shown, nan=lo), (S, S), "magma", lo, hi)
    hidden = np.isnan(shown)
    if hidden.any():
        mask = Image.fromarray((np.kron(hidden, np.ones((4, 4))) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
        oimg.paste(Image.new("RGB", (S, S), (24, 27, 36)), (0, 0), mask)
    img.paste(oimg, (826, Y0))
    text(d, (826 + S // 2, Y0 + S + 30), "output feature map", F_S, MUTED, 1, "mm")
    # moving windows
    sc = S / 96
    d.rectangle((X0 + (cc - 2) * sc, Y0 + (rr - 2) * sc, X0 + (cc + 3) * sc, Y0 + (rr + 3) * sc), outline=WARN, width=3)
    d.rectangle((826 + cc * sc, Y0 + rr * sc, 826 + (cc + 1) * sc, Y0 + (rr + 1) * sc), outline=WARN, width=3)
    d.line((X0 + (cc + 0.5) * sc, Y0 + (rr + 0.5) * sc, 552, 337), fill=(120, 100, 50), width=1)
    d.line((727, 337, 826 + (cc + 0.5) * sc, Y0 + (rr + 0.5) * sc), fill=(120, 100, 50), width=1)
    text(d, (W // 2, 650), "Bright = \"this filter's pattern is here\" — the network LEARNED these 25 weights by itself",
         F_S, ACCENT, seg(t, T - 4, T - 3), "mm")


def s_relu(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Step 3 · ReLU", "Keep the positive responses, set every negative one to zero:  ReLU(z) = max(0, z)", t)
    S = 340
    pre = D.pre1[D.fi]
    lim = float(np.abs(pre).max())
    img.paste(heat(pre, (S, S), "coolwarm", -lim, lim), (90, 200))
    text(d, (260, 570), "before (blue = negative)", F_S, MUTED, 1, "mm")
    # graph of ReLU
    gx, gy, gw = 520, 260, 240
    d.line((gx, gy + gw // 2, gx + gw, gy + gw // 2), fill=MUTED, width=2)
    d.line((gx + gw // 2, gy, gx + gw // 2, gy + gw), fill=MUTED, width=2)
    p = seg(t, 1, 3)
    d.line((gx, gy + gw // 2, gx + gw // 2, gy + gw // 2), fill=WARN, width=5)
    if p > 0:
        d.line((gx + gw // 2, gy + gw // 2, gx + gw // 2 + p * gw / 2, gy + gw // 2 - p * gw / 2), fill=WARN, width=5)
    text(d, (gx + gw // 2, gy + gw + 30), "ReLU", F_B, WARN, 1, "mm")
    a = seg(t, 3.5, 5)
    post = np.maximum(pre, 0)
    after = heat(post, (S, S), "coolwarm", -lim, lim)
    paste_alpha(img, after, (850, 200), a)
    text(d, (1020, 570), "after (negatives gone)", F_S, MUTED, a, "mm")
    text(d, (W // 2, 650), "Without it, stacking layers would still only draw straight lines (see the V-data part)",
         F_S, ACCENT, seg(t, 5.5, 6.5), "mm")


def s_pool(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Step 4 · Max-pooling", "Keep only the strongest value in every 2 × 2 block: half the width, half the height", t)
    a = D.relu1[D.fi]
    gy, gx = int(np.unravel_index(np.argmax(a), a.shape)[0]), int(np.unravel_index(np.argmax(a), a.shape)[1])
    gy, gx = min(max(gy - 4, 0), 88), min(max(gx - 4, 0), 88)
    patch = a[gy:gy + 8, gx:gx + 8]
    C, X0, Y0 = 52, 90, 190
    vmax = float(patch.max()) + 1e-6
    done = int(16 * seg(t, 1.5, T - 4))
    for r in range(8):
        for c in range(8):
            v = patch[r, c]
            shade = int(40 + 180 * v / vmax)
            d.rectangle((X0 + c * C, Y0 + r * C, X0 + (c + 1) * C - 2, Y0 + (r + 1) * C - 2), fill=(shade, shade // 2, shade // 3))
            text(d, (X0 + c * C + C // 2, Y0 + r * C + C // 2), f"{v:.1f}", F_XS, FG, 1, "mm")
    OX = 700
    for b in range(16):
        br, bc = divmod(b, 4)
        blk = patch[br * 2:br * 2 + 2, bc * 2:bc * 2 + 2]
        if b < done:
            m = float(blk.max())
            shade = int(40 + 180 * m / vmax)
            d.rectangle((OX + bc * C * 1.3, Y0 + br * C * 1.3, OX + (bc + 1) * C * 1.3 - 3, Y0 + (br + 1) * C * 1.3 - 3),
                        fill=(shade, shade // 2, shade // 3))
            text(d, (OX + bc * C * 1.3 + C * 0.65, Y0 + br * C * 1.3 + C * 0.65), f"{m:.1f}", F_S, FG, 1, "mm")
        if b == min(done, 15) and done < 16:
            d.rectangle((X0 + bc * 2 * C - 2, Y0 + br * 2 * C - 2, X0 + (bc + 1) * 2 * C, Y0 + (br + 1) * 2 * C), outline=WARN, width=4)
    text(d, (X0 + 4 * C, Y0 + 8 * C + 30), "8 × 8 corner of the feature map", F_S, MUTED, 1, "mm")
    text(d, (OX + 2.6 * C, Y0 + 8 * C + 30), "→ 4 × 4 after pooling", F_S, MUTED, 1, "mm")
    text(d, (W // 2, 660), "Cares WHETHER a finger is there, less exactly WHERE — and 4× less work for the next layer",
         F_S, ACCENT, seg(t, T - 4, T - 3), "mm")


def grid_maps(maps, cols, cell, gap=3):
    n = maps.shape[0]
    rows = (n + cols - 1) // cols
    out = Image.new("RGB", (cols * (cell + gap), rows * (cell + gap)), BG)
    for i in range(n):
        r, c = divmod(i, cols)
        out.paste(heat(maps[i], (cell, cell), "magma"), (c * (cell + gap), r * (cell + gap)))
    return out


def s_deep(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Steps 2–4, three times", "Each block finds bigger patterns: edges → finger parts → \"raised finger here\"", t)
    cols = [(D.pool1, 4, 52, "conv 5×5 · 16 filters", "16 maps · 48×48"),
            (D.pool2, 6, 38, "conv 3×3 · 32 filters", "32 maps · 24×24"),
            (D.pool3, 8, 30, "conv 3×3 · 64 filters", "64 maps · 12×12")]
    xs = [70, 470, 880]
    for i, (maps, nc, cell, l1, l2) in enumerate(cols):
        a = seg(t, 0.8 + i * 3, 2 + i * 3)
        if a <= 0:
            continue
        g = grid_maps(maps, nc, cell)
        g.thumbnail((360, 400))
        paste_alpha(img, g, (xs[i], 170), a)
        text(d, (xs[i] + 180, 590), l1, F_S, FG, a, "mm")
        text(d, (xs[i] + 180, 620), l2, F_S, MUTED, a, "mm")
        if i < 2 and a > 0.9:
            arrow(d, xs[i] + 365, 360, xs[i] + 400)
    text(d, (W // 2, 675), "width = filters per layer (16, 32, 64)   ·   depth = 3 conv layers + 1 linear layer",
         F_S, ACCENT, seg(t, 10, 11), "mm")


def s_answer(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Step 5 · One linear layer → 6 scores → probabilities", "64 × 12 × 12 = 9,216 numbers + 168 grid numbers, each with its own weight", t)
    # flatten bar
    a = seg(t, 0.8, 2)
    flat = np.concatenate([D.pool3.reshape(-1), np.zeros(168)])
    strip = heat(flat[None, :], (700, 40), "magma")
    paste_alpha(img, strip, (90, 220), a)
    text(d, (90, 280), "all features in one long row (9,384 numbers)", F_S, MUTED, a)
    # lines to 6 neurons
    b = seg(t, 2.5, 3.5)
    for k in range(6):
        y = 200 + k * 70
        if b > 0:
            for j in range(0, 700, 70):
                d.line((90 + j, 260, 880, y + 20), fill=(40, 46, 62), width=1)
            d.ellipse((870, y, 910, y + 40), outline=FG, width=2)
            text(d, (930, y + 20), f"{k}", F_B, FG, b, "lm")
    # probability bars
    c = seg(t, 4.5, 7.5)
    for k in range(6):
        y = 200 + k * 70
        p = float(D.probs[k]) * c
        col = OK if k == int(np.argmax(D.probs)) else MUTED
        if c > 0:
            d.rounded_rectangle((970, y + 8, 970 + max(4, 200 * p), y + 32), 6, fill=col)
            text(d, (1245, y + 20), f"{100 * p:.0f}%", F_S, FG, c, "rm")
    k = int(np.argmax(D.probs))
    text(d, (W // 2, 660), f"softmax turns scores into probabilities → \"{k} fingers\", {100 * D.probs[k]:.1f}% sure",
         F_B, OK, seg(t, 8, 9), "mm")


def s_train(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "How it learned: cross-entropy + backpropagation",
           "Show it a photo → compare with the right answer → nudge all 80,822 weights a little", t)
    # loss curve
    X0, Y0, GW, GH = 80, 190, 560, 360
    d.rectangle((X0, Y0, X0 + GW, Y0 + GH), outline=(50, 55, 70))
    tl, vl = np.array(D.hist["train_loss"]), np.array(D.hist["val_loss"])
    ymax = float(max(tl.max(), vl.max()))
    n = max(2, int(len(tl) * seg(t, 1, T - 5)))
    for arr, col in ((tl, ACCENT), (vl, WARN)):
        pts = [(X0 + i / (len(arr) - 1) * GW, Y0 + GH - arr[i] / ymax * GH) for i in range(n)]
        d.line(pts, fill=col, width=3)
    text(d, (X0, Y0 + GH + 15), "epoch →", F_XS, MUTED)
    text(d, (X0 + 10, Y0 + 10), "loss (blue = training, yellow = held-out photos)", F_XS, MUTED)
    # explanation
    a = seg(t, 2, 3)
    text(d, (700, 200), "Cross-entropy loss = −log( p of the right answer )", F_S, FG, a)
    text(d, (700, 240), "  p = 0.25 → loss 1.39   (bad)", F_MONO, BAD, a)
    text(d, (700, 265), "  p = 0.99 → loss 0.01   (good)", F_MONO, OK, a)
    b = seg(t, 5, 6)
    layers = ["input", "conv1", "conv2", "conv3", "linear", "loss"]
    for i, l in enumerate(layers):
        x = 700 + i * 92
        box(d, (x, 330, x + 80, 380), outline=MUTED)
        text(d, (x + 40, 355), l, F_XS, FG, b, "mm")
    if b > 0:
        ph = (t * 0.8) % 2
        if ph < 1:
            px = 700 + ph * 5 * 92
            d.ellipse((px + 30, 395, px + 50, 415), fill=ACCENT)
            text(d, (700, 430), "forward: photo → answer", F_S, ACCENT, b)
        else:
            px = 700 + (2 - ph) * 5 * 92
            d.ellipse((px + 30, 395, px + 50, 415), fill=WARN)
            text(d, (700, 430), "backward: chain rule → how much each weight caused the error", F_S, WARN, b)
    text(d, (700, 490), "Adam (gradient descent) then moves every weight downhill.", F_S, FG, seg(t, 8, 9))
    text(d, (700, 530), f"80 epochs · 3,600 photos · rotated, shifted, re-lit every time", F_S, MUTED, seg(t, 9, 10))
    text(d, (W // 2, 660), f"Result: {100 * D.val_acc:.0f}% on held-out photos through the UNO Q's camera",
         F_B, OK, seg(t, T - 3, T - 2), "mm")


def s_vdata(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "Why ReLU matters: fitting the 'V' from Anna's Code",
           "Gradient descent: step downhill on the error, again and again", t)
    X0, Y0, GW, GH = 80, 170, 700, 440

    def P(x, y):
        return X0 + (x - 0.15) / 0.9 * GW, Y0 + GH - (y + 10) / 100 * GH

    d.rectangle((X0, Y0, X0 + GW, Y0 + GH), outline=(50, 55, 70))
    for x, y in zip(D.vx, D.vy):
        px, py = P(x, y)
        d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(110, 120, 140))
    xs = np.linspace(0.15, 1.05, 200)
    half = T * 0.42
    if t < half:
        i = min(int(seg(t, 1, half - 1) * 59), 59)
        m, b, loss = D.line_snaps[i]
        d.line([P(x, m * x + b) for x in xs], fill=ACCENT, width=4)
        text(d, (840, 220), "One neuron, no ReLU:", F_B, FG)
        text(d, (840, 260), "y = m·x + b", F_H, ACCENT)
        text(d, (840, 330), f"step {i + 1:2d}   m = {m:6.2f}   b = {b:6.2f}", F_MONO, FG)
        text(d, (840, 360), f"error (MSE) = {loss:7.1f}", F_MONO, FG)
        text(d, (840, 420), "It gets stuck near 491:", F_S, BAD, seg(t, half - 4, half - 3))
        text(d, (840, 450), "a straight line can't bend.", F_S, BAD, seg(t, half - 4, half - 3))
    else:
        i = min(int(seg(t, half + 0.5, T - 3) * (len(D.relu_snaps) - 1)), len(D.relu_snaps) - 1)
        ep, w, bb, v, c, loss = D.relu_snaps[i]
        h = np.maximum(0, np.outer(xs, w) + bb)
        yy = h @ v + c
        d.line([P(x, y) for x, y in zip(xs, yy)], fill=OK, width=4)
        for j, col in ((0, WARN), (1, ACCENT)):
            d.line([P(x, y) for x, y in zip(xs, v[j] * h[:, j] + c)], fill=col, width=2)
        text(d, (840, 220), "Two neurons with ReLU:", F_B, FG)
        text(d, (840, 260), "two ramps → a V", F_H, OK)
        text(d, (840, 330), f"epoch {ep:6d}", F_MONO, FG)
        text(d, (840, 360), f"error (MSE) = {loss:7.2f}", F_MONO, FG)
        text(d, (840, 420), "Each ReLU neuron learned one arm,", F_S, OK, seg(t, T - 4, T - 3))
        text(d, (840, 450), "both bending at x = 60.", F_S, OK, seg(t, T - 4, T - 3))


def s_policy(img, t, T, D):
    d = ImageDraw.Draw(img)
    header(d, "The lock policy: one guess is never trusted",
           "A digit counts only after the SAME answer at ≥ 85% for 0.7 s", t)
    X0, Y0, GW, GH = 80, 200, 760, 330
    d.rectangle((X0, Y0, X0 + GW, Y0 + GH), outline=(50, 55, 70))
    y85 = Y0 + GH * 0.15
    d.line((X0, y85, X0 + GW, y85), fill=WARN, width=2)
    text(d, (X0 + GW + 10, y85), "85%", F_S, WARN, 1, "lm")
    tr = D.trace
    n = max(2, int(len(tr) * seg(t, 1, T - 3)))
    colors = {1: ACCENT, 3: OK, 4: (200, 120, 255)}
    entered, cand, since, latched = [], None, 0, None
    for i in range(n):
        k, conf = tr[i]
        x = X0 + i / (len(tr) - 1) * GW
        y = Y0 + GH * (1 - conf)
        col = colors.get(k, MUTED) if conf >= 0.85 else BAD
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=col)
        steady = conf >= 0.85 and 1 <= k <= 5
        if not steady:
            cand = None
            continue
        if k == latched:
            continue
        if k != cand:
            cand, since = k, i
        if i - since >= 7:
            entered.append(k)
            latched, cand = k, None
            d.line((x, Y0, x, Y0 + GH), fill=OK, width=2)
    text(d, (X0, Y0 + GH + 20), "network confidence, 10 frames a second  (red = unsure while the hand changes)", F_XS, MUTED)
    for j in range(3):
        bx = 920 + j * 100
        filled = j < len(entered)
        box(d, (bx, 280, bx + 80, 370), outline=OK if filled else MUTED, fill=OK if filled else None)
        if filled:
            text(d, (bx + 40, 325), str(entered[j]), F_H, BG, 1, "mm")
    if len(entered) == 3:
        ok = entered == [3, 1, 4]
        text(d, (1070, 430), "OPEN" if ok else "WRONG", F_H, OK if ok else BAD, 1, "mm")
    text(d, (W // 2, 650), "Fist between equal digits · 20 s to enter · 3 wrong tries = 30 s lockout",
         F_S, ACCENT, seg(t, T - 3, T - 2), "mm")


def s_end(img, t, T, D):
    d = ImageDraw.Draw(img)
    text(d, (W // 2, 270), "LunchLock", F_TITLE, FG, seg(t, 0.2, 1), "mm")
    text(d, (W // 2, 350), "trained in PyTorch on the laptop  ·  runs in NumPy on the Arduino UNO Q", F_S, MUTED, seg(t, 0.6, 1.4), "mm")
    text(d, (W // 2, 390), "~15 frames a second  ·  80,822 weights  ·  your hand is the key", F_S, MUTED, seg(t, 1, 1.8), "mm")
    text(d, (W // 2, 470), "ME193", F_B, ACCENT, seg(t, 1.5, 2.3), "mm")


SCENES = [
    ("Title", 8, s_title),
    ("The idea", 13, s_lock),
    ("What the network sees", 14, s_camera),
    ("The hand mask (early fusion)", 25, s_mask),
    ("Convolution", 24, s_conv),
    ("ReLU", 12, s_relu),
    ("Max-pooling", 19, s_pool),
    ("Three blocks deep", 18, s_deep),
    ("Linear layer + softmax", 16, s_answer),
    ("Training: cross-entropy + backprop", 23, s_train),
    ("Gradient descent on the V data", 28, s_vdata),
    ("The lock policy", 17, s_policy),
    ("End", 8, s_end),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    args = ap.parse_args()
    print("Collecting the network's internals...")
    D = Data()
    out = HERE / "LunchLock_explainer.mp4"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    timeline, start = [], 0.0
    if args.stills:
        (HERE / "stills").mkdir(exist_ok=True)
    for idx, (name, T, fn) in enumerate(SCENES):
        timeline.append({"scene": idx + 1, "name": name, "start_s": round(start, 1), "length_s": T})
        frames = int(T * FPS)
        for f in range(frames):
            t = f / FPS
            img = canvas()
            fn(img, t, T, D)
            # small scene counter + fade between scenes
            ImageDraw.Draw(img).text((W - 30, H - 25), f"{idx + 1}/{len(SCENES)}", font=F_XS, fill=(70, 76, 92), anchor="rm")
            fade = min(1.0, t / 0.35, (T - t) / 0.35)
            if fade < 1:
                img = Image.blend(canvas(), img, max(fade, 0))
            proc.stdin.write(np.asarray(img, np.uint8).tobytes())
            if args.stills and f == int(frames * 0.85):
                slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")[:28]
                img.save(HERE / "stills" / f"{idx + 1:02d}_{slug}.png")
        start += T
        print(f"  scene {idx + 1:2d}/{len(SCENES)}  {name}  ({T} s)")
    proc.stdin.close()
    proc.wait()
    (HERE / "scenes.json").write_text(json.dumps(timeline, indent=1))
    print(f"saved {out.name}  ({start:.0f} s, {out.stat().st_size / 1e6:.1f} MB) and scenes.json")


if __name__ == "__main__":
    main()
