"""Drawing helpers for the ping-pong game.

Everything here draws onto an OpenCV BGR frame (numpy uint8 array):
    text()        nice TrueType text (Windows Segoe UI via Pillow), cached
    panel()       translucent rounded rectangle
    glow_circle() circle with a soft neon glow
    Ball          3D-shaded ball sprite (normal + "on fire" versions)
    Table         ping-pong table + net in perspective (pre-rendered once)
    draw_paddle() red paddle held at the wrist, pointing away from the elbow
    Effects       particle sparks, shockwave rings, pop-up text, red miss flash
"""

import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WHITE, BLACK = (255, 255, 255), (0, 0, 0)

# ── text ─────────────────────────────────────────────────────────────────────

FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONT_FILES = {"black": "seguibl.ttf", "bold": "segoeuib.ttf", "regular": "segoeui.ttf"}
_fonts = {}
_text_cache = {}


def _font(weight, size):
    key = (weight, size)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, FONT_FILES[weight]), size)
        except OSError:
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


def _render_text(s, size, color, weight, stroke):
    key = (s, size, color, weight, stroke)
    patch = _text_cache.get(key)
    if patch is None:
        if len(_text_cache) > 800:
            _text_cache.clear()
        font = _font(weight, size)
        left, top, right, bottom = font.getbbox(s, stroke_width=stroke)
        shadow = max(2, size // 16)
        im = Image.new("RGBA", (right - left + shadow + 2, bottom - top + shadow + 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((-left + shadow, -top + shadow), s, font=font, fill=(0, 0, 0, 120),
               stroke_width=stroke, stroke_fill=(0, 0, 0, 120))
        d.text((-left, -top), s, font=font, fill=(color[2], color[1], color[0], 255),
               stroke_width=stroke, stroke_fill=(12, 12, 18, 255))
        arr = np.asarray(im)
        patch = (arr[:, :, 2::-1].astype(np.float32), arr[:, :, 3:4].astype(np.float32) / 255.0)
        _text_cache[key] = patch
    return patch


def blit(img, bgr, alpha, x, y, opacity=1.0):
    """Alpha-blend a float BGR patch (with float alpha HxWx1) onto img at top-left (x, y)."""
    h, w = bgr.shape[:2]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    px, py = x0 - x, y0 - y
    a = alpha[py:py + y1 - y0, px:px + x1 - x0] * opacity
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + bgr[py:py + y1 - y0, px:px + x1 - x0] * a).astype(np.uint8)


def text(img, s, pos, size=28, color=WHITE, weight="bold", anchor="mm", opacity=1.0, stroke=None):
    """anchor: first letter l/m/r (horizontal), second t/m/b (vertical)."""
    size = max(6, int(size))
    if stroke is None:
        stroke = max(1, size // 18)
    bgr, a = _render_text(str(s), size, tuple(int(c) for c in color), weight, stroke)
    h, w = bgr.shape[:2]
    x = pos[0] - {"l": 0, "m": w / 2, "r": w}[anchor[0]]
    y = pos[1] - {"t": 0, "m": h / 2, "b": h}[anchor[1]]
    blit(img, bgr, a, int(x), int(y), opacity)
    return w, h


# ── shapes ───────────────────────────────────────────────────────────────────

def _clip_box(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    return max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))


def _rounded(img, x0, y0, x1, y1, r, color, thickness=-1):
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        t = thickness
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, t, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, t, cv2.LINE_AA)
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                              ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, t, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, color=(28, 22, 18), alpha=0.62, radius=16, border=None, border_thick=2):
    """Translucent rounded rectangle (optionally with a solid border)."""
    cx0, cy0, cx1, cy1 = _clip_box(img, x0 - 2, y0 - 2, x1 + 3, y1 + 3)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    roi = img[cy0:cy1, cx0:cx1]
    overlay = roi.copy()
    _rounded(overlay, int(x0) - cx0, int(y0) - cy0, int(x1) - cx0, int(y1) - cy0, radius, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    if border is not None:
        _rounded(img, int(x0), int(y0), int(x1), int(y1), radius, border, border_thick)


def glow_circle(img, center, r, color, thick=3, glow=14, strength=1.0, core=True):
    """Neon ring: a blurred additive halo plus a crisp core line."""
    x, y, r = int(center[0]), int(center[1]), int(max(1, r))
    pad = r + glow * 2 + thick
    x0, y0, x1, y1 = _clip_box(img, x - pad, y - pad, x + pad, y + pad)
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    cv2.circle(layer, (x - x0, y - y0), r, color, thick + glow // 2, cv2.LINE_AA)
    k = glow * 2 + 1
    layer = cv2.GaussianBlur(layer, (k, k), 0)
    if strength != 1.0:
        layer = cv2.convertScaleAbs(layer, alpha=strength)
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.add(roi, layer)
    if core:
        cv2.circle(img, (x, y), r, color, thick, cv2.LINE_AA)


def additive_glow(img, points, color, blur=9):
    """Soft additive blobs for trails: points = [(x, y, radius, brightness 0..1), ...]."""
    if not points:
        return
    pad = blur * 2 + max(int(p[2]) for p in points) + 2
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0, x1, y1 = _clip_box(img, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    for x, y, r, b in points:
        c = tuple(int(ch * b) for ch in color)
        cv2.circle(layer, (int(x) - x0, int(y) - y0), int(max(1, r)), c, -1, cv2.LINE_AA)
    k = blur * 2 + 1
    layer = cv2.GaussianBlur(layer, (k, k), 0)
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.add(roi, layer)


def soft_ellipse(img, center, axes, color=(0, 0, 0), alpha=0.4):
    x, y = int(center[0]), int(center[1])
    ax, ay = int(max(1, axes[0])), int(max(1, axes[1]))
    x0, y0, x1, y1 = _clip_box(img, x - ax - 6, y - ay - 6, x + ax + 7, y + ay + 7)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(mask, (x - x0, y - y0), (ax, ay), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)[..., None] * alpha
    roi[:] = (roi * (1 - mask) + np.array(color, np.float32) * mask).astype(np.uint8)


# ── ball ─────────────────────────────────────────────────────────────────────

class Ball:
    """Pre-rendered 3D-shaded ball sprites, resized per frame (cached per radius)."""

    SPRITE = 160

    def __init__(self):
        self.sprites = {
            "normal": self._make((40, 145, 255)),   # classic orange ping-pong ball (BGR)
            "fire": self._make((30, 70, 255)),      # hot red-orange when on a streak
            "white": self._make((235, 235, 240)),
            "gold": self._make((30, 205, 255)),
        }
        self._cache = {}

    def _make(self, base):
        n = self.SPRITE
        c = (n - 1) / 2
        R = n / 2 - 1
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        nx, ny = (xx - c) / R, (yy - c) / R
        d2 = nx * nx + ny * ny
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        light = np.array([-0.45, -0.6, 0.66], np.float32)
        light /= np.linalg.norm(light)
        lam = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
        half = light + np.array([0, 0, 1], np.float32)
        half /= np.linalg.norm(half)
        spec = np.clip(nx * half[0] + ny * half[1] + nz * half[2], 0, 1) ** 40
        rim = (1 - nz) ** 3 * 0.25  # slight edge darkening
        base = np.array(base, np.float32)
        bgr = base * (0.30 + 0.80 * lam[..., None]) * (1 - rim[..., None]) + 255 * 0.75 * spec[..., None]
        bgr = np.clip(bgr, 0, 255)
        dist = np.sqrt(d2) * R
        alpha = np.clip(R - dist + 0.5, 0, 1)[..., None]
        return bgr.astype(np.float32), alpha.astype(np.float32)

    def draw(self, img, x, y, r, kind="normal", opacity=1.0):
        r = max(2, int(round(r)))
        key = (kind, r)
        if key not in self._cache:
            if len(self._cache) > 300:
                self._cache.clear()
            bgr, a = self.sprites[kind]
            size = 2 * r + 1
            self._cache[key] = (cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                                cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)[..., None])
        bgr, a = self._cache[key]
        blit(img, bgr, a, int(x) - r, int(y) - r, opacity)


# ── table ────────────────────────────────────────────────────────────────────

class Table:
    """Ping-pong table in perspective, pre-rendered once and alpha-blended each frame."""

    NET_DEPTH = 0.42  # net position, as a fraction from the far edge to the near edge

    def __init__(self, w, h):
        self.w, self.h = w, h
        # Far edge sits lower than the top of the screen to leave room for the opponent.
        self.far_y, self.near_y = int(0.25 * h), int(0.64 * h)
        fl, fr, nl, nr = 0.38 * w, 0.62 * w, 0.04 * w, 0.96 * w
        self.far_left, self.far_right = fl, fr
        self.top = np.array([[fl, self.far_y], [fr, self.far_y], [nr, self.near_y], [nl, self.near_y]], np.int32)
        thick = int(0.03 * h)

        layer = np.zeros((h, w, 3), np.uint8)
        alpha = np.zeros((h, w), np.float32)

        # Table top: blue, slightly lighter toward you.
        far_c, near_c = np.array([92, 48, 14], np.float32), np.array([150, 85, 25], np.float32)
        rows = np.linspace(0, 1, self.near_y - self.far_y + 1)[:, None]
        grad = (far_c * (1 - rows) + near_c * rows).astype(np.uint8)
        top_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(top_mask, [self.top], 255)
        band = slice(self.far_y, self.near_y + 1)
        layer[band] = np.where(top_mask[band][..., None] > 0, grad[:, None, :], layer[band])
        alpha[top_mask > 0] = 0.68

        # Front edge (gives the table thickness).
        front = np.array([[nl, self.near_y], [nr, self.near_y], [nr, self.near_y + thick],
                          [nl, self.near_y + thick]], np.int32)
        cv2.fillPoly(layer, [front], (60, 30, 8))
        cv2.fillPoly(alpha, [front], 0.85)

        # White lines: border + center line.
        cv2.polylines(layer, [self.top], True, WHITE, 3, cv2.LINE_AA)
        cv2.polylines(alpha, [self.top], True, 0.95, 3)
        mid_far, mid_near = (int(w / 2), self.far_y), (int(w / 2), self.near_y)
        cv2.line(layer, mid_far, mid_near, WHITE, 2, cv2.LINE_AA)
        cv2.line(alpha, mid_far, mid_near, 0.8, 2)

        # Net: translucent mesh with a white top tape and two posts.
        f = self.NET_DEPTH
        ny = int(self.far_y + (self.near_y - self.far_y) * f)
        nxl = int(fl + (nl - fl) * f) - int(0.02 * w)
        nxr = int(fr + (nr - fr) * f) + int(0.02 * w)
        nh = int(0.065 * h)
        self.net_y = ny
        cv2.rectangle(layer, (nxl, ny - nh), (nxr, ny), (35, 35, 35), -1)
        cv2.rectangle(alpha, (nxl, ny - nh), (nxr, ny), 0.35, -1)
        for x in range(nxl, nxr, 9):
            cv2.line(layer, (x, ny - nh), (x, ny), (200, 200, 200), 1)
            cv2.line(alpha, (x, ny - nh), (x, ny), 0.35, 1)
        for y in range(ny - nh, ny, 9):
            cv2.line(layer, (nxl, y), (nxr, y), (200, 200, 200), 1)
            cv2.line(alpha, (nxl, y), (nxr, y), 0.35, 1)
        cv2.rectangle(layer, (nxl, ny - nh - 3), (nxr, ny - nh + 3), WHITE, -1)
        cv2.rectangle(alpha, (nxl, ny - nh - 3), (nxr, ny - nh + 3), 0.95, -1)
        for x in (nxl, nxr):
            cv2.rectangle(layer, (x - 5, ny - nh - 6), (x + 5, ny + 4), (50, 50, 50), -1)
            cv2.rectangle(alpha, (x - 5, ny - nh - 6), (x + 5, ny + 4), 0.95, -1)

        ys, xs = np.nonzero(alpha)
        self.box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        x0, y0, x1, y1 = self.box
        self.layer = np.ascontiguousarray(layer[y0:y1, x0:x1])
        self.a = np.ascontiguousarray(alpha[y0:y1, x0:x1])
        self.inv_a = np.ascontiguousarray(1.0 - self.a)

    def draw(self, img):
        x0, y0, x1, y1 = self.box
        roi = img[y0:y1, x0:x1]
        roi[:] = cv2.blendLinear(roi, self.layer, self.inv_a, self.a)

    def on_table(self, x, y):
        return cv2.pointPolygonTest(self.top, (float(x), float(y)), False) >= 0


# ── paddle ───────────────────────────────────────────────────────────────────

def draw_paddle(img, wrist, elbow, size, glow_color=None):
    """Red ping-pong paddle in the hand: handle at the wrist, blade pointing away from the elbow."""
    wx, wy = wrist
    if elbow is not None:
        dx, dy = wx - elbow[0], wy - elbow[1]
        n = math.hypot(dx, dy)
        dx, dy = (dx / n, dy / n) if n > 1 else (0.0, -1.0)
    else:
        dx, dy = 0.0, -1.0
    bx, by = wx + dx * size * 0.75, wy + dy * size * 0.75  # blade center
    px, py = -dy, dx  # perpendicular

    # Handle (wood) from below the wrist up to the blade.
    hw = size * 0.17
    hx0, hy0 = wx - dx * size * 0.55, wy - dy * size * 0.55
    hx1, hy1 = bx - dx * size * 0.7, by - dy * size * 0.7
    handle = np.array([[hx0 + px * hw, hy0 + py * hw], [hx1 + px * hw, hy1 + py * hw],
                       [hx1 - px * hw, hy1 - py * hw], [hx0 - px * hw, hy0 - py * hw]], np.int32)
    cv2.fillPoly(img, [handle], (70, 130, 190), cv2.LINE_AA)
    cv2.polylines(img, [handle], True, (30, 60, 100), 2, cv2.LINE_AA)

    if glow_color is not None:
        glow_circle(img, (bx, by), size * 1.02, glow_color, thick=3, glow=16, core=False)
    c = (int(bx), int(by))
    cv2.circle(img, (c[0] + 4, c[1] + 5), int(size), (20, 20, 20), -1, cv2.LINE_AA)        # shadow
    cv2.circle(img, c, int(size), (25, 25, 30), -1, cv2.LINE_AA)                          # edge
    cv2.circle(img, c, int(size * 0.92), (45, 40, 200), -1, cv2.LINE_AA)                  # red rubber
    cv2.ellipse(img, (int(bx - size * 0.2), int(by - size * 0.25)), (int(size * 0.5), int(size * 0.32)),
                -30, 0, 360, (90, 90, 235), -1, cv2.LINE_AA)                              # highlight
    cv2.circle(img, c, int(size * 0.92), (20, 20, 140), 2, cv2.LINE_AA)
    return bx, by


# ── effects ──────────────────────────────────────────────────────────────────

class Effects:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.particles = []  # [x, y, vx, vy, life, max_life, color, size]
        self.rings = []      # [x, y, t0, dur, r0, r1, color]
        self.popups = []     # [text, x, y, t0, dur, size, color, weight]
        self.flash_t0, self.flash_dur = -1.0, 0.0
        self.shake_t0, self.shake_dur, self.shake_amp = -1.0, 0.0, 0.0

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        self.vignette = np.clip((d - 0.35) / 0.65, 0, 1) ** 1.4
        self.red = np.zeros((h, w, 3), np.uint8)
        self.red[:] = (30, 20, 220)

    def burst(self, x, y, n=30, colors=((0, 220, 255), (0, 140, 255), (255, 255, 255)), speed=520, life=0.7):
        for _ in range(n):
            ang = random.uniform(0, 2 * math.pi)
            sp = speed * random.uniform(0.25, 1.0)
            lf = life * random.uniform(0.5, 1.0)
            self.particles.append([x, y, math.cos(ang) * sp, math.sin(ang) * sp - speed * 0.25,
                                   lf, lf, random.choice(colors), random.uniform(2.5, 5.5)])

    def ring(self, x, y, now, color, r0=20, r1=160, dur=0.45):
        self.rings.append([x, y, now, dur, r0, r1, color])

    def popup(self, s, x, y, now, color, size=64, dur=0.9, weight="black"):
        self.popups.append([s, x, y, now, dur, size, color, weight])

    def shake(self, now, amp=14, dur=0.3):
        self.shake_t0, self.shake_dur, self.shake_amp = now, dur, amp

    def apply_shake(self, img, now):
        """Jolt the whole picture for a moment (call last, after everything is drawn)."""
        k = (now - self.shake_t0) / self.shake_dur if self.shake_dur else 1
        if not 0 <= k < 1:
            return
        a = self.shake_amp * (1 - k) ** 2
        m = np.float32([[1, 0, random.uniform(-a, a)], [0, 1, random.uniform(-a, a)]])
        img[:] = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def red_flash(self, now, dur=0.6):
        self.flash_t0, self.flash_dur = now, dur

    def draw(self, img, now, dt):
        # Particles (simple physics with gravity + drag).
        alive = []
        for p in self.particles:
            p[4] -= dt
            if p[4] <= 0:
                continue
            p[2] *= 0.97
            p[3] = p[3] * 0.97 + 1100 * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            k = p[4] / p[5]
            color = tuple(int(c * (0.4 + 0.6 * k)) for c in p[6])
            cv2.circle(img, (int(p[0]), int(p[1])), max(1, int(p[7] * (0.5 + 0.5 * k))), color, -1, cv2.LINE_AA)
            alive.append(p)
        self.particles = alive

        # Expanding shockwave rings.
        rings = []
        for x, y, t0, dur, r0, r1, color in self.rings:
            k = (now - t0) / dur
            if k >= 1:
                continue
            ease = 1 - (1 - k) ** 3
            glow_circle(img, (x, y), r0 + (r1 - r0) * ease, color, thick=max(1, int(5 * (1 - k))),
                        glow=10, strength=1 - k, core=False)
            rings.append([x, y, t0, dur, r0, r1, color])
        self.rings = rings

        # Pop-up text: pops in big, settles, floats up, fades out.
        popups = []
        for s, x, y, t0, dur, size, color, weight in self.popups:
            age = now - t0
            if age >= dur:
                continue
            pop = 1 + 0.7 * max(0.0, 1 - age / 0.15) ** 2
            fade = 1.0 if age < dur * 0.6 else 1 - (age - dur * 0.6) / (dur * 0.4)
            text(img, s, (x, y - 50 * age), size * pop, color, weight, opacity=max(0.0, fade))
            popups.append([s, x, y, t0, dur, size, color, weight])
        self.popups = popups

        # Red edge flash on a miss.
        k = (now - self.flash_t0) / self.flash_dur if self.flash_dur else 1
        if 0 <= k < 1:
            s = (1 - k) * 0.75
            wgt = self.vignette * s
            img[:] = cv2.blendLinear(img, self.red, 1 - wgt, wgt)


# ── opponent ─────────────────────────────────────────────────────────────────

TAUNTS = ["Too slow!", "Ha ha!", "Nope!", "Is that all?", "Point to me!", "Beep boop, missed!"]
SHOCKED = ["Whoa!", "Nice shot!", "What?!", "Huh?!"]
GRUMPY = ["Argh!", "No way!", "Lucky shot!", "Error 404: ball", "I slipped!"]
GLOAT = ["GG, human!", "Robots rule!", "Better luck next time!"]


def _smooth(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3 - 2 * k)


class Opponent:
    """LEGO-minifigure robot standing behind the far end of the table.

    It slides sideways to where your return lands, swings to hit the ball back
    (the next serve comes off its paddle), and reacts with faces + speech bubbles.
    """

    SWING_S = 0.35
    TAUNTS, SHOCKED, GRUMPY, GLOAT = TAUNTS, SHOCKED, GRUMPY, GLOAT

    def __init__(self, table, w, h):
        self.table = table
        self.w, self.h = w, h
        self.u = 0.038 * h  # size unit (minifig proportions are multiples of this)
        self.color = (0, 185, 255)
        self.name = "BOT"
        self.reset(w / 2)

    def reset(self, x=None, color=None, name=None):
        x = self.w / 2 if x is None else x
        self.x0 = self.x1 = x
        self.move_t0, self.move_dur = 0.0, 1.0
        self.swing_t0 = -10.0
        self.mood, self.mood_until = "idle", 0.0
        self.speech, self.speech_until = "", 0.0
        if color is not None:
            self.color = color
        if name is not None:
            self.name = name

    # ── commands ──

    def x_at(self, now):
        return self.x0 + (self.x1 - self.x0) * _smooth((now - self.move_t0) / self.move_dur)

    def move_to(self, x, now, dur):
        lo, hi = self.table.far_left - 0.04 * self.w, self.table.far_right + 0.04 * self.w
        self.x0, self.x1 = self.x_at(now), min(max(x, lo), hi)
        self.move_t0, self.move_dur = now, max(0.05, dur)

    def swing(self, now):
        self.swing_t0 = now

    def react(self, mood, now, dur=1.2, say=None):
        self.mood, self.mood_until = mood, now + dur
        if say:
            self.speech, self.speech_until = say, now + dur

    def taunt(self, now):
        self.react("happy", now, 1.3, random.choice(self.TAUNTS))

    def shocked(self, now, say=None):
        self.react("shocked", now, 1.0, say or random.choice(self.SHOCKED))

    def grumpy(self, now, dur=1.3, say=None):
        self.react("sad", now, dur, say or random.choice(self.GRUMPY))

    def gloat(self, now):
        self.react("happy", now, 99, random.choice(self.GLOAT))

    def _speech_bubble(self, img, now, x, by, gap):
        """Speech bubble level with by, `gap` px to the side of x (side with more room)."""
        if now >= self.speech_until or not self.speech:
            return
        size = 22
        tw = len(self.speech) * size * 0.55 + 24
        side = 1 if x < self.w / 2 else -1
        bx = x + side * (gap + 14 + tw / 2)
        panel(img, bx - tw / 2, by - 19, bx + tw / 2, by + 19, (250, 250, 250), 0.92, 14)
        ex0 = bx - side * tw / 2
        tri = np.array([[ex0, by - 8], [ex0, by + 8], [ex0 - side * 14, by + 2]], np.int32)
        cv2.fillPoly(img, [tri], (250, 250, 250), cv2.LINE_AA)
        text(img, self.speech, (bx, by), size, (40, 40, 40), "bold", stroke=0)

    # ── geometry ──

    def _pose(self, now, x=None):
        """Key points of the figure: body top, shoulders, hands, paddle."""
        u = self.u
        x = self.x_at(now) if x is None else x
        base = self.table.far_y + 4
        bob = math.sin(now * 3.0) * 1.5
        top = base - 2.3 * u + bob
        mood = self.mood if now < self.mood_until else "idle"
        sh_l, sh_r = (x - 1.15 * u, top + 0.35 * u), (x + 1.15 * u, top + 0.35 * u)

        k = (now - self.swing_t0) / self.SWING_S
        bump = math.sin(math.pi * k) if 0 <= k <= 1 else 0.0
        rest = (x + 2.0 * u, top + 0.5 * u)
        fwd = (x + 0.5 * u, top + 1.6 * u)
        hand_r = (rest[0] + (fwd[0] - rest[0]) * bump, rest[1] + (fwd[1] - rest[1]) * bump)
        hand_l = (x - 1.7 * u, top + 1.7 * u)
        if mood == "happy":  # arms up, waving
            wave = math.sin(now * 14) * 0.3 * u
            hand_l = (x - 1.8 * u + wave, top - 1.5 * u)
            hand_r = (x + 1.9 * u - wave, top - 1.3 * u)
        elif mood == "shocked":  # hands to the side of the head
            hand_l = (x - 1.6 * u, top - 0.8 * u)
        elif mood == "sad":  # slumped, paddle hanging down
            hand_l = (x - 1.4 * u, top + 2.0 * u)
            if bump == 0:
                hand_r = (x + 1.5 * u, top + 2.0 * u)

        dx, dy = hand_r[0] - sh_r[0], hand_r[1] - sh_r[1]
        n = math.hypot(dx, dy) or 1.0
        reach = 0.75 * (0.75 * u)  # same blade offset draw_paddle() uses for size 0.75u
        paddle = (hand_r[0] + dx / n * reach, hand_r[1] + dy / n * reach)
        return dict(x=x, base=base, top=top, mood=mood, sh_l=sh_l, sh_r=sh_r,
                    hand_l=hand_l, hand_r=hand_r, paddle=paddle)

    def paddle_pos(self, now, x=None):
        """Where the opponent's paddle is (or would be, standing at x)."""
        return self._pose(now, x)["paddle"]

    # ── drawing ──

    def draw(self, img, now):
        u = self.u
        p = self._pose(now)
        x, top, base, mood = p["x"], p["top"], p["base"], p["mood"]
        body = self.color
        dark = tuple(int(c * 0.6) for c in body)
        yellow, yellow_dark = (40, 215, 255), (20, 150, 200)

        def P(pt):
            return int(pt[0]), int(pt[1])

        # Left (free) arm behind the torso.
        cv2.line(img, P(p["sh_l"]), P(p["hand_l"]), dark, int(0.5 * u), cv2.LINE_AA)
        cv2.circle(img, P(p["hand_l"]), int(0.3 * u), yellow, -1, cv2.LINE_AA)

        # Torso (trapezoid) with a chest print.
        torso = np.array([[x - 1.1 * u, top], [x + 1.1 * u, top], [x + 1.35 * u, base], [x - 1.35 * u, base]],
                         np.int32)
        cv2.fillPoly(img, [torso], body, cv2.LINE_AA)
        cv2.polylines(img, [torso], True, dark, 2, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(top + 1.1 * u)), int(0.45 * u), WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(top + 1.1 * u)), int(0.45 * u), dark, 2, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(top + 1.1 * u)), int(0.15 * u), body, -1, cv2.LINE_AA)

        # Neck + head + stud.
        cv2.rectangle(img, (int(x - 0.35 * u), int(top - 0.3 * u)), (int(x + 0.35 * u), int(top + 2)),
                      yellow_dark, -1)
        hx0, hy0, hx1, hy1 = int(x - 0.8 * u), int(top - 1.75 * u), int(x + 0.8 * u), int(top - 0.25 * u)
        _rounded(img, hx0, hy0, hx1, hy1, int(0.35 * u), yellow)
        _rounded(img, hx0, hy0, hx1, hy1, int(0.35 * u), yellow_dark, 2)
        cv2.rectangle(img, (int(x - 0.42 * u), int(hy0 - 0.35 * u)), (int(x + 0.42 * u), hy0 + 2), yellow, -1)
        cv2.rectangle(img, (int(x - 0.42 * u), int(hy0 - 0.35 * u)), (int(x + 0.42 * u), hy0 + 2),
                      yellow_dark, 2)

        # Face.
        cy = (hy0 + hy1) / 2
        ex, ey, er = 0.32 * u, cy - 0.15 * u, max(2, int(0.12 * u))
        ink = (20, 20, 20)
        if mood == "happy":  # ^ ^ eyes, big open grin
            for s in (-1, 1):
                cv2.ellipse(img, (int(x + s * ex), int(ey + 0.05 * u)), (er + 2, er + 2), 0, 180, 360, ink, 2,
                            cv2.LINE_AA)
            cv2.ellipse(img, (int(x), int(cy + 0.2 * u)), (int(0.4 * u), int(0.3 * u)), 0, 0, 180, ink, -1,
                        cv2.LINE_AA)
        elif mood == "shocked":  # wide eyes, O mouth
            for s in (-1, 1):
                cv2.circle(img, (int(x + s * ex), int(ey)), er + 2, WHITE, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x + s * ex), int(ey)), er, ink, -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(cy + 0.35 * u)), int(0.17 * u), ink, 2, cv2.LINE_AA)
        elif mood == "sad":  # flat eyes, frown
            for s in (-1, 1):
                cv2.line(img, (int(x + s * ex - er), int(ey)), (int(x + s * ex + er), int(ey)), ink, 2,
                         cv2.LINE_AA)
            cv2.ellipse(img, (int(x), int(cy + 0.5 * u)), (int(0.3 * u), int(0.2 * u)), 0, 200, 340, ink, 2,
                        cv2.LINE_AA)
        else:  # calm smile
            for s in (-1, 1):
                cv2.circle(img, (int(x + s * ex), int(ey)), er, ink, -1, cv2.LINE_AA)
            cv2.ellipse(img, (int(x), int(cy + 0.1 * u)), (int(0.35 * u), int(0.25 * u)), 0, 20, 160, ink, 2,
                        cv2.LINE_AA)

        # Paddle arm in front.
        cv2.line(img, P(p["sh_r"]), P(p["hand_r"]), dark, int(0.5 * u), cv2.LINE_AA)
        draw_paddle(img, p["hand_r"], p["sh_r"], 0.75 * u)
        cv2.circle(img, P(p["hand_r"]), int(0.3 * u), yellow, -1, cv2.LINE_AA)

        self._speech_bubble(img, now, x, (hy0 + hy1) / 2 - 4, 1.2 * u)


class PhotoOpponent(Opponent):
    """Opponent drawn from a cut-out photo (PNG with a transparent background).

    Same behaviour as the robot (moves, swings, serves off its paddle, speech
    bubbles); since a photo can't change its face, moods are shown with motion:
    swing = lunge forward, happy = hop, shocked = shake + zoom, sad = tilt + grey.
    """

    TAUNTS = ["Did you calibrate that?", "Check your IMU!", "Too slow!", "No frisbee for you!",
              "Office hours are open!", "Swing harder!"]
    SHOCKED = ["Impressive!", "Now THAT's robotics!", "A+ shot!", "Whoa!"]
    GRUMPY = ["Hmm. Lucky.", "Bad sensor data!", "My network lagged!", "Rematch!", "Noisy IMU..."]
    GLOAT = ["Class dismissed!", "See you in office hours!", "Back to the lab!"]

    def __init__(self, table, w, h, image_path, paddle_xy, height_frac=0.26, sink_frac=0.06,
                 head_frac=0.25, face_box=(0.3, 0.0, 0.75, 0.5), lines=None):
        """paddle_xy: paddle centre in the ORIGINAL image (pixels).
        height_frac: sprite height as a fraction of the screen height.
        sink_frac: how far below the table's far edge the bottom of the photo sits
                   (taller / full-body photos sink further so their head stays on screen).
        head_frac: where the face is, as a fraction of the (cropped) photo height -- for the speech bubble.
        face_box: (x0, y0, x1, y1) fractions of the cropped photo around the face -- for the menu thumbnail.
        lines: optional {"taunts": [...], "shocked": [...], "grumpy": [...], "gloat": [...]}."""
        super().__init__(table, w, h)
        self.sink_frac = sink_frac
        for key, attr in (("taunts", "TAUNTS"), ("shocked", "SHOCKED"), ("grumpy", "GRUMPY"), ("gloat", "GLOAT")):
            if lines and lines.get(key):
                setattr(self, attr, list(lines[key]))
        rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"{image_path} must be a PNG with a transparent background")
        ys, xs = np.nonzero(rgba[:, :, 3] > 10)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        rgba = rgba[y0:y1, x0:x1]
        s = height_frac * h / rgba.shape[0]
        rgba = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.bgr = rgba[:, :, :3].astype(np.float32)
        self.alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        sh, sw = self.alpha.shape
        self.anchor = (sw / 2, sh)                       # bottom-centre of the sprite
        self.paddle_sprite = ((paddle_xy[0] - x0) * s, (paddle_xy[1] - y0) * s)
        self.head_y = head_frac * sh                     # roughly where the face is
        # Small round portrait of the face for the menu cards.
        fx0, fy0, fx1, fy1 = face_box
        self.portrait_src = rgba[int(sh * fy0): int(sh * fy1), int(sw * fx0): int(sw * fx1)]

    def _transform(self, now, x=None):
        """2x3 affine matrix: sprite pixel -> screen pixel, plus extra info."""
        x = self.x_at(now) if x is None else x
        base = self.table.far_y + self.sink_frac * self.h  # table hides the bottom of the photo
        mood = self.mood if now < self.mood_until else "idle"
        k = (now - self.swing_t0) / self.SWING_S
        bump = math.sin(math.pi * k) if 0 <= k <= 1 else 0.0
        angle = -7 * bump               # lunge into the shot
        scale = 1 + 0.07 * bump
        dy = math.sin(now * 3.0) * 2 + 0.02 * self.h * bump
        dx = 0.0
        if mood == "happy":
            dy -= abs(math.sin(now * 9)) * 0.03 * self.h
        elif mood == "shocked":
            dx += random.uniform(-3, 3)
            scale *= 1.05
        elif mood == "sad":
            angle += 6
            dy += 0.01 * self.h
        m = cv2.getRotationMatrix2D(self.anchor, angle, scale)
        m[0, 2] += x + dx - self.anchor[0]
        m[1, 2] += base + dy - self.anchor[1]
        return m, mood

    def paddle_pos(self, now, x=None):
        m, _ = self._transform(now, x)
        px, py = self.paddle_sprite
        return (m[0, 0] * px + m[0, 1] * py + m[0, 2], m[1, 0] * px + m[1, 1] * py + m[1, 2])

    def draw(self, img, now):
        m, mood = self._transform(now)
        sh, sw = self.alpha.shape
        corners = np.array([[0, 0, 1], [sw, 0, 1], [0, sh, 1], [sw, sh, 1]], np.float32) @ m.T
        H, W = img.shape[:2]
        x0, y0 = max(0, int(corners[:, 0].min()) - 1), max(0, int(corners[:, 1].min()) - 1)
        x1, y1 = min(W, int(corners[:, 0].max()) + 2), min(H, int(corners[:, 1].max()) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        mr = m.copy()
        mr[0, 2] -= x0
        mr[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        bgr = cv2.warpAffine(self.bgr, mr, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        a = cv2.warpAffine(self.alpha, mr, size, flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        if mood == "sad":  # a bit grey and gloomy
            grey = bgr.mean(axis=2, keepdims=True)
            bgr = (0.5 * bgr + 0.5 * grey) * 0.8
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (roi * (1 - a) + bgr * a).astype(np.uint8)

        hx = m[0, 0] * self.anchor[0] + m[0, 1] * self.head_y + m[0, 2]
        hy = m[1, 0] * self.anchor[0] + m[1, 1] * self.head_y + m[1, 2]
        self._speech_bubble(img, now, hx, max(30, hy), sw * 0.35)

    def portrait(self, size):
        """(bgr float, alpha float HxWx1) round thumbnail of the face, size x size px."""
        face = self.portrait_src
        s = size / max(face.shape[:2])
        face = cv2.resize(face, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 4), np.uint8)
        fy, fx = (size - face.shape[0]) // 2, (size - face.shape[1]) // 2
        canvas[fy:fy + face.shape[0], fx:fx + face.shape[1]] = face
        mask = np.zeros((size, size), np.float32)
        cv2.circle(mask, (size // 2, size // 2), size // 2 - 1, 1.0, -1, cv2.LINE_AA)
        bg = np.full((size, size, 3), (60, 60, 200), np.float32)  # red disc behind him
        a = canvas[:, :, 3:].astype(np.float32) / 255.0
        bgr = canvas[:, :, :3].astype(np.float32) * a + bg * (1 - a)
        return bgr, mask[..., None]
