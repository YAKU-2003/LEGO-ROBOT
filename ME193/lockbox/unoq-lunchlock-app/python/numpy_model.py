"""
The finger-counting network as plain NumPy, for the UNO Q (no PyTorch on the board).

Exactly the same maths as finger_model.FingerCNN in eval mode:
  colour crop -> hand mask (skin redness vs background) + normalised grayscale
  -> [conv 5x5 16 -> ReLU -> maxpool] -> [conv 3x3 32 -> ReLU -> maxpool] -> [conv 3x3 64 -> ReLU -> maxpool]
  -> flatten (9216) + 12x12 grid + 24-column finger profile (168) -> linear -> 6 scores -> softmax

BatchNorm layers are already folded into the conv / grid weights by export_for_unoq.py,
so here every conv is just  output = weights * input + bias.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IMG = 96
GRID = 12
PROFILE_COLS = 24


class FingerNet:
    def __init__(self, weights_path):
        w = np.load(weights_path)
        self.convs = [(w[f"conv{i}_w"], w[f"conv{i}_b"]) for i in range(3)]
        self.grid_scale, self.grid_shift = w["grid_scale"], w["grid_shift"]
        self.fc_w, self.fc_b = w["fc_w"], w["fc_b"]

    # ---------------------------------------------------------------- building blocks
    @staticmethod
    def conv_same(x, wt, b):
        """x: C x H x W, wt: O x C x k x k  ->  O x H x W (zero padding, stride 1)."""
        k = wt.shape[-1]
        p = k // 2
        xp = np.pad(x, ((0, 0), (p, p), (p, p)))
        cols = sliding_window_view(xp, (k, k), axis=(1, 2))           # C x H x W x k x k
        C, H, W = x.shape
        cols = cols.transpose(1, 2, 0, 3, 4).reshape(H * W, C * k * k)  # every k x k patch as a row
        out = cols @ wt.reshape(wt.shape[0], -1).T + b                  # (H*W) x O
        return out.T.reshape(-1, H, W)

    @staticmethod
    def maxpool2(x):
        C, H, W = x.shape
        return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))

    @staticmethod
    def skin_mask(rgb):
        """rgb: 3 x H x W in [0,1] -> H x W hand mask in [0,1] (same as finger_model.skin_mask)."""
        r, g, b = rgb
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (0.5 * r - 0.4187 * g - 0.0813 * b) / (y + 0.05)
        flat = cr.ravel()
        bg = np.partition(flat, (flat.size - 1) // 2)[(flat.size - 1) // 2]   # torch.median = lower median
        m = np.clip((cr - bg) / 0.045, 0, 1)
        mp = np.pad(m, 1)
        return sliding_window_view(mp, (3, 3)).mean(axis=(2, 3))             # 3x3 average, zero padded

    @staticmethod
    def grid_features(mask):
        H = mask.shape[0]
        density = mask.reshape(GRID, H // GRID, GRID, H // GRID).mean(axis=(1, 3)).ravel()
        s = H // PROFILE_COLS
        cells = mask.reshape(PROFILE_COLS, s, PROFILE_COLS, s).mean(axis=(1, 3))   # rows x cols
        filled = cells > 0.5
        rows = np.arange(PROFILE_COLS)[:, None]
        top = np.where(filled, rows, PROFILE_COLS).min(axis=0)
        profile = 1 - top / PROFILE_COLS
        return np.concatenate([density, profile])

    # ---------------------------------------------------------------- full network
    def preprocess(self, bgr_crop):
        small = cv2.resize(bgr_crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0

    def scores(self, rgb):
        mask = self.skin_mask(rgb)
        gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        gray = (gray - gray.mean()) / (gray.std(ddof=1) + 1e-3)          # lighting fix
        x = np.stack([gray, mask * 2 - 1]).astype(np.float32)           # early fusion: 2 channels
        for wt, b in self.convs:
            x = self.maxpool2(np.maximum(self.conv_same(x, wt, b), 0))  # conv -> ReLU -> pool
        grid = self.grid_features(mask) * self.grid_scale + self.grid_shift
        feats = np.concatenate([x.ravel(), grid])
        return self.fc_w @ feats + self.fc_b

    def predict(self, bgr_crop):
        """Returns (finger_count, confidence, all 6 probabilities)."""
        s = self.scores(self.preprocess(bgr_crop))
        p = np.exp(s - s.max())
        p /= p.sum()
        k = int(p.argmax())
        return k, float(p[k]), p
