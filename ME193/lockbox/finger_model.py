"""
Shared pieces for the finger-counting lock: the network and the image preprocessing.

Training (train_cnn.py), the dashboard (app.py), lock_pc.py / live_test.py and later the
UNO Q all import from here, so every stage sees images prepared in exactly the same way.

ONE model, two ways of looking at the same 96x96 colour image:

  1. CNN branch  (learned features)
       grayscale -> normalise brightness/contrast
       conv 5x5 (16) -> ReLU -> maxpool   96 -> 48
       conv 3x3 (32) -> ReLU -> maxpool   48 -> 24
       conv 3x3 (64) -> ReLU -> maxpool   24 -> 12      => 64 x 12 x 12 = 9216 numbers

  2. Grid branch (hand-designed features, no learning)
       skin mask  = how much redder (Cr channel) each pixel is than the image's background
       12 x 12 grid: fraction of each cell covered by hand                 => 144 numbers
       finger profile: for each of 24 columns, how high the hand reaches    =>  24 numbers
       (raised fingers = tall narrow bumps in the profile)

  Both are joined into one long vector -> dropout -> ONE linear layer -> 6 scores (0-5 fingers),
  trained together with cross-entropy loss.

  EARLY FUSION (default): the skin mask is ALSO fed into the CNN as a 2nd input channel, so the
  very first conv filters see "photo" and "where the hand is" side by side:
       input 2 x 96 x 96  =  [ grayscale photo , hand mask ]
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 96            # 360x360 crop -> 96x96 colour
NUM_CLASSES = 6          # 0..5 fingers
GRID = 12                # hand-density grid is GRID x GRID cells
PROFILE_COLS = 24        # finger-height profile resolution
DATA_DIR = Path.home() / "Downloads" / "gestures"
ROI_FILE = DATA_DIR / "roi.json"
MODEL_FILE = Path(__file__).with_name("finger_cnn.pt")


def skin_mask(rgb):
    """
    rgb: N x 3 x H x W in [0, 1]  ->  N x 1 x H x W in [0, 1] (1 = hand).

    Cr = "how red" a pixel is, independent of how bright it is. Skin is redder than the
    wall/desk, so pixels whose Cr is above the image's median (the background - the hand
    covers less than half the box) are hand. Being relative to the image's own median makes
    it tolerant to lighting and white balance changes.
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (0.5 * r - 0.4187 * g - 0.0813 * b) / (y + 0.05)      # red-difference, divided by brightness
    bg = cr.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    m = ((cr - bg) / 0.045).clamp(0, 1)                        # soft threshold a bit redder than background
    return F.avg_pool2d(m, 3, stride=1, padding=1)             # smooth away speckle


def grid_features(mask):
    """mask N x 1 x H x W -> N x (GRID*GRID + PROFILE_COLS) hand-designed features."""
    density = F.adaptive_avg_pool2d(mask, GRID).flatten(1)               # 12x12 grid of hand coverage
    cols = F.adaptive_avg_pool2d(mask, (PROFILE_COLS, PROFILE_COLS))[:, 0]  # N x 24 rows x 24 cols
    filled = (cols > 0.5).float()
    rows = torch.arange(PROFILE_COLS, device=mask.device, dtype=mask.dtype).view(1, -1, 1)
    # first filled row from the top (24 if column empty) -> height 0..1 of the hand in that column
    top = torch.where(filled > 0, rows, torch.full_like(filled, PROFILE_COLS)).amin(dim=1)
    profile = 1 - top / PROFILE_COLS
    return torch.cat([density, profile], dim=1)


class FingerCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, use_cnn=True, use_grid=True, early_fusion=False):
        super().__init__()
        self.use_cnn, self.use_grid, self.early_fusion = use_cnn, use_grid, early_fusion
        self.features = nn.Sequential(
            nn.Conv2d(2 if early_fusion else 1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        ) if use_cnn else None
        n_cnn = 64 * (IMG_SIZE // 8) ** 2 if use_cnn else 0
        n_grid = GRID * GRID + PROFILE_COLS if use_grid else 0
        self.grid_norm = nn.BatchNorm1d(n_grid) if use_grid else None   # put grid numbers on the same scale
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(n_cnn + n_grid, num_classes)

    def branches(self, rgb):
        parts = {}
        mask = skin_mask(rgb) if (self.use_grid or self.early_fusion) else None
        if self.use_cnn:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            # lighting fix: every image rescaled to mean 0 / std 1 before the convolutions
            gray = (gray - gray.mean(dim=(1, 2, 3), keepdim=True)) / (gray.std(dim=(1, 2, 3), keepdim=True) + 1e-3)
            x = torch.cat([gray, mask * 2 - 1], dim=1) if self.early_fusion else gray   # early fusion
            parts["cnn"] = self.features(x).flatten(1)
        if self.use_grid:
            parts["grid"] = self.grid_norm(grid_features(mask))
        return parts

    def forward(self, rgb):
        return self.fc(self.dropout(torch.cat(list(self.branches(rgb).values()), dim=1)))


def load_roi():
    return tuple(json.loads(ROI_FILE.read_text()))


def to_rgb_array(bgr):
    """BGR image (any size) -> 96 x 96 x 3 uint8 RGB."""
    return cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def preprocess(bgr_crop):
    """BGR image (any size) -> float tensor 1 x 3 x 96 x 96 in [0, 1]."""
    return torch.from_numpy(to_rgb_array(bgr_crop).astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def load_model(path=MODEL_FILE):
    ckpt = torch.load(path, map_location="cpu")
    model = FingerCNN(**ckpt.get("config", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, bgr_crop):
    """Returns (finger_count, confidence, all 6 probabilities)."""
    probs = torch.softmax(model(preprocess(bgr_crop)), dim=1)[0]
    k = int(probs.argmax())
    return k, float(probs[k]), probs.numpy()


@torch.no_grad()
def explain(bgr_crop):
    """Skin mask + grid for display (dashboard 'grid view'). Returns mask HxW, density GxG, profile."""
    m = skin_mask(preprocess(bgr_crop))
    f = grid_features(m)[0]
    return m[0, 0].numpy(), f[:GRID * GRID].view(GRID, GRID).numpy(), f[GRID * GRID:].numpy()
