"""
Train the finger-counting CNN on ~/Downloads/gestures/<fingers>_<n>.jpg

  python train_cnn.py              (25 epochs, CPU is fine: a few minutes)
  python train_cnn.py --epochs 40

Validation split: the LAST 20% of every burst (e.g. 3_161..3_200) is held out instead of a
random 20%. Photos taken 0.2 s apart are near-duplicates, so a random split would let the
model "cheat" by memorising neighbours and report a falsely high accuracy.

Outputs (next to this file):
  finger_cnn.pt            trained weights
  training_curves.png      loss + accuracy vs epoch
  confusion_matrix.png     which finger counts get mixed up
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from finger_model import DATA_DIR, IMG_SIZE, MODEL_FILE, NUM_CLASSES, FingerCNN, to_rgb_array

OUT = Path(__file__).parent
torch.manual_seed(0)
# Train on the NVIDIA GPU when PyTorch can see it (python train_cnn.py --cpu forces the CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def load_images(val_frac=0.2):
    """Load every image once into memory as 96x96 grayscale, split per class by shot number."""
    train, val = [], []
    pat = re.compile(r"^(\d)_(\d+)\.jpg$")
    ex_file = OUT / "exclude.txt"          # bad photos (no hand in view) listed one stem per line
    exclude = {l.strip() for l in ex_file.read_text().splitlines()
               if l.strip() and not l.startswith("#")} if ex_file.exists() else set()
    for cls in range(NUM_CLASSES):
        files = sorted((p for p in DATA_DIR.glob(f"{cls}_*.jpg") if pat.match(p.name) and p.stem not in exclude),
                       key=lambda p: int(pat.match(p.name).group(2)))
        n_val = int(len(files) * val_frac)
        for i, p in enumerate(files):
            img = to_rgb_array(cv2.imread(str(p)))          # 96 x 96 x 3 colour (grid branch needs colour)
            (val if i >= len(files) - n_val else train).append((img, cls))
        print(f"  class {cls}: {len(files) - n_val} train / {n_val} val")
    return train, val


def random_lighting(x):
    """Simulate lighting the camera never saw during capture (x is 1 x H x W in [0, 1])."""
    if torch.rand(1) < 0.7:                                    # gamma: crushes or lifts shadows
        x = x.clamp(1e-4, 1) ** float(torch.empty(1).uniform_(0.5, 2.0))
    if torch.rand(1) < 0.5:                                    # light from one side: brightness ramp
        H, W = x.shape[-2:]
        ang = float(torch.rand(1)) * 2 * np.pi
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        ramp = (np.cos(ang) * xx + np.sin(ang) * yy) / 2 + 0.5   # 0..1 across the image
        lo = float(torch.empty(1).uniform_(0.35, 0.85))
        x = x * (lo + (1 - lo) * ramp)
    if torch.rand(1) < 0.4:                                    # sensor noise (dim room = grainy)
        x = x + torch.randn_like(x) * float(torch.empty(1).uniform_(0.01, 0.05))
    return x.clamp(0, 1)


def lighting_report(model, val_ds):
    """Accuracy on the validation photos re-lit in ways the camera never saw."""
    tests = {
        "as captured": lambda x: x,
        "dark (x0.4)": lambda x: x * 0.4,
        "bright (+0.25)": lambda x: (x + 0.25).clamp(0, 1),
        "low contrast": lambda x: 0.5 + (x - 0.5) * 0.4,
        "side light": lambda x: x * (0.4 + 0.6 * torch.linspace(0, 1, x.shape[-1])),
        "gamma 2.0 (dim)": lambda x: x ** 2.0,
    }
    model.eval()
    out = {}
    with torch.no_grad():
        for name, f in tests.items():
            pred = model(f(val_ds.x).to(DEVICE)).argmax(1).cpu()
            out[name] = (pred == val_ds.y).float().mean().item()
    return out


class GestureSet(Dataset):
    def __init__(self, items, augment):
        self.x = torch.from_numpy(np.stack([i for i, _ in items]).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        self.y = torch.tensor([c for _, c in items])
        # Data augmentation: every epoch the network sees slightly different versions
        # of each photo -> effectively a much bigger dataset, and robustness to how
        # other people hold their hands.
        self.tf = v2.Compose([
            v2.RandomHorizontalFlip(),                       # left hand vs right hand
            v2.RandomAffine(degrees=20, translate=(0.12, 0.12), scale=(0.85, 1.15)),
            v2.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.3),  # brighter / darker room (no hue: skin colour matters)
            v2.Lambda(random_lighting),                      # gamma, side light, noise
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.3, 1.5))], p=0.3),  # focus / motion blur
        ]) if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.tf is not None:
            x = self.tf(x)
        return x, self.y[i]


def run_epoch(model, loader, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    preds = []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            scores = model(x)                 # forward pass
            loss = loss_fn(scores, y)         # cross-entropy: how wrong were we?
            if training:
                opt.zero_grad()
                loss.backward()               # back-propagation: d(loss)/d(every weight)
                opt.step()                    # gradient-descent step on every weight
            total_loss += loss.item() * len(y)
            p = scores.argmax(1)
            correct += (p == y).sum().item()
            n += len(y)
            preds.append(p.cpu())
    return total_loss / n, correct / n, torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--branch", choices=["early", "both", "cnn", "grid"], default="early",
                    help="early = mask as 2nd CNN channel + grid (default); both = CNN + grid joined at the end; "
                         "cnn / grid = only one branch, for comparison")
    ap.add_argument("--out", default=str(MODEL_FILE), help="where to save the model")
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available")
    args = ap.parse_args()
    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")
    print(f"Training on: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}")

    print(f"Loading images from {DATA_DIR}")
    train_items, val_items = load_images()
    train_ds, val_ds = GestureSet(train_items, augment=True), GestureSet(val_items, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128)

    config = {"use_cnn": args.branch in ("early", "both", "cnn"), "use_grid": args.branch in ("early", "both", "grid"),
              "early_fusion": args.branch == "early"}
    model = FingerCNN(**config).to(DEVICE)
    print(f"Branches: {args.branch}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} trainable weights")

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta, _ = run_epoch(model, train_dl, loss_fn, opt)
        vl, va, _ = run_epoch(model, val_dl, loss_fn)
        sched.step()
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if va >= best_acc:
            best_acc, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}  train loss {tl:.3f} acc {ta:.1%}   val loss {vl:.3f} acc {va:.1%}"
              f"   ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "img_size": IMG_SIZE, "val_acc": best_acc, "config": config}, args.out)
    print(f"\nBest val accuracy {best_acc:.1%} -> saved {Path(args.out).name}")

    # ---- plots ----  (a comparison model saved elsewhere gets its own file names)
    tag = "" if Path(args.out).resolve() == MODEL_FILE.resolve() else Path(args.out).stem + "_"
    ep = range(1, args.epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["train_loss"], label="train"); a1.plot(ep, hist["val_loss"], label="validation")
    a1.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss"); a1.legend()
    a2.plot(ep, hist["train_acc"], label="train"); a2.plot(ep, hist["val_acc"], label="validation")
    a2.set(title="Accuracy", xlabel="epoch", ylabel="fraction correct", ylim=(0, 1.02)); a2.legend()
    fig.tight_layout(); fig.savefig(OUT / f"{tag}training_curves.png", dpi=130)

    _, _, preds = run_epoch(model, val_dl, loss_fn)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for t, p in zip(val_ds.y.tolist(), preds.tolist()):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(6), yticks=range(6), xlabel="predicted fingers", ylabel="true fingers",
           title=f"Validation confusion matrix ({best_acc:.1%})")
    fig.tight_layout(); fig.savefig(OUT / f"{tag}confusion_matrix.png", dpi=130)
    print("Per-class val accuracy:",
          "  ".join(f"{c}:{cm[c, c] / cm[c].sum():.0%}" for c in range(NUM_CLASSES)))
    print("Lighting robustness (val photos re-lit):")
    light = lighting_report(model, val_ds)
    for name, acc in light.items():
        print(f"  {name:16s} {acc:.1%}")

    report = {"branch": args.branch, "val_acc": best_acc, "epochs": args.epochs, "params": n_params,
              "per_class": [cm[c, c] / max(1, cm[c].sum()) for c in range(NUM_CLASSES)],
              "confusion": cm.tolist(), "lighting": light,
              "history": hist, "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    (OUT / f"{tag}model_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
