"""
How much does the grid actually help?  (ablation test)

Takes the trained model (finger_cnn.pt) and the held-out photos, then switches parts of the
network OFF one at a time and measures accuracy again:

  full model            everything on
  no mask channel       the hand mask fed into the CNN is replaced by a flat "don't know" image
  no grid features      the 144 grid + 24 profile numbers are replaced by their average value
  CNN photo only        both of the above off -> only the grayscale photo is left

A big drop = the network really leans on that part.  Also runs the lighting test for each.

  python ablation.py
"""

import contextlib
import io

import torch

import finger_model as fm
import train_cnn as T


def run(model, x, y, mask_on, grid_on):
    def branches(rgb):
        parts = {}
        mask = fm.skin_mask(rgb)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gray = (gray - gray.mean(dim=(1, 2, 3), keepdim=True)) / (gray.std(dim=(1, 2, 3), keepdim=True) + 1e-3)
        m = mask * 2 - 1 if mask_on else torch.zeros_like(mask)        # 0 = halfway, carries no info
        parts["cnn"] = model.features(torch.cat([gray, m], dim=1)).flatten(1)
        g = model.grid_norm(fm.grid_features(mask))
        parts["grid"] = g if grid_on else torch.zeros_like(g)          # 0 = the average after BatchNorm
        return parts

    with torch.no_grad():
        scores = model.fc(model.dropout(torch.cat(list(branches(x).values()), dim=1)))
    return (scores.argmax(1) == y).float().mean().item()


def main():
    model = fm.load_model()
    if not model.early_fusion or not model.use_grid:
        raise SystemExit("finger_cnn.pt is not the early-fusion + grid model.")
    with contextlib.redirect_stdout(io.StringIO()):
        _, val_items = T.load_images()
    val = T.GestureSet(val_items, augment=False)

    lights = {"as captured": lambda x: x, "dark (x0.4)": lambda x: x * 0.4,
              "low contrast": lambda x: 0.5 + (x - 0.5) * 0.4,
              "side light": lambda x: x * (0.4 + 0.6 * torch.linspace(0, 1, x.shape[-1]))}
    setups = {"full model": (True, True), "no mask channel": (False, True),
              "no grid features": (True, False), "CNN photo only": (False, False)}

    print(f"{len(val)} held-out photos\n")
    print(f"{'':18s}" + "".join(f"{k:>15s}" for k in lights))
    for name, (m_on, g_on) in setups.items():
        accs = [run(model, f(val.x), val.y, m_on, g_on) for f in lights.values()]
        print(f"{name:18s}" + "".join(f"{a:15.1%}" for a in accs))


if __name__ == "__main__":
    main()
