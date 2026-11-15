"""
Export the trained model (finger_cnn.pt) for the UNO Q and prove the NumPy version agrees.

  python export_for_unoq.py

- folds every BatchNorm into the layer before it (fewer operations on the board)
- saves unoq-lunchlock-app/python/finger_weights.npz  (+ roi.json and the lock code)
- runs PyTorch and the NumPy network side by side on the held-out photos and reports
  how often they pick the same finger count and the biggest probability difference
- times the NumPy network (this laptop; the UNO Q will be slower)
"""

import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import finger_model as fm
import train_cnn as T
from lock_logic import load_code

HERE = Path(__file__).parent
APP_PY = HERE / "unoq-lunchlock-app" / "python"
sys.path.insert(0, str(APP_PY))
from numpy_model import FingerNet  # noqa: E402

EPS = 1e-5   # PyTorch BatchNorm default


def fold(conv_w, conv_b, bn):
    s = bn["weight"] / torch.sqrt(bn["running_var"] + EPS)
    return (conv_w * s[:, None, None, None]).numpy(), ((conv_b - bn["running_mean"]) * s + bn["bias"]).numpy()


def main():
    model = fm.load_model()
    if not (model.early_fusion and model.use_grid and model.use_cnn):
        raise SystemExit("export expects the early-fusion CNN + grid model")
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    bn = lambda p: {k: sd[f"{p}.{k}"] for k in ("weight", "bias", "running_mean", "running_var")}

    out = {}
    for i, (c, b) in enumerate([("features.0", "features.1"), ("features.4", "features.5"), ("features.8", "features.9")]):
        out[f"conv{i}_w"], out[f"conv{i}_b"] = fold(sd[f"{c}.weight"], sd[f"{c}.bias"], bn(b))
    g = bn("grid_norm")
    s = g["weight"] / torch.sqrt(g["running_var"] + EPS)
    out["grid_scale"], out["grid_shift"] = s.numpy(), (g["bias"] - g["running_mean"] * s).numpy()
    out["fc_w"], out["fc_b"] = sd["fc.weight"].numpy(), sd["fc.bias"].numpy()
    out = {k: v.astype(np.float32) for k, v in out.items()}

    APP_PY.mkdir(parents=True, exist_ok=True)
    np.savez(APP_PY / "finger_weights.npz", **out)
    shutil.copy(fm.ROI_FILE, APP_PY / "roi.json")
    (APP_PY / "lock_config.json").write_text(json.dumps({"code": list(load_code())}))
    shutil.copy(HERE / "lock_logic.py", APP_PY / "lock_logic.py")      # same lock policy on the board
    size = (APP_PY / "finger_weights.npz").stat().st_size / 1024
    print(f"saved finger_weights.npz ({size:.0f} kB), roi.json, lock_config.json, lock_logic.py -> {APP_PY}")

    # ---- check NumPy == PyTorch on the held-out photos (raw files, full preprocessing path)
    net = FingerNet(APP_PY / "finger_weights.npz")
    with contextlib.redirect_stdout(io.StringIO()):
        T.load_images()                                   # only to reuse its file ordering / exclude list
    files = []
    for cls in range(6):
        fs = sorted(fm.DATA_DIR.glob(f"{cls}_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
        files += fs[-40:]                                 # newest 40 photos of each class
    same, max_diff, t_np = 0, 0.0, []
    for p in files:
        img = cv2.imread(str(p))
        k_t, _, p_t = fm.predict(model, img)
        t0 = time.perf_counter()
        k_n, _, p_n = net.predict(img)
        t_np.append(time.perf_counter() - t0)
        same += k_t == k_n
        max_diff = max(max_diff, float(np.abs(p_t - p_n).max()))
    print(f"NumPy vs PyTorch on {len(files)} photos: same answer {same}/{len(files)}, "
          f"largest probability difference {max_diff:.2e}")
    print(f"NumPy network on this laptop: {1000 * np.median(t_np):.1f} ms per frame")


if __name__ == "__main__":
    main()
