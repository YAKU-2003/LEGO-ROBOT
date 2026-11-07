"""The PyTorch training loop, shared by pretrain.py (public data) and train_steps.py (your data).

A "recording" here is a dict with:
    X       [n, 6]  accel (m/s^2) + gyro (rad/s) on the 50 Hz grid
    inp     [n, 8]  steplib.net_inputs(X)
    ok      [n]     False inside Wi-Fi gaps
    target  [n]     step density to learn (bumps at the labelled steps), or None if only the total is known
    steps   int     how many steps the recording has
"""

import numpy as np
import torch

import steplib as sl

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CROP = int(8 * sl.FS)       # training crops are 8 s
EDGE = int(0.5 * sl.FS)     # the first/last 0.5 s of a crop has no context: not scored


def random_rotation(rng, max_deg=25):
    """Small random 3-D rotation: the same motion seen by a sensor strapped on a bit differently."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K).astype(np.float32)


def augment(x, y, rng, max_deg=25):
    """x [n,8], y [n]. Rotate the sensor, scale the signal, stretch time (= walk a bit faster/slower)."""
    R = random_rotation(rng, max_deg)
    x = x.copy()
    x[:, 0:3] = x[:, 0:3] @ R.T
    x[:, 3:6] = x[:, 3:6] @ R.T
    x[:, 0:6] *= rng.uniform(0.9, 1.1)
    f = rng.uniform(0.88, 1.12)
    n = len(x)
    src = np.clip(np.arange(n) * f, 0, n - 1)
    x = np.column_stack([np.interp(src, np.arange(n), x[:, i]) for i in range(x.shape[1])])
    y = np.interp(src, np.arange(n), y) * f  # area of each bump stays 1 step
    x[:, 6] = np.linalg.norm(x[:, 0:3], axis=1) - 1.0
    x[:, 7] = np.linalg.norm(x[:, 3:6], axis=1)
    x += rng.normal(0, 0.01, x.shape)
    return x.astype(np.float32), y.astype(np.float32)


def make_batch(pool, rng, n, max_deg=25):
    weights = np.array([len(r["X"]) * r.get("weight", 1.0) for r in pool], float)
    xs, ys = [], []
    for r in rng.choice(pool, size=n, p=weights / weights.sum()):
        L = len(r["X"])
        for _ in range(5):  # avoid crops with a Wi-Fi gap
            s = int(rng.integers(0, max(1, L - CROP)))
            if r["ok"][s:s + CROP].all():
                break
        x, y = r["inp"][s:s + CROP], r["target"][s:s + CROP]
        if len(x) < CROP:
            x = np.pad(x, ((0, CROP - len(x)), (0, 0)), mode="edge")
            y = np.pad(y, (0, CROP - len(y)))
        x, y = augment(x, y, rng, max_deg)
        xs.append(x.T)
        ys.append(y)
    return torch.from_numpy(np.stack(xs)).to(DEV), torch.from_numpy(np.stack(ys)).to(DEV)


def train_net(train_recs, iters=1500, batch=32, seed=0, log=None, example=None, init=None, max_lr=3e-3,
              max_deg=25):
    """Plain PyTorch loop: forward -> loss -> loss.backward() (back-propagation) -> Adam step.

    init     a state_dict to start from (the pre-trained net) instead of random weights
    example  (inputs [n,8], target [n]) -- its predicted density is saved to log every 50 steps
             so animate.py can show the net learning."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = sl.build_net()
    if init is not None:
        net.load_state_dict(init)
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=max_lr * 0.7, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=iters)
    placed = [r for r in train_recs if r["target"] is not None]
    count_only = [r for r in train_recs if r["target"] is None and r["steps"] > 0]
    for it in range(iters):
        if log is not None and example is not None and (it % 50 == 0):
            net.eval()
            with torch.no_grad():
                pred = net(torch.from_numpy(example[0].T[None]).to(DEV))[0].cpu().numpy()
            log["snapshots"].append({"iter": it, "density": [round(float(v), 5) for v in pred]})
        net.train()
        x, y = make_batch(placed, rng, batch, max_deg)
        d = net(x)
        d, y = d[:, EDGE:-EDGE], y[:, EDGE:-EDGE]
        # (1) the density should look like the labelled bumps; (2) its sum should be the step count
        shape_loss = 50 * torch.mean((d - y) ** 2)
        count_loss = 0.1 * torch.mean((d.sum(1) - y.sum(1)) ** 2)
        loss = shape_loss + count_loss
        if count_only and it % 4 == 0:  # recordings the gyro couldn't place: only their total is known
            r = count_only[int(rng.integers(len(count_only)))]
            full = net(torch.from_numpy(r["inp"].T[None]).to(DEV))[0]
            ok = torch.from_numpy(r["ok"]).to(DEV)
            loss = loss + 0.05 * (full[ok].sum() - r["steps"]) ** 2 / (r["steps"] + 1)
        opt.zero_grad()
        loss.backward()  # back-propagation: d(loss)/d(every weight)
        if log is not None and (it % 10 == 0 or it == iters - 1):
            log["iter"].append(it)
            log["loss"].append(loss.item())
            log["shape_loss"].append(shape_loss.item())
            log["count_loss"].append(count_loss.item())
            log["grad_norm"].append([float(p.grad.norm()) for n, p in net.named_parameters()
                                     if n.endswith("weight") and p.dim() == 3])
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
    net.eval()
    return net.cpu()


# ── scoring ──────────────────────────────────────────────────────────────────

def summarize(errs, trues):
    """Errors of whole-recording counts -> accuracy, mean |error|, mean |%error|, bias and spread (%)."""
    errs, trues = np.asarray(errs, float), np.asarray(trues, float)
    rel = errs / trues
    return {"mae": float(np.mean(np.abs(errs))), "mape": float(np.mean(np.abs(rel)) * 100),
            "bias": float(np.mean(rel) * 100), "sd": float(np.std(rel, ddof=1) * 100) if len(rel) > 1 else 0.0,
            "accuracy": float(100 * (1 - np.sum(np.abs(errs)) / np.sum(trues)))}


def bootstrap_ci(errs, trues, n=2000, seed=0):
    """95% interval of the mean absolute % error, by resampling recordings."""
    rel = np.abs(np.asarray(errs, float) / np.asarray(trues, float)) * 100
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(rel, len(rel))) for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
