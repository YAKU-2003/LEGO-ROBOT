"""
Fit the V data from Anna's Code with gradient descent - three models, gradients written by hand.

  python fit_v.py

  A  one neuron, no activation (straight line)      y_hat = m*x + b
  B  one neuron + ReLU (+ output weight)            y_hat = v * ReLU(w*x + b) + c
  C  hidden layer of 2 ReLU neurons + linear output y_hat = v1*ReLU(w1*x+b1) + v2*ReLU(w2*x+b2) + c

A can't bend, B can bend once (half a V), C can bend twice (a V = two ramps).
Back-propagation is done by hand (chain rule) and checked once against PyTorch autograd.

x is scaled to x/100 (0.2..1.0) so one learning rate suits every weight.
Outputs: fit_line.png, fit_relu1.png, fit_relu2.png, fit_all.png, loss_vs_epoch.png, first_5_iterations.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "annacode-v.tsv"


def load():
    d = np.genfromtxt(DATA, delimiter="\t", skip_header=1, encoding="utf-8")
    d = d[~np.isnan(d).any(axis=1)]
    return d[:, 1] / 100.0, d[:, 2]


# ------------------------------------------------------------------ A: straight line
def train_line(x, y, lr=0.3, epochs=2000, m=0.0, b=0.0, log_first=5):
    n, hist, rows = len(x), [], []
    for ep in range(epochs + 1):
        err = m * x + b - y
        loss = np.mean(err ** 2)
        dm, db = 2 / n * np.sum(err * x), 2 / n * np.sum(err)     # dL/dm, dL/db
        hist.append(loss)
        if ep < log_first:
            rows.append((ep + 1, m, b, loss, dm, db, m - lr * dm, b - lr * db))
        if ep == epochs:
            break
        m, b = m - lr * dm, b - lr * db
    return m, b, hist, rows


# ------------------------------------------------------------------ B, C: ReLU hidden layer
def forward(x, w, b, v, c):
    """w, b, v are arrays with one entry per hidden neuron."""
    z = np.outer(x, w) + b              # n x H   pre-activation
    h = np.maximum(0, z)                # n x H   ReLU
    return z, h, h @ v + c              # y_hat


def grads(x, y, w, b, v, c):
    """Back-propagation by hand."""
    n = len(x)
    z, h, y_hat = forward(x, w, b, v, c)
    g = 2 / n * (y_hat - y)             # dL/dy_hat for every point
    dc = g.sum()                        # y_hat = sum_j v_j h_j + c
    dv = h.T @ g                        # dy_hat/dv_j = h_j
    dz = np.outer(g, v) * (z > 0)       # dy_hat/dh_j = v_j,  dh/dz = ReLU'(z) = 1 if z>0 else 0
    dw = (dz * x[:, None]).sum(0)       # dz/dw = x
    db = dz.sum(0)                      # dz/db = 1
    return np.mean((y_hat - y) ** 2), dw, db, dv, dc


def train_relu(x, y, w, b, v, c, lr, epochs):
    w, b, v = (np.array(a, float) for a in (w, b, v))
    hist = []
    for _ in range(epochs):
        loss, dw, db, dv, dc = grads(x, y, w, b, v, c)
        hist.append(loss)
        w, b, v, c = w - lr * dw, b - lr * db, v - lr * dv, c - lr * dc
    hist.append(grads(x, y, w, b, v, c)[0])
    return (w, b, v, c), hist


def check_with_torch(x, y, w, b, v, c):
    try:
        import torch
    except ImportError:
        print("(PyTorch not installed - skipping autograd check)")
        return
    W, B, V = (torch.tensor(np.array(a, float), requires_grad=True) for a in (w, b, v))
    C = torch.tensor(float(c), requires_grad=True)
    X, Y = torch.tensor(x), torch.tensor(y)
    loss = ((torch.relu(X[:, None] * W + B) @ V + C - Y) ** 2).mean()
    loss.backward()
    _, dw, db, dv, dc = grads(x, y, np.array(w, float), np.array(b, float), np.array(v, float), c)
    print("Gradient check (2-neuron net, starting weights)   hand-derived   vs   PyTorch")
    for name, a, t in [("w1", dw[0], W.grad[0]), ("w2", dw[1], W.grad[1]), ("b1", db[0], B.grad[0]),
                       ("b2", db[1], B.grad[1]), ("v1", dv[0], V.grad[0]), ("v2", dv[1], V.grad[1]),
                       ("c ", dc, C.grad)]:
        print(f"  dL/d{name}   {a:14.6f}   {float(t):14.6f}")


def plot_fit(x, y, xs, ys, title, name, extra=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x * 100, y, s=16, alpha=.7, label="Anna's Code data")
    ax.plot(xs * 100, ys, color="tab:red", lw=2.5, label="model")
    if extra:
        for xs2, ys2, lab, st in extra:
            ax.plot(xs2 * 100, ys2, st, lw=1.5, label=lab)
    ax.set(xlabel="x", ylabel="output", title=title, ylim=(-15, 100))
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / name, dpi=130); plt.close(fig)


def main():
    x, y = load()
    print(f"{len(x)} points, x {x.min()*100:.1f}..{x.max()*100:.1f}, y {y.min():.1f}..{y.max():.1f}\n")
    xs = np.linspace(0.18, 1.02, 400)

    # ---- A
    m, b, hist_a, rows = train_line(x, y)
    print("A) straight line - first 5 iterations (same as the Excel sheet):")
    print(" iter        m          b        loss        dL/dm      dL/db     m_new      b_new")
    for r in rows:
        print(" {:3d} {:10.4f} {:10.4f} {:11.3f} {:10.4f} {:10.4f} {:9.4f} {:9.4f}".format(*r))
    np.savetxt(HERE / "first_5_iterations.csv", np.array(rows), delimiter=",", fmt="%.6f",
               header="iter,m,b,loss,dL_dm,dL_db,m_new,b_new", comments="")
    print(f"   after {len(hist_a)-1} epochs: y = {m:.2f}*(x/100) + {b:.2f}   MSE {hist_a[-1]:.1f}\n")

    # ---- B: one ReLU neuron (starts 'on' for the right-hand ramp, output bias near the data's middle;
    #      starting c at 0 instead lets the bend slide off the data and it just becomes the line)
    pB, hist_b = train_relu(x, y, w=[5.0], b=[-3.0], v=[10.0], c=20.0, lr=0.002, epochs=30000)
    w, bb, v, c = pB
    print(f"B) one ReLU neuron: w={w[0]:.2f} b={bb[0]:.2f} v={v[0]:.2f} c={c:.2f}   MSE {hist_b[-1]:.1f}"
          f"   bend at x={-bb[0]/w[0]*100:.1f}\n")

    # ---- C: two ReLU neurons, one per side of the V
    w0, b0, v0, c0 = [-5.0, 5.0], [3.0, -3.0], [10.0, 10.0], 0.0
    check_with_torch(x, y, w0, b0, v0, c0)
    pC, hist_c = train_relu(x, y, w0, b0, v0, c0, lr=0.005, epochs=30000)
    w, bb, v, c = pC
    print(f"\nC) two ReLU neurons   MSE {hist_c[-1]:.2f}   (RMS miss {np.sqrt(hist_c[-1]):.2f})")
    for j in range(2):
        print(f"   neuron {j+1}: w={w[j]:7.2f}  b={bb[j]:6.2f}  v={v[j]:6.2f}   on for x {'<' if w[j] < 0 else '>'} "
              f"{-bb[j]/w[j]*100:.1f}   slope there {v[j]*w[j]/100:+.2f} per unit x")
    print(f"   c = {c:.2f}  (the output at the bottom of the V)")

    # ---- C with a learning rate that is too big
    pD, hist_dead = train_relu(x, y, w0, b0, v0, c0, lr=0.01, epochs=30000)
    alive = [(np.outer(x, pD[0]) + pD[1] > 0).any(0)]
    print(f"\nC with lr=0.01: MSE {hist_dead[-1]:.1f}; neurons still on for some point: {alive[0].tolist()}")

    # ---- plots
    plot_fit(x, y, xs, m * xs + b, f"A) straight line   (MSE {hist_a[-1]:.0f})", "fit_line.png")
    plot_fit(x, y, xs, forward(xs, *pB)[2], f"B) one neuron + ReLU   (MSE {hist_b[-1]:.0f})", "fit_relu1.png")
    zc, hc, yc = forward(xs, *pC)
    plot_fit(x, y, xs, yc, f"C) 2 ReLU neurons   (MSE {hist_c[-1]:.1f})", "fit_relu2.png",
             extra=[(xs, v[j] * hc[:, j] + c, f"neuron {j+1}: v·ReLU(w·x+b) + c", "--") for j in range(2)])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x * 100, y, s=14, alpha=.5, color="gray", label="data")
    ax.plot(xs * 100, m * xs + b, lw=2, label=f"A line (MSE {hist_a[-1]:.0f})")
    ax.plot(xs * 100, forward(xs, *pB)[2], lw=2, label=f"B 1 ReLU (MSE {hist_b[-1]:.0f})")
    ax.plot(xs * 100, yc, lw=2.5, label=f"C 2 ReLUs (MSE {hist_c[-1]:.1f})")
    ax.set(xlabel="x", ylabel="output", title="Each extra bend fits more of the V", ylim=(-15, 100))
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "fit_all.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist_a, label="A straight line")
    ax.plot(hist_b, label="B one ReLU neuron")
    ax.plot(hist_c, label="C two ReLU neurons, lr 0.005")
    ax.plot(hist_dead, "--", color="gray", label="C, lr 0.01 (dead ReLUs)")
    ax.set(xscale="log", yscale="log", xlabel="epoch", ylabel="mean squared error", title="Loss vs epoch")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(HERE / "loss_vs_epoch.png", dpi=130); plt.close(fig)
    print("\nSaved fit_line.png, fit_relu1.png, fit_relu2.png, fit_all.png, loss_vs_epoch.png, first_5_iterations.csv")


if __name__ == "__main__":
    main()
