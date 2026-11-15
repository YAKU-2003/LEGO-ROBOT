# Fitting the V data from Anna's Code with gradient descent

**Data:** 120 points exported from Anna's Code trainer ([annacode-v.tsv](annacode-v.tsv)), x from 20 to 99, y from 0 to 81.
- The shape is a **V**: y falls in a straight line to about 0 at x ≈ 60, then rises again.
- The math uses **x/100** (0.2 to 1.0) so that one learning rate works for every weight.

Files:
- `fit_v.py`: the code. It produces these outputs:
  - `fit_line.png`, `fit_relu1.png`, `fit_relu2.png`, `fit_all.png`
  - `loss_vs_epoch.png`
  - `first_5_iterations.csv`
- `V_gradient_descent.xlsx`: 5 iterations of m and b by hand, all live formulas.

(The earlier colour-sensor run, Reflection vs. motor speed, is in `hinge_colour_sensor/`. It only produced half a V.)

| model | weights | best MSE | RMS miss |
|---|---|---|---|
| A straight line | m, b | 490.7 | 22.2 |
| B one neuron + ReLU | w, b, v, c | 300.8 | 17.3 |
| **C hidden layer of 2 ReLU neurons** | w₁, b₁, w₂, b₂, v₁, v₂, c | **4.13** | **2.03** |

The 2.03 RMS miss matches Anna's page, which reports "average miss 2.1".

---

## 1. One neuron, no activation: a straight line

$$\hat y_i = m\,x_i + b$$

**Loss: mean squared error over the n = 120 points**

$$L(m,b) = \frac1n\sum_{i=1}^{n}(m x_i + b - y_i)^2$$

**Gradient (chain rule)**

Write $e_i = m x_i + b - y_i$ for the error on point i. Then:

$$\frac{\partial L}{\partial m} = \frac1n\sum 2e_i\cdot\frac{\partial e_i}{\partial m} = \frac2n\sum e_i\,x_i
\qquad
\frac{\partial L}{\partial b} = \frac2n\sum e_i$$

- If the line sits too low on average, $\sum e_i < 0$, so the gradient is negative and b goes **up**.
- If it's too low where x is big, $\sum e_i x_i < 0$, so m goes **up**.

**Gradient-descent update with learning rate α**

$$m \leftarrow m - \alpha\frac{\partial L}{\partial m}\qquad b \leftarrow b - \alpha\frac{\partial L}{\partial b}$$

The gradient points uphill, so we step the other way, scaled by α.

### Iteration 1 by hand (α = 0.3, start m = 0, b = 0)

| quantity | value |
|---|---|
| every prediction $\hat y_i$ | 0, so every error $e_i = -y_i$ |
| $\sum e_i = -\sum y_i$ | −4831.06 |
| $\sum e_i x_i$ | −3081.55 |
| **loss** $= \frac1n\sum y_i^2$ | **2124.56** |
| $\partial L/\partial m = \frac{2}{120}\cdot(-3081.55)$ | −51.3592 |
| $\partial L/\partial b = \frac{2}{120}\cdot(-4831.06)$ | −80.5177 |
| new m = 0 − 0.3 · (−51.3592) | **15.4078** |
| new b = 0 − 0.3 · (−80.5177) | **24.1553** |

### All 5 iterations (identical in Python and in the Excel sheet)

| iter | m | b | loss | dL/dm | dL/db | new m | new b |
|---|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 2124.56 | −51.3592 | −80.5177 | 15.4078 | 24.1553 |
| 2 | 15.4078 | 24.1553 | 534.16 | −8.1789 | −13.1860 | 17.8615 | 28.1111 |
| 3 | 17.8615 | 28.1111 | 492.17 | −1.1678 | −2.2453 | 18.2118 | 28.7847 |
| 4 | 18.2118 | 28.7847 | 491.05 | −0.0325 | −0.4656 | 18.2216 | 28.9244 |
| 5 | 18.2216 | 28.9244 | 491.00 | 0.1484 | −0.1742 | 18.1770 | 28.9766 |

- **Step 1:** the loss drops from 2125 to 534, because the line jumps up from y = 0 to the data's average height.
- **Steps 3–5:** the gradients are almost 0. The line has already found the best a straight line can do.
- **After 2000 steps:** $\hat y = 15.79\,x + 30.51$ with **MSE 490.7**, almost flat through the middle of the V.

The gradient isn't wrong. The *model* is too simple: no choice of m and b can make a line go down and then up.

---

## 2. One neuron with a ReLU activation: one bend

$$z_i = w\,x_i + b,\qquad h_i = \mathrm{ReLU}(z_i) = \max(0, z_i),\qquad \hat y_i = v\,h_i + c$$

ReLU passes positive numbers through and turns negative ones into 0, so the neuron is
- **on** (a ramp) on one side of $x = -b/w$, and
- **off** (flat at $c$) on the other.

**Back-propagation, the chain rule from the loss back to each weight.** Let $g_i = \partial L/\partial \hat y_i = \frac{2}{n}(\hat y_i - y_i)$.

| weight | chain | gradient |
|---|---|---|
| c | $\hat y = vh + c$ | $\sum g_i$ |
| v | $\hat y = vh + c$ | $\sum g_i\,h_i$ |
| w | $\hat y \to h \to z = wx + b$ | $\sum g_i\cdot v\cdot \mathrm{ReLU}'(z_i)\cdot x_i$ |
| b | $\hat y \to h \to z$ | $\sum g_i\cdot v\cdot \mathrm{ReLU}'(z_i)$ |

Here $\mathrm{ReLU}'(z) = 1$ if $z > 0$, otherwise 0. Points where the neuron is off send **no** gradient to w and b.

**Result (α = 0.002, starting at w = 5, b = −3, v = 10, c = 20):** the bend lands at x ≈ 76.
- Right of the bend, the neuron fits the rising ramp.
- Left of it, everything is flat at 31.7.
- **MSE 300.8**, better than the line but still only half the V.

Starting with c = 0 instead, the bend slid off the left edge of the data, the neuron was "on" for every point, and it turned into the same straight line (MSE 490.7). **Where you start matters.**

---

## 3. A hidden layer of 2 ReLU neurons: two bends, a V

$$\hat y_i = v_1\,\mathrm{ReLU}(w_1 x_i + b_1) + v_2\,\mathrm{ReLU}(w_2 x_i + b_2) + c$$

The gradients are exactly the table above, once per neuron. Each neuron only receives gradient from the points where it's on.

**Checking the hand-written back-propagation.** At the starting weights, it gives the same numbers as PyTorch's automatic differentiation (`loss.backward()`):

| | hand-derived | PyTorch |
|---|---|---|
| dL/dw₁ | −99.265412 | −99.265412 |
| dL/dw₂ | −286.259971 | −286.259971 |
| dL/db₁ | −278.981577 | −278.981577 |
| dL/db₂ | −325.868552 | −325.868552 |
| dL/dv₁ | −34.061767 | −34.061767 |
| dL/dv₂ | −45.369420 | −45.369420 |
| dL/dc | −60.485013 | −60.485012 |

**Result (α = 0.005, 30 000 steps, starting at w = [−5, 5], b = [3, −3], v = [10, 10], c = 0):**

| | w | b | v | on when | slope it adds |
|---|---|---|---|---|---|
| neuron 1 | −12.92 | 7.77 | 15.38 | x < 60.1 | −1.99 per unit of x |
| neuron 2 | 13.07 | −7.84 | 15.18 | x > 60.0 | +1.98 per unit of x |
| output bias c | | | 0.39 | | bottom of the V |

**MSE 4.13 (an RMS miss of 2.03).** The network found the V by itself:
- one neuron for each arm, both bending at x = 60,
- slopes of ±2,
- a bottom at 0.4.

That's what the tip on Anna's page means by "a V is two ramps: add a hidden layer with at least 2 neurons and ReLU". The loss curve hovers around 200–450 for about the first 100 epochs while the bends move into place, then drops to about 4 within a few hundred epochs.

### Learning rate too big: dead ReLUs

The same network with α = 0.01:
1. The first gradients are large (−280 to −330 on the biases).
2. The first steps throw both bends outside the data.
3. Now z < 0 for **every** point, so ReLU' = 0, so w and b get **zero gradient forever**.
4. Both neurons are dead. Only c keeps learning (it becomes the average of y), and the loss stays stuck at 503.8.

That's the dashed grey curve in `loss_vs_epoch.png`. Fixes: a smaller learning rate (what we used), good starting weights, or a "leaky" ReLU that still lets a little gradient through when it's off.

---

## 4. From this to the lock's CNN

The finger-counting network uses exactly the same idea, just larger:
- **Neurons:** each conv filter is a neuron like these, with weights, a bias and a ReLU, slid across the image.
- **Loss:** cross-entropy over 6 classes replaces squared error, because we want a category, not a number.
- **Training:** back-propagation applies the same chain rule through 3 conv layers, 3 pooling layers and 1 linear layer, about 80 000 weights.
- **Optimizer:** Adam, a gradient-descent variant that adapts each weight's step size, replaces plain gradient descent.
