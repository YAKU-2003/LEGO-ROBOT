# 🥪 LunchLock: A Finger-Code Lunch Safe

**Your lunch, your code. No keys, no app, no more "someone ate my sandwich."**

LunchLock is a compact, fridge-sized lunch box that only opens for its owner. Press the button, the LEDs light up, show your secret three-number code with your fingers to the camera, and the latch opens. Wrong code? It stays shut, and after three tries it locks everyone out.

Under the hood, a convolutional neural network that I built and trained from scratch in PyTorch reads how many fingers you are holding up. It runs on an Arduino UNO Q inside the box.

> 🎥 **[VIDEO: LunchLock opening in the fridge — paste here]**

---

## 🤖 Skills

**Programming:** Python, PyTorch, OpenCV, NumPy, HTML/JS (dashboard)

**Machine learning:** CNNs, pooling, data augmentation, cross-entropy loss, backpropagation, gradient descent

**Hardware:** Arduino UNO Q, USB webcam, push button, LEDs, latch [servo/motor — fill in]

**CAD / Manufacturing:** [Fusion 360 / SolidWorks], FDM 3D printing [PLA/PETG — fill in]

**Tools:** Excel (hand gradient descent), Matplotlib, Git

---

# 🍱 The Problem

Shared office and dorm fridges have one universal problem: food goes missing. Locks with keys get lost, combination padlocks are fiddly with cold hands, and phone apps are overkill for a sandwich.

**The idea:** a box small enough to sit on a fridge shelf, with a "key" you always carry, your hand. Your code is a sequence of three finger counts (each 1–5), like a PIN you show instead of type. That gives 125 possible codes, and nothing to lose.

**Design goals**

- Compact enough to fit on a standard fridge shelf
- Works in the fridge's dim, cold light
- Unlocks only for the right 3-digit code
- Runs fully on the box, with no laptop or internet needed
- Simple to use: one button, show three numbers, done

---

# 🔐 How It Works

1. **Wake:** press the button on the front. The LEDs light up, and you have 20 seconds.
2. **Show digit 1:** hold up your fingers inside the marked area under the camera. When the network is sure, the first LED fills.
3. **Digits 2 and 3:** change to the next number. To repeat a digit (for example 3-3-1), make a fist in between.
4. **Decide:** all three correct → the latch opens for 6 seconds and then relocks. Wrong → it stays locked. Three wrong codes in a row → a 30-second lockout.

> 📷 **[PHOTO: box front with button + LEDs]**   📷 **[PHOTO: hand under the camera]**

---

# 🧠 The Lock Policy: How It Decides

The neural network looks at about 10 camera frames per second and guesses a finger count each time. A single guess is never trusted. The lock policy turns that noisy stream into clean digits:

| Rule | Value | Why |
|---|---|---|
| Minimum confidence per frame | **85%** | Ignores frames where the network is unsure (hand moving, half out of view) |
| Hold time to enter a digit | **0.7 s** of the *same* answer | Changing from 1 to 4 briefly passes through 2 and 3, which must not count |
| Repeat a digit | make a **fist** for 0.3 s first | Otherwise holding a 3 would enter 3-3-3 |
| Time to enter the code | **20 s** after the button | Walk away and it resets |
| Unlocked time | **6 s**, then relocks | You never have to remember to lock it |
| Wrong attempts | **3**, then a **30 s lockout** | Stops someone from cycling through codes |

The policy is a small state machine: **LOCKED → (button) → ENTERING → 3 digits → OPEN / WRONG / LOCKED OUT → LOCKED**. The same code (`lock_logic.py`) runs on my laptop and on the UNO Q, so the box behaves identically in both.

---

# 🧩 The Neural Network

> 🎬 **[STOP-MOTION VIDEO: "How LunchLock sees your fingers" — paste here]**

The network takes a 96 × 96 crop of the camera image and outputs 6 scores, one for each finger count from 0 to 5. The highest score wins. **It is one network that looks at the hand in two ways at once.**

**1. The hand mask: my "grid" idea (early fusion).** Before anything is learned, each pixel's redness (the Cr channel of YCrCb) is compared with the background. Skin is redder than the wall or box, so the hand lights up as a white silhouette. Because the comparison is relative to each image, it barely changes with lighting. The mask is stacked with the grayscale photo, so the network's very first layer sees **two pictures: the photo and where the hand is.**

**2. Three convolution + pooling blocks.**

| Layer | What it does | Output size |
|---|---|---|
| Input | grayscale photo + hand mask | 2 × 96 × 96 |
| Conv 5×5, 16 filters + ReLU | finds edges and blobs | 16 × 96 × 96 |
| Max-pool 2×2 | keeps the strongest response in each 2×2, halves the size | 16 × 48 × 48 |
| Conv 3×3, 32 filters + ReLU | combines edges into finger parts | 32 × 48 × 48 |
| Max-pool 2×2 | | 32 × 24 × 24 |
| Conv 3×3, 64 filters + ReLU | combines parts into "raised finger here" | 64 × 24 × 24 |
| Max-pool 2×2 | | 64 × 12 × 12 |
| + 12×12 hand-density grid + finger-height profile | 168 hand-designed numbers | |
| Linear layer | 9,384 numbers → 6 scores | 6 |

- **Width** is the number of filters in each layer (16, 32, 64). **Depth** is 3 convolution layers + 1 linear layer.
- **Convolution:** a small filter slides across the image, multiplying and adding, and lights up wherever its pattern appears.
- **ReLU** keeps positive responses and zeroes negative ones. It is what lets the network represent bends and shapes, not just straight lines.
- **Max-pooling** shrinks each map by keeping only the strongest value in every 2×2 block. That makes the network care *whether* a finger is there more than its exact pixel position, and it cuts the computation 4× per layer.
- **Cross-entropy loss** turns the 6 scores into probabilities (softmax) and punishes the network for putting low probability on the right answer. I want a category, not a number, so this fits better than squared error.
- **Backpropagation** uses the chain rule to work out how much each of the ~80,000 weights contributed to the error. **Adam** (a gradient-descent variant) then nudges every weight downhill.

> 📷 **[IMAGE: dashboard "What the grid sees" screenshot — hand mask with 12×12 grid]**

---

# 📸 Data and Training

**Dataset:** 2,400 photos, 400 per finger count from 0 to 5, captured with my own capture script. The same webcam and the same crop box are used for training, on the laptop and on the UNO Q. Each burst is 200 shots, 0.2 s apart, while I keep tilting and rotating my hand.

**Augmentation:** every epoch each photo is randomly rotated (±20°), shifted, zoomed, mirrored (left vs right hand), brightened or darkened, gamma-shifted, lit from one side, given sensor noise and blurred. The network effectively never sees the same image twice.

**Honest validation:** the last 20% of every burst is held out, not a random 20%. Photos taken 0.2 s apart are near-duplicates, so a random split would let the network "cheat" by memorising neighbours.

**Training:** 80 epochs, Adam (lr 0.001, cosine schedule), batch 32, about 6 minutes on a laptop CPU.

> 📷 **[IMAGE: training_curves.png]**   📷 **[IMAGE: confusion_matrix.png]**

### What I learned the hard way (and fixed)

**1. Labels.** My first capture run saved images under the wrong labels because the file name came from the key I pressed, not what my hand showed. Lesson: always look at your data before training.

**2. Lighting.** The first model scored 99% on test photos but **dropped to 29% when the room got darker.** I fixed it two ways. First, every image is now normalised to the same brightness and contrast inside the network. Second, training adds the random lighting augmentation above.

| Test photos re-lit | Before fix | After fix |
|---|---|---|
| As captured | 99.2% | 98.3% |
| Dark | **29.2%** | **97.9%** |
| Low contrast | 46.7% | 97.9% |
| Light from one side | 61.3% | 92.1% |

**3. Background shortcut.** When I added new photos of only 3 and 4 fingers against a different background, the network learned "that background = 3 or 4" instead of counting fingers. It called 64 of 77 test 4s a "3". Lesson: every class has to appear in every setting. Recapturing all six classes in the same place fixed it.

### Final model

| | Result |
|---|---|
| Held-out photos, all 6 finger counts | **100%** (480 / 480) |
| Dark / low contrast / side light | **100%** |
| Weights | ~80,800 |

These held-out photos come from my own hand and setup. See Limitations.

### Does the grid actually help? (ablation test)

I switched parts of the trained network off and re-tested it on the same photos:

| Configuration | Accuracy |
|---|---|
| Full model | 100% |
| Without the hand-mask input | **33%** |
| Without the 12×12 grid numbers at the end | 100% |
| Grid features alone (no CNN, trained separately) | 85% |

The **hand mask is what the network relies on most.** It learned to read finger shapes from the silhouette, which is why changing the background stopped mattering. The 12×12 grid numbers at the end turned out to be redundant, because the CNN already extracts the same information from the mask.

---

# 📈 Warm-Up: Gradient Descent on the "V" Data (Anna's Code)

Before building the lock, I fit a model by hand to the V-shaped data from Anna's Code: 120 points with a minimum at x ≈ 60.

**Straight line (one neuron, no activation):** ŷ = m·x + b, loss L = (1/n)·Σ(m·x + b − y)²

- ∂L/∂m = (2/n)·Σ(m·x + b − y)·x
- ∂L/∂b = (2/n)·Σ(m·x + b − y)
- update: m ← m − α·∂L/∂m, b ← b − α·∂L/∂b

> 📎 **[FILE: V_gradient_descent.xlsx — 5 iterations of m and b, every cell a live formula]**

| iter | m | b | loss | dL/dm | dL/db |
|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 2124.6 | −51.36 | −80.52 |
| 2 | 15.408 | 24.155 | 534.2 | −8.18 | −13.19 |
| 3 | 17.861 | 28.111 | 492.2 | −1.17 | −2.25 |
| 4 | 18.212 | 28.785 | 491.0 | −0.03 | −0.47 |
| 5 | 18.222 | 28.924 | 491.0 | 0.15 | −0.17 |

(x scaled to x/100, α = 0.3). The line gets stuck at MSE ≈ 491: **no straight line can go down and then up.**

**Adding ReLU:** a neuron ŷ = v·ReLU(w·x + b) + c is a ramp on one side of x = −b/w and flat on the other, so it can bend once. **Two ReLU neurons** in a hidden layer can bend twice, and a V is exactly two ramps.

| Model | MSE | Average miss |
|---|---|---|
| Straight line | 490.7 | 22.2 |
| 1 ReLU neuron | 300.8 | 17.3 |
| **2 ReLU neurons** | **4.13** | **2.03** |

The two neurons found the V on their own: one per arm, both bending at x = 60, with slopes of ±2. I wrote the backpropagation by hand and checked it against PyTorch autograd; they agree to 6 decimal places. I also hit a **"dead ReLU"**: with too big a learning rate, the first step pushed both bends off the data, the ReLU output 0 everywhere, its gradient became 0, and it never recovered.

> 📷 **[IMAGE: fit_all.png]**   📷 **[IMAGE: fit_relu2.png]**   📷 **[IMAGE: loss_vs_epoch.png]**

---

# 🖥️ The Dashboard (Laptop Version)

I built a local web dashboard (`app.py`) to develop and demo the lock before moving it to the UNO Q:

- **Safe tab:** live camera with the capture box and the network's 6 probabilities. It also shows the live hand-mask grid, the code LEDs with a "hold" ring that fills, the timer, sounds and an activity log.
- **Capture tab:** take a new burst of training photos for any finger count with one click.
- **Model tab:** retrain with one click, then watch the training log, accuracy, lighting test and confusion matrix.

> 📷 **[IMAGE: dashboard screenshot]**

---

# 🧰 Mechanical Design

> **[FILL IN once the 3D prints are done]**
>
> - Box size: ___ × ___ × ___ mm (fits a fridge shelf)
> - Material / printer: ___
> - Camera mount: fixed height of ___ mm above the hand area, so the hand is always the same size in the image
> - Hand area: marked square on ___, with a dark, non-skin-coloured background so the hand mask is clean
> - Latch: ___ (servo / solenoid / motor)
> - Where the UNO Q, button and LEDs sit: ___
> - What I would change: ___

> 📷 **[CAD renders]**   📷 **[print + assembly photos]**

---

# 🔌 Running on the Arduino UNO Q

> **[FILL IN after the UNO Q build]**
>
> - The model is trained on the laptop and copied to the UNO Q; it is **not** trained on the board.
> - The same USB webcam and the same crop box are used, so the board sees exactly what the model was trained on.
> - The button and LEDs are wired to ___; the latch to ___.
> - Inference speed on the UNO Q: ___ fps (laptop: ~10 fps, limited by the webcam)

---

# ⚖️ UNO Q vs. Computer

> **[FILL IN after the UNO Q build — suggested comparison table]**

| | Laptop | UNO Q |
|---|---|---|
| Frames per second | ~10 (webcam-limited) | ___ |
| Time to recognise one digit | ~0.7 s hold + ___ | ___ |
| Accuracy in live use | ___ | ___ |
| Power / size | needs a laptop | fits in the box |
| Setup / debugging | easy (dashboard) | ___ |
| Training | yes (6 min) | no, the model is copied over |

---

# ⚠️ Limitations

1. **It only knows my hand.** All 2,400 training photos are of my own hand. Other people's hands (size, skin tone, rings, sleeves) may be read less reliably. **[Add: results when friends tried it]**
2. **It depends on the skin mask.** The network leans heavily on the hand mask (accuracy falls to 33% without it). A skin-coloured background, very bright light that washes out skin colour, or gloves would break it.
3. **It is not a secure lock.** Anyone who watches you enter the code can copy it. There are only 125 codes, and the 3-try lockout only slows down a determined person. It keeps honest people honest; it is not a bank vault.
4. **4 vs 5 and 1 vs 2 are the hardest pairs.** When fingers touch or the thumb is half-tucked, the silhouettes look alike. Most of the mistakes I saw were between these counts.
5. **It needs a fixed camera setup.** The model was trained with the camera at one height and angle and with one crop box. Move the camera or change the background, and it needs new photos and retraining.
6. **Held-out ≠ real world.** 100% on held-out photos means "it learned my setup", not "it never makes mistakes". Those photos are from the same sessions, lighting and hand as the training data.
7. **Fridge conditions.** Cold, dim light, condensation on the lens, and the battery life for a box that lives in a fridge were not fully tested. **[update after testing]**

---

# 🎯 Reflection

**What I'm proud of:** [personalise] Turning my own "grid" idea into part of the network (early fusion) and then *proving* with an ablation test that it is what the network relies on. Also finding and fixing three real data problems: wrong labels, lighting collapse, and the background shortcut.

**What was hardest:** [personalise] Realising that most failures were data problems, not network problems. A 99% model can fall to 29% just because the room got darker.

**What I learned:** [personalise] How gradient descent and backpropagation actually move the weights (I did it by hand in Excel before letting PyTorch do it), why ReLU is needed to make bends, and how pooling makes a CNN care about *what* is in the image more than *where*.

**What I'd do next:** [personalise] Train on many people's hands, try MediaPipe-style hand keypoints for comparison, add an "intruder photo" for lockouts, and seal the box for fridge use.

**Something we did not do in class:** early fusion of a hand-designed skin-mask grid with a learned CNN, plus an ablation study to measure its effect; lighting-robust training; and a full web dashboard for capture, training and live testing.

---

# 💻 Code Repository

[GitHub link — fill in]

| File | What it does |
|---|---|
| `finger_model.py` | the network (CNN + hand mask + grid) and image preprocessing |
| `train_cnn.py` | training, augmentation, lighting test, plots |
| `lock_logic.py` | the lock policy state machine |
| `app.py` + `web/index.html` | the dashboard |
| `lock_pc.py` | simple OpenCV-window version of the lock |
| `ablation.py` | switches parts of the network off to measure what helps |
| `vdata/fit_v.py`, `vdata/V_gradient_descent.xlsx`, `vdata/MATH.md` | the gradient-descent warm-up |
| `[unoq app folder]` | UNO Q version |

---

# 📚 Citations and Attributions

[1] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019. https://pytorch.org/

[2] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000. https://opencv.org/

[3] Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, 1998. (convolution + pooling)

[4] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *ICLR*, 2015.

[5] S. Ioffe and C. Szegedy, "Batch Normalization," *ICML*, 2015.

[6] Tufts CEEO, "AnnaCode / Neural Network Trainer," ME193 course materials. https://github.com/chrisbuerginrogers/ME193-Robotics

[7] Arduino, "Arduino UNO Q documentation." https://docs.arduino.cc/

[8] Code developed with assistance from Claude (Anthropic).
