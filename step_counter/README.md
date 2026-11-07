# Step counter: a PyTorch neural net on the ankle IMU

This extends the [gait identifier](../gait_identifier) and uses the same wearable: an Arduino UNO Q with an MPU6050 on the outside of the ankle, streaming all six IMU outputs at 50 Hz to the laptop over MQTT. A small **1-D convolutional neural net**, trained in **PyTorch**, counts your steps live. The count also shows on the board's LED matrix.

## How it works

| Stage | What happens |
|---|---|
| Board | Reads accel x/y/z and gyro x/y/z every 20 ms. Sends them over MQTT (`gait/imu`). Shows the count it gets back (`gait/steps`) as digits on the LED matrix. |
| Inputs | 8 channels: accel (g), gyro (rad/s ÷ 4), \|a\| − 1, \|ω\| ÷ 4. |
| **StepNet** | 4 dilated `Conv1d` layers (32 filters × 9 taps, dilation 1, 2, 4, 8, each with BatchNorm, ReLU and Dropout), then a 1×1 conv and softplus. About 30k weights. Each output sample sees about 2.4 s of signal, roughly two strides. |
| Output | A **step density** for every sample. It's about 0 between steps, with a bump around each step whose area is 1 step. **Count = sum of the density.** |
| Live | Every 0.5 s the net reads the last 6 s and adds up the half second that ended 1 s ago. That way every step gets counted once, with context on both sides. |

### Optional experiment: transfer learning from public data

**By default the net trains on your own recordings only.** Pre-training on public data was tried and is now opt-in (`--transfer`). The public sensors are on the foot, not the ankle, and fine-tuning with only 0-step ankle data taught the net the shortcut "the ankle never steps". It needs plenty of your own counted walks to be a fair comparison.

1. **Pre-train** (`pretrain.py`) on the open [IPOL 2023 healthy-gait dataset](https://www.ipol.im/pub/art/2023/497/) (Voisard et al., doi:10.5201/ipol.2023.497). It has 19 people and 110 walks, each one standing, 10 m, a U-turn, back, then standing. It used XSens IMUs on each foot at 100 Hz, with accel in m/s² and gyro in rad/s, the same units as our MPU6050. **Every heel strike of both feet is marked and was checked by hand.** Each foot sensor becomes one 50 Hz recording, with both feet's heel strikes as its steps. That gives about 3,800 exactly timed steps. Validation holds out whole people: 4 groups, each counted by a net that never saw them.
2. **Fine-tune** (`python train_steps.py --transfer`) on your own counted walks, starting from the pre-trained weights: 800 steps at a lower learning rate. All accuracy numbers come from **your** sensor only. The report also counts every held-out recording with the pre-trained net *before* fine-tuning ("Pre-trained only"), which shows what fine-tuning adds.

The public data is not in git: download [497-main.zip](https://www.ipol.im/pub/art/2023/497/497-main.zip) (34 MB) and unzip it into `laptop/public_data/ipol2023/`.

### Training (PyTorch, back-propagation)

- **Ground truth is you.** You count every step out loud while recording and type the total at the end.
- **Placing the steps in time.** The gyro finds each swing of the strapped leg, which is 2 steps (the other foot lands halfway between). Those times are only trusted if their total matches your count. Recordings where it doesn't match still train the net through their total alone.
- **Loss** = shape loss (the density vs. bumps at the labelled steps) + count loss ((sum of density − counted steps)²).
- **Each training step:** take a batch of 8 s crops, run them forward, compute the loss, call `loss.backward()` (back-propagation), then `Adam.step()`.
- **Augmentation:** every crop gets a random sensor tilt (±25°, as if strapped on differently), a signal scale (±10%) and a time stretch (±12%, as if walking faster or slower).
- **Not-a-step examples:** standing, fidgeting and biking recordings, plus the gait identifier's standing and biking data, teach the net what isn't a step. A plain peak counter counts every pedal stroke as a step; the net learns not to.

### Uncertainty analysis

`train_steps.py` holds out **one recording at a time**, trains a fresh net on the rest, and counts the held-out one exactly the way the live counter does. From that it reports:

- **Accuracy:** 1 − Σ|error| / Σsteps, and the mean |% error| per walk with a **bootstrap 95% CI**.
- **Bias and spread:** the Bland–Altman 95% limits, i.e. how far off one count can be. The live page shows this as **± steps**.
- **False steps per minute** while standing, fidgeting and biking.
- **The same numbers for two classical pedometers**, to compare against: an |a| peak counter, and 2 × the gyro swings.

## Files

```
laptop/steplib.py       recordings, step labeler, StepNet, live counter, baselines
laptop/trainlib.py      the PyTorch training loop + augmentation + scoring (shared)
laptop/publicdata.py    loads the IPOL 2023 public gait data onto our 50 Hz format
laptop/pretrain.py      pre-train on public data -> models/pretrained.pt, pretrain_report.txt
laptop/train_steps.py   fine-tune on your data: uncertainty analysis + final net -> models/ (stepnet.pt, report.txt, plots)
laptop/app.py + web/    dashboard at http://localhost:8001: Record / Count / Model
laptop/animate.py       renders models/how_it_works.mp4 (system -> net -> loss -> back-prop -> training)
laptop/data/            <kind>_<N>steps_<session>_<time>.csv
```

The board runs the gait identifier's App Lab app (`../gait_identifier/unoq-gait-app`). Its sketch now also has `show_steps`, so **re-upload it once in App Lab** to get the count on the LED matrix.

## Steps

```bash
pip install torch numpy scipy matplotlib paho-mqtt imageio-ffmpeg
cd step_counter/laptop
python app.py          # http://localhost:8001
```

1. **Record** (Record tab). Stand still and press Start. On "GO", walk while **counting every step out loud**. Stop, stand still, press Stop, then type your count. Aim for 30–100 steps each:

   | Kind | Recordings | Why |
   |---|---|---|
   | walk | 3 | normal pace |
   | slowwalk, fastwalk | 2 each | so the net handles different speeds |
   | run | 2 | jogging |
   | stairs | 2 | different impacts |
   | stopgo | 2 | starts, stops and turns, where pedometers miscount |
   | still, fidget, bike | 1–2 each | 0 steps: what isn't a step |

2. **Train** (Model tab, or `python train_steps.py`). It trains on your recordings only and runs the leave-one-recording-out uncertainty test. That takes a few minutes on the GPU. *(Optional: run `python pretrain.py` once, then `python train_steps.py --transfer` for the public-data experiment.)*
3. **Count** (Count tab). Press Start counting and walk. For the demo video, count out loud and compare with the screen and the LED matrix.
4. **Animation:** `python animate.py` renders `models/how_it_works.mp4` from your trained net.
5. **Writeup:** `python ../make_writeup.py` rebuilds `Step_Counter_Writeup.docx` from the latest results.
