# Gait Identifier 🦶

**ME193 – AI in Mobile Robots · Yash Kulkarni**

> 🎥 *Demo video: (add link here)*

---

## What it is

A wearable that straps to the **outside of my ankle** and tells, live, whether I'm **standing still, walking, running, or biking** (on a bike, or lying on my back air-biking).

It's an **Arduino UNO Q** with an **MPU6050 IMU** (3-axis accelerometer + 3-axis gyroscope). Two different machine-learning models classify the motion side by side, as the assignment asks:

| | What it's allowed to see | Classifier |
|---|---|---|
| **Model 1** | **One number:** the peak acceleration in one direction on one axis | Decision tree with 4 leaves, so 3 thresholds you can read off |
| **Model 2** | **Everything:** all six IMU outputs (accel x/y/z + gyro x/y/z) | Random forest, 300 trees, 35 features |

Neither one is a neural net. Both answers show up on the UNO Q's LED matrix as two letters (**left = model 1, right = model 2**; S / W / R / B). A dashboard on the laptop shows the details.

---

## Hardware

| Part | Role |
|---|---|
| MPU6050 (GY-521) | Accelerometer set to **±16 g** (running heel strikes pass 8 g at the ankle). Gyro set to **±2000 °/s** (the shank swings fast). 44 Hz low-pass filter so impact peaks stay sharp. |
| Arduino UNO Q, microcontroller side (`sketch.ino`) | Reads the IMU every 20 ms (**50 Hz**) over I²C and sends each reading to the Linux side over the Bridge. Draws the two answer letters on the LED matrix. |
| Arduino UNO Q, Linux side (`main.py`) | Forwards readings to the laptop over **MQTT** and passes the predictions back to the matrix. |
| Laptop (`gaitlib.py`, `train.py`, `app.py`) | MQTT broker (Mosquitto), recording, training, live classification, dashboard. |
| USB power bank | Keeps the board powered while I move. |

Wiring: VCC → 3V3, GND → GND, SDA → SDA, SCL → SCL. The sensor sits flat against the outside of the ankle, with "UP" marked on the case so it goes on the same way each time.

### Data flow

```
MPU6050 --I2C 50 Hz--> UNO Q MCU --Bridge--> UNO Q Linux --MQTT "gait/imu"--> laptop
                                                                               |
                                                    3 s window -> model 1 + model 2
                                                                               |
LED matrix  <--Bridge-- UNO Q Linux <--MQTT "gait/prediction" "m1,m2"----------+
```

Each MQTT message is one reading: `t, ax, ay, az, gx, gy, gz` (m/s² and rad/s). Mosquitto runs with `set_tcp_nodelay` so readings aren't held back and sent in bursts over Wi-Fi.

### Sensor health checks

Two hardware faults showed up during testing, so the system now catches both:

- **Stuck gyro:** after a brown-out, one gyro axis sat at **−373 °/s** while the sensor was still. If any gyro axis averages over **60 °/s**, the dashboard and `record.py` flag it. Unplugging the board fixes it.
- **Loose wire:** the library then returns all zeros. Gravity alone is 1 g, so |a| ≈ 0 is impossible. After 25 zero readings in a row (0.5 s), the sketch stops sending, shows an ✕ on the matrix and reports *"MPU6050 STOPPED RESPONDING."*

---

## What it tracks, and how

### 1. Windows

Live data and training data go through the same steps:

1. **Resample** the stream onto an exact 50 Hz grid, because Wi-Fi delivers readings unevenly.
2. Cut it into **3-second windows** (150 samples), with **a new one every 0.5 s**. 3 s holds at least 2–3 strides even at a slow walk.
3. **Skip any window with a gap over 0.25 s** (a Wi-Fi dropout) instead of interpolating over it.
4. In recordings, **trim 2 s from each end** so starting and stopping don't get learned as the activity.

### 2. Model 1: one peak, three thresholds

**Input:** for one signed axis direction (for example "+z" = how hard the leg is shoved toward +z), take the largest value in the window.

```
peak = max( ±a_axis − median(a_axis) )    over the 3 s window, in g
```

**Why subtract the median:** a raw axis also carries part of gravity, and how much depends on the strap angle. Tilting the sensor just 5° puts about 0.1 g of gravity on an axis that "should" read 0. My first version used the raw value. After I re-strapped the sensor, standing still read 0.10 g. The model had learned "standing ≤ −0.03 g", so standing still showed up as **biking or walking**. Subtracting the window's median removes gravity, and the strap angle with it. The peak is still one axis in one direction, so it's within the rules. This one change took model 1 from **60% to 83%**.

**Choosing the direction:** `train.py` tries all six (±x, ±y, ±z) and keeps the one that scores best on data it never trained on. **+z** won, which with my mounting is the axis that takes the heel-strike shock.

**Classifier:** a `DecisionTreeClassifier` with `max_leaf_nodes = 4` and `class_weight = "balanced"`. On a single number, a 4-leaf tree is just 3 cut points, which come out as readable rules:

| Peak of +z (g) | Prediction |
|---|---|
| ≤ 0.11 | **Standing** |
| 0.11 – 0.46 | **Biking** |
| 0.46 – 1.05 | **Walking** |
| > 1.05 | **Running** |

This matches the physics:

| Activity | Median peak | 10–90 % range | Why |
|---|---|---|---|
| Standing | 0.01 g | 0.00 – 0.09 | No motion, only sensor noise |
| Biking | 0.26 g | 0.16 – 0.56 | The foot moves in a smooth circle with no impacts |
| Walking | 0.73 g | 0.43 – 0.97 | Heel strike |
| Running | 2.15 g | 1.11 – 3.51 | Much harder heel strike |

> 📊 *Insert `models/model1_peak_histogram.png`*

### 3. Model 2: all six outputs, random forest

**Features.** I tried two feature sets and kept the one that scored better on strap-ons it had never seen:

- **`tiltfree` (35 features), the one that won.** These are only quantities that don't change if the sensor is turned on the strap:
  - **Size of the acceleration vector |a|**, measured from the window's median: mean, std, min, max, 10th/90th percentile, range. Using the median cancels a small calibration drift between sessions (1.00 g one day, 1.01 g after re-wiring) that the model had been using to tell *sessions* apart.
  - **Size of the rotation vector |ω|:** the same seven stats.
  - **The leg's main swing axis**, found with **PCA on the gyro**. Instead of trusting the sensor's x/y/z, I take the axis the leg actually rotates about, then measure the spread of rotation about each principal axis, plus the **planarity**: what share of the rotation is about the main axis. Walking and running swing in one plane; pedalling stays in a plane but rotates the whole time.
  - **Spread of the gravity-free acceleration** about its principal axes.
  - **Cadence:** the strongest frequency (FFT, 0.3–6 Hz) and how much of the power it holds, for |a|, |ω| and the main swing axis. Also the **autocorrelation cadence and regularity**. The FFT of a 3 s window only resolves 0.33 Hz steps; the autocorrelation is finer, which matters for a fast walk vs. a slow jog.
  - **Jerk** (the rate of change of |a|): mean and max. Impacts are sharp; pedalling is smooth.
  - **Impacts per second:** |a| crossing 1.5 g.
  - The correlation between |a| and |ω|.
- **`peraxis` (78 features):** per-axis stats, axis-to-axis correlations and gravity direction. On unseen strap-ons it scored 65.7%, against 77.5% for `tiltfree`. It learned *how the sensor was strapped on* instead of *how I moved*.

**Classifier:** `RandomForestClassifier(n_estimators = 300, min_samples_leaf = 2, class_weight = "balanced")`. Each tree trains on a random resample of the windows and looks at random subsets of features, and the forest takes a majority vote. The share of trees voting for each activity is the confidence the dashboard shows.

**Most useful features** (random-forest importance):

| Feature | Importance | What it captures |
|---|---|---|
| `gyro_planarity` | 0.084 | How much of the rotation is about one axis |
| `accdyn_spread1` | 0.066 | Size of the motion along its main direction |
| `amag_freqpow` | 0.060 | How rhythmic the acceleration is |
| `gmag_p90` | 0.057 | How fast the shank rotates at its fastest |
| `amag_p90` | 0.049 | How hard the impacts are |

All five of these come from the gyro or need all three axes, so model 1 can't see any of them.

> 📊 *Insert `models/example_windows.png`*

### 4. Live rules on top of the models

- **Stillness shortcut (model 2 only).** If the last 0.4 s varies by less than 0.08 g in |a| and rotates less than 15 °/s, model 2 says **standing** straight away, instead of waiting ~2 s for the motion to leave the 3 s window. In training it fires on 90% of standing windows and 0% of the others. It uses the gyro, so model 1 doesn't get it.
- **Smoothing.** Each model's displayed answer is the majority of its last 3 predictions (1.5 s), so a single odd window doesn't flicker the display.

---

## Training data

All the data is real recordings from me. None of it is synthetic. Each recording is about 60 s of one activity.

| | Recordings | 3 s windows |
|---|---|---|
| Standing | 4 | 458 |
| Walking | 4 | 460 |
| Running | 4 | 459 |
| Biking (air-biking on my back) | 3 | 344 |
| **Total** | **15**, over 3 strap-on sessions | **1721** |

Files are named `activity_session_time.csv`. Every time I take the wearable off and put it back on, I use a new session name (`strap1`, `2ndround`, `strap3`) so testing can check whether it survives re-strapping.

---

## How I tested it (and why not a random split)

Windows overlap by 2.5 s, so neighbouring windows are almost the same data. A random 80/20 split would test on near-copies of training windows and report accuracy that's too high. I used two stricter tests:

1. **Leave-one-recording-out.** Train on 14 recordings and test on the 15th, for every recording in turn. This answers: *does it work on a new bout of the activity?*
2. **Leave-one-session-out.** Train on two strap-on sessions and test on the third. This answers: *does it still work after I take it off and put it back on?* That's what happens when peers test it. Both models' design choices (model 1's axis, model 2's feature set) were picked using this test.

---

## Results

| | Unseen recording | Unseen strap-on |
|---|---|---|
| **Model 1** (peak of +z) | **83.0 %** | **86.6 %** |
| **Model 2** (random forest + stillness) | **89.5 %** | **77.5 %** |

> 📊 *Insert `models/confusion_matrices.png`*

**Model 1 confusion matrix** (rows = truth, columns = predicted, leave-one-recording-out):

| | Standing | Walking | Running | Biking |
|---|---|---|---|---|
| **Standing** | **422** | 0 | 0 | 36 |
| **Walking** | 2 | **365** | 36 | 57 |
| **Running** | 0 | 52 | **407** | 0 |
| **Biking** | 5 | 97 | 7 | **235** |

**Model 2 confusion matrix:**

| | Standing | Walking | Running | Biking |
|---|---|---|---|---|
| **Standing** | **442** | 16 | 0 | 0 |
| **Walking** | 4 | **411** | 45 | 0 |
| **Running** | 0 | 86 | **372** | 1 |
| **Biking** | 0 | 2 | 27 | **315** |

### What the numbers say

- **Model 2 wins on what model 1 can't see.** Biking goes from 68% to 92% correct. To one peak, pedalling looks like a gentle walk. To the gyro it's obvious: the shank rotates *continuously* in one plane, with no impacts.
- **Model 1's weak spot is biking vs. walking.** 97 biking windows were called walking, because hard pedalling and a slow walk have similar peaks. With only one number, there's no way to separate them.
- **Both models confuse a fast walk with a slow jog.** It's the hardest pair: same cadence, similar impacts. Model 2 called 86 running windows walking. My 10/28 running recordings were harder (average |a| 1.7 g against 1.2 g before), so the speeds in the training data don't overlap as much as they should.
- **Model 2 drops more than model 1 on a new strap-on** (89.5% → 77.5%). Its 35 features can still pick up quirks of a particular session. Model 1's one gravity-free peak has almost nothing session-specific to latch onto, so it holds up. More strap-on sessions is the fix: right now the model only has two sessions to learn from each time.

---

## What I learned

1. **Gravity is the enemy of single-axis features.** A few degrees of tilt moved 0.1 g onto the axis, which was more than the whole gap between standing and biking. Subtracting the window median made model 1 independent of the strap angle and lifted it from 60% to 83%.
2. **Features that ignore orientation beat raw axes** once the sensor gets re-strapped. Taking the main swing axis from the data (PCA) instead of the sensor's x/y/z is what made model 2's features portable.
3. **Test the way it'll be used.** A random split tests on near-copies of the training windows, so it would have scored much higher and hidden every one of these problems. Holding out whole recordings and whole strap-ons is what exposed them.
4. **Check the sensor before blaming the model.** A stuck gyro and a loose wire both produced data that looked like "the model is bad". Now the system checks for both.

### Next steps

- Record more strap-on sessions, and walks and runs at several speeds, to close the fast-walk / slow-jog gap.
- Record real biking (not just air-biking) so the biking class covers both.

---

## Code

```
gait_identifier/
  unoq-gait-app/sketch/sketch.ino   IMU at 50 Hz -> Bridge; LED letters; sensor checks
  unoq-gait-app/python/main.py      Bridge <-> MQTT
  laptop/gaitlib.py                 windows, model 1 peak, model 2 features, live classifier
  laptop/record.py                  record labelled data
  laptop/train.py                   train + cross-validate both models, report + plots
  laptop/live.py                    live classification in the terminal
  laptop/app.py + web/              dashboard: record, live view, train, results
```
