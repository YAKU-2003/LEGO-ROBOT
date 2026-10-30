# Gait wearable: standing, walking, running or biking

An Arduino UNO Q with an MPU6050, strapped to the ankle. The board streams the IMU
at 50 Hz to the laptop over MQTT. The laptop sorts each 3 s window into
**standing / walking / running / biking** (biking includes lying on your back and
air-biking) using two models. Neither one is a neural net:

| | Input | Classifier |
|---|---|---|
| **Model 1** | One number per window: the **peak acceleration in one direction on one axis**, such as max(+a_y), measured from the window's median so gravity (and the strap angle) drops out | Decision tree with at most 4 leaves, which works out to 3 thresholds you can read off |
| **Model 2** | Features from **all six IMU outputs**. `train.py` tries two sets and keeps the one that does best on data it never saw (below) | Random forest (300 trees) |

Model 2's two feature sets:
- **`tiltfree`** (35 features, preferred): only quantities that stay the same if the sensor is strapped on at a different angle.
  - \|a\| and \|g\| stats.
  - The leg's main rotation axis, found from the data with PCA on the gyro, and how much it rotates about each principal axis.
  - Cadence from the FFT and from the autocorrelation. The autocorrelation is finer than a 3 s FFT's 0.33 Hz steps, which helps with fast walk vs jog.
  - Jerk and impacts per second.
- **`peraxis`** (78 features): per-axis stats, axis correlations and gravity direction. This can edge out `tiltfree` if the strap angle never changes. It can also learn *how the sensor was strapped* instead of *how you moved*. That trap is what sank a classmate's first model.

The board's LED matrix shows both answers live: **left letter = model 1, right letter = model 2**
(S = standing, W = walking, R = running, B = biking, ? = no data yet).

`live.py` also has a **stillness shortcut** for model 2. Once the last 0.4 s varies by less than 0.08 g with under 15 °/s of rotation, model 2 says standing right away, instead of waiting ~2 s for the motion to leave the window. Model 1 doesn't get the shortcut because it uses the gyro, which model 1 isn't allowed to see.

## Files

```
unoq-gait-app/          App Lab app for the board
  sketch/sketch.ino     reads the MPU6050 at 50 Hz -> Bridge; draws the two letters
  python/main.py        Bridge -> MQTT "gait/imu"; MQTT "gait/prediction" -> LED matrix
laptop/
  gaitlib.py            MQTT link, windowing, the features for both models
  record.py             record labelled data  -> data/<activity>_<session>_<time>.csv
  train.py              train + cross-validate both models -> models/ (+ report.txt, plots)
  live.py               live classification with both models side by side
  app.py + web/         dashboard at http://localhost:8000 that does all three:
                        Record (activity/session buttons, live graph, recordings list),
                        Live (both models, model 1's thresholds, model 2's confidence,
                        accel/gyro graphs, 60 s timeline), Models (train, accuracy, plots).
                        Warns if the gyro gets stuck or the sensor stops responding.
```

## Hardware

- MPU6050 (GY-521): VCC→**3V3**, GND→GND, SDA→SDA, SCL→SCL.
- Mount it on the **outside of the ankle**, flat against the leg, and keep it the same way up every time (mark "UP" on the case). The models learn the orientation.
- The board needs power while you move. Use a USB power bank in a pocket or strapped to the shin.
- In App Lab, add the libraries **Adafruit MPU6050, Adafruit Unified Sensor, Adafruit BusIO**.
- Set `MQTT_HOST` in `python/main.py` to the laptop's IP (run `ipconfig`).

## Steps

```bash
pip install scikit-learn numpy paho-mqtt matplotlib
```

1. Run the app on the board. The matrix shows a dot when the sensor is found and an X when it isn't.
2. **Record data.** Do each activity at least 3 times for about 60 s, ideally at different speeds and on different days. Biking covers a real bike *and* air-biking on your back. Record both if you want the model to handle both.
   ```bash
   cd gait_identifier/laptop
   python record.py standing
   python record.py walking
   python record.py running
   python record.py biking
   ```
   A recording covers at most one activity. You get 5 s to start moving, and 2 s are trimmed from each end.

   **Take it off and strap it back on at least once.** Then record every activity again with a new session name:
   ```bash
   python record.py walking --session strap2
   ```
   Files are saved as `data/<activity>_<session>_<time>.csv`.
3. **Train both models.**
   ```bash
   python train.py
   ```
   This tries all 6 directions (±x, ±y, ±z) for model 1 and both feature sets for model 2, and keeps whichever scores best on unseen data. `--axis +ay` and `--features tiltfree` force a choice. It prints model 1's thresholds, both confusion matrices and model 2's most useful features. It saves `models/report.txt`, `model1_peak_histogram.png`, `confusion_matrices.png` and `example_windows.png` for the writeup.
4. **Run live.**
   ```bash
   python live.py
   ```

## How accuracy is measured

Windows overlap: a new 3 s window starts every 0.5 s. A random train/test split would therefore test on samples it trained on and report accuracy that's too high. `train.py` uses two stricter tests:

1. **Leave-one-recording-out.** Train on every recording but one, test on that one, and repeat for each recording. This needs at least 2 recordings of every activity. With only one, it falls back to holding out 20 s chunks and prints a warning.
2. **Leave-one-session-out.** Train on every strap-on but one, then test on that one. This is the honest answer to "does it still work after I take it off and put it back on?", which is exactly what happens when peers test it. When sessions exist, both models' choices (axis, feature set) are judged on this test.

## What to expect (and to discuss in the writeup)

- **Model 1** separates standing < walking < running well, because heel-strike peaks grow with speed: about 0 g standing, under 1 g walking, 2+ g running above the median at the ankle. **Biking is its weak spot.** Pedalling has no impacts, so its peak sits between standing's and walking's. The peak is measured from the window's median: on a raw axis, a few degrees of strap tilt put ~0.1 g of gravity on it, which was enough to make standing still read as biking after a re-strap.
- **Model 2** also sees the gyro and the cadence. The shank rotates smoothly and continuously when pedalling, and in a jerkier swing when running. That's why it should beat model 1, mostly on biking vs standing.
- **Re-strapping hurts model 1 the most.** If the sensor tilts, gravity moves onto or off the axis it uses, so its thresholds shift. Compare the two SUMMARY lines in `report.txt`.
- **Fast walk vs slow jog** is the hardest pair for model 2. Recording walks and runs at several speeds helps more than anything else.
