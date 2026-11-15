# LunchLock: a finger-code lunch safe

A small fridge lunch box that opens only when you show your 3-digit code with your fingers
(default **3 → 1 → 4**). A PyTorch CNN counts the fingers. It's **trained on the laptop** and
**runs on an Arduino UNO Q** (as a NumPy copy), which also drives the lid motor and the LED matrix.

ME193 midterm project. The write-up is on the Notion page.

## How it fits together

```
UNO Q (unoq-lunchlock-app/)                          laptop (this folder)
  python/main.py   webcam -> hand box -> FingerNet     train_cnn.py        trains the CNN (CPU or GPU)
                   (NumPy) -> lock_logic.py            export_for_unoq.py  -> python/finger_weights.npz
  sketch/sketch.ino  TT motor (Maker Drive) + LED      app.py              dashboard: live view, capture, retrain
                     matrix, lid never past closed     latch_cli.py        lid calibration, "button"
        |                                              phone_relay.py      live view + button in a browser
        +---------------- MQTT over Wi-Fi ------------ latch_link.py       the MQTT link (starts Mosquitto)
```

## Files

| File | What it does |
|---|---|
| `finger_model.py` | the network: hand mask (early fusion) + 3 conv/pool blocks + grid features + linear layer |
| `train_cnn.py` | training with augmentation and a lighting test; `--branch early/both/cnn/grid`, `--cpu` |
| `lock_logic.py` | the lock policy (85 % confidence, 0.7 s hold, fist to repeat, 20 s, 3 tries → 30 s lockout) |
| `app.py`, `web/index.html` | dashboard at http://localhost:8001 (`--board` = use the UNO Q's camera, `--latch` = drive the real lid) |
| `capture_gestures.py` | the original photo-capture script |
| `export_for_unoq.py` | exports weights, ROI, code and lock policy into the UNO Q app; checks NumPy = PyTorch |
| `unoq-lunchlock-app/` | the App Lab app for the UNO Q (copy the whole folder to the board) |
| `latch_cli.py`, `latch_link.py`, `mosquitto.conf` | lid calibration and the laptop ↔ board link |
| `phone_relay.py`, `board_view.py` | watch the board's camera and press the button from a browser (`:8003`) or a window |
| `ablation.py` | switches parts of the trained network off to measure what helps |
| `lock_pc.py`, `live_test.py` | simple OpenCV-window versions (laptop camera) |
| `vdata/` | gradient descent on Anna's V data: `fit_v.py`, `V_gradient_descent.xlsx`, `MATH.md` |
| `movie/` | `make_movie.py` (animated explainer) and `build_doc.py` (voice-over script + write-up) |
| `finger_cnn.pt`, `model_report.json`, plots | the final trained model and its results |

## Running it

```bash
pip install torch torchvision opencv-python numpy matplotlib paho-mqtt openpyxl python-docx imageio-ffmpeg
python app.py                    # dashboard (laptop camera) -> http://localhost:8001
python train_cnn.py              # retrain on ~/Downloads/gestures/<fingers>_<n>.jpg (photos not in the repo)
python export_for_unoq.py        # then copy unoq-lunchlock-app/ to the board and Run it in App Lab
python latch_cli.py              # lid: home, + / ++, setopen, open, close; b = button
python phone_relay.py            # live view of the board + button -> http://localhost:8003
```

Training photos live in `~/Downloads/gestures/` (not committed). `exclude.txt` lists photos the trainer skips.

## Wiring (UNO Q + Cytron Maker Drive)

- Maker Drive **M1A → pin 6**, **M1B → pin 5**, TT motor on the Motor 1 terminal
- AA pack + / − → Maker Drive VB+ / VB−
- **UNO Q GND → Maker Drive GND** (shared ground: without it the driver can't see the signals)
- Webcam → Arduino USB-C hub on USB-C PD wall power → UNO Q
- Power on with the **lid closed**: the sketch takes that position as "closed" and never drives past it.
