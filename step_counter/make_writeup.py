"""Builds Step_Counter_Writeup.docx (for importing into Notion) from the real results in laptop/models/.

    python make_writeup.py
"""

import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "laptop", "models")
OUT = os.path.join(HERE, "Step_Counter_Writeup.docx")
REPO = "https://github.com/YAKU-2003/LEGO-ROBOT/tree/main/step_counter"
ACCENT = RGBColor(0x4F, 0x5B, 0xD5)

doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
for name, size in (("Title", 26), ("Heading 1", 17), ("Heading 2", 13.5)):
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.color.rgb = "Calibri", Pt(size), (ACCENT if name != "Title" else None)
    s.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)


def add_link(par, text, url):
    rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                             is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "4F5BD5")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(c)
    rpr.append(u)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    par._p.append(h)


def para(*parts, style=None, align=None, space_after=6):
    """parts: plain strings, ("b", text) bold, ("i", text) italic, ("c", text) code, ("a", text, url) link."""
    p = doc.add_paragraph(style=style)
    for part in parts:
        if isinstance(part, str):
            p.add_run(part)
        elif part[0] == "a":
            add_link(p, part[1], part[2])
        else:
            r = p.add_run(part[1])
            r.bold, r.italic = part[0] == "b", part[0] == "i"
            if part[0] == "c":
                r.font.name = "Consolas"
                r.font.size = Pt(10)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(*parts):
    return para(*parts, style="List Bullet", space_after=2)


def number(*parts):
    return para(*parts, style="List Number", space_after=2)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def table(rows, widths, header=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Inches(widths[j])
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(10)
            if header and i == 0:
                run.bold = True
                shade(cell, "E8EAFB")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(name, caption, width=6.5):
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(("i", caption), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    for r in p.runs:
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def callout(text):
    t = doc.add_table(rows=1, cols=1)
    c = t.cell(0, 0)
    c.width = Inches(6.7)
    shade(c, "F3F4FE")
    c.text = ""
    r = c.paragraphs[0].add_run(text)
    r.font.size = Pt(10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ── real numbers ─────────────────────────────────────────────────────────────
cv = json.load(open(os.path.join(MODELS, "cv_results.json")))
walks = [r for r in cv if r["true"] > 0]
zeros = [r for r in cv if r["true"] == 0]
log = json.load(open(os.path.join(MODELS, "training_log.json")))
n_params = log.get("params", 30369)


def acc(key, rs=walks):
    return 100 * (1 - sum(abs(r[key] - r["true"]) for r in rs) / sum(r["true"] for r in rs))


def fpm(key):
    return sum(r[key] for r in zeros) / sum(r["minutes"] for r in zeros)


# ── title ────────────────────────────────────────────────────────────────────
doc.add_heading("Step Counter: a PyTorch Neural Net on an Ankle IMU", level=0)
para(("b", "ME193 – AI in Mobile Robots · Yash Kulkarni"))
p = para(("b", "Code: "))
add_link(p, "github.com/YAKU-2003/LEGO-ROBOT → step_counter", REPO)
para(("b", "🎥 Demo video: "), ("i", "(add link here)"))
para(("b", "🎬 How-it-works animation (made with Claude): "), ("i", "(add link here – file: step_counter/laptop/models/how_it_works.mp4)"))

callout("TL;DR – The gait wearable (Arduino UNO Q + MPU6050 on the ankle) streams all six IMU channels at 50 Hz "
        "to the laptop over MQTT. A 30k-weight 1-D convolutional neural net, trained in PyTorch on my own counted "
        f"walks, turns that into a live step count shown on a dashboard and on the board's LED matrix. On walks it "
        f"never saw it got {acc('net'):.1f}% of steps right, and in a live test it showed 30 for a walk I counted "
        "as 27–28.")

# ── system ───────────────────────────────────────────────────────────────────
doc.add_heading("1. The system", level=1)
para("This project extends the gait identifier and reuses the same wearable and board app. Nothing is "
     "computed on the ankle except reading the sensor: the board is a sensor + display, the laptop is the brain.")
table([
    ["Part", "What it does"],
    ["MPU6050 (on the outside of the ankle)", "Accelerometer x/y/z (g) and gyroscope x/y/z (rad/s) – all six sensors are used"],
    ["Arduino UNO Q", "Reads the IMU every 20 ms (50 Hz), publishes it on MQTT topic gait/imu, shows the count it gets "
                      "back on gait/steps as digits on its LED matrix"],
    ["Mosquitto (laptop)", "MQTT broker connecting the board and the laptop over Wi-Fi"],
    ["app.py + web dashboard", "Record tab (counted walks), Model tab (train + uncertainty report), Count tab (live count ± uncertainty)"],
    ["StepNet (PyTorch)", "Turns the last 6 s of IMU data into a step density; the count is its sum"],
], [2.2, 4.5])
para(("c", "MPU6050 → UNO Q --MQTT gait/imu (50 Hz)--> laptop: StepNet (PyTorch) --MQTT gait/steps--> UNO Q LED matrix"))

# ── data ─────────────────────────────────────────────────────────────────────
doc.add_heading("2. The data (all real, recorded on my own sensor)", level=1)
para("Ground truth is me: I counted every step out loud while walking and typed the total into the dashboard "
     "when I stopped. Steps count both feet (every footfall), which is what a pedometer reports. "
     "To teach the net what is ", ("i", "not"), " a step, I reused the standing and biking recordings from the gait "
     "identifier project – the same sensor on the same ankle – as 0-step data.")
table([["Recording", "Steps (counted)", "Length", "Used for"]] +
      [[r["name"].rsplit("_", 1)[0], r["true"], f"{r['minutes'] * 60:.0f} s", "steps + timing" if r["name"] != walks[1]["name"] else "total count only"]
       for r in walks] +
      [[f"{len([z for z in zeros if z['kind'] == 'still'])} standing recordings (gait project)", 0,
        f"{sum(z['minutes'] for z in zeros if z['kind'] == 'still'):.1f} min", "what isn't a step"],
       [f"{len([z for z in zeros if z['kind'] == 'bike'])} biking recordings (gait project)", 0,
        f"{sum(z['minutes'] for z in zeros if z['kind'] == 'bike'):.1f} min", "what isn't a step"]],
      [2.6, 1.2, 0.9, 2.0])
para("Placing each step in time: the gyro sees each swing of the strapped leg (= 2 steps, the other foot lands "
     "halfway between). Those times are only trusted if their total matches my count; walk3 (150) didn't match "
     "(the gyro found 170), so it trains the net through its total only. One recording (walk2) came back with all-zero "
     "readings after a loose cable and was excluded, not deleted.")

# ── NN ───────────────────────────────────────────────────────────────────────
doc.add_heading("3. How the neural net works", level=1)
para(("b", "What goes in and what comes out. "),
     "Every sample of the IMU becomes 8 numbers: the three accelerations, the three rotation rates, and the total "
     "size of each (|a| − 1 g and |ω|), so the net doesn't depend on exactly how the strap is rotated. "
     "StepNet is a 1-D convolutional neural net: four Conv1d layers, each with 32 filters that are 9 samples wide, "
     "followed by BatchNorm, ReLU and Dropout. The layers are dilated (1, 2, 4, 8), so each layer looks at a wider "
     "stretch of time than the last, and every output sample ends up “seeing” about 2.4 s of signal – roughly two "
     f"strides. A final 1×1 convolution and a softplus give one non-negative number per sample: a ",
     ("b", "step density"), f". It's about zero between steps and makes a bump around each footfall whose area "
     f"is one step, so the step count is simply the sum of the density. The whole net has {n_params:,} weights.")
para(("b", "How it learns (PyTorch and back-propagation). "),
     "Training takes random 8-second crops of my recordings, 32 at a time. The forward pass produces a density for "
     "each crop, and the loss compares it with the truth in two ways: a shape loss (the density should look like "
     "bumps at the labelled step times) and a count loss ((sum of density − counted steps)²), which is the only term "
     "for recordings whose step times aren't trusted. Calling loss.backward() makes PyTorch run back-propagation: "
     "it applies the chain rule backwards through every layer and works out how much each of the 30k weights "
     "contributed to the error. Adam then nudges every weight a little in the direction that lowers the loss, and "
     "this repeats 1,500 times (about 40 s on the laptop GPU). Each crop is randomly tilted (±25°, as if the strap "
     "were put on differently), scaled and stretched in time, so the net learns steps rather than memorising my "
     "exact recordings.")
para(("b", "How it counts live. "),
     "Every 0.5 s the laptop gives the net the last 6 s of IMU data and adds up the density for the half second "
     "that ended 1 s ago. Using a slightly delayed window means every step is judged with signal on both sides of it "
     "and counted exactly once; the cost is that the display lags about 1 s behind your feet. The count is sent back "
     "to the board over MQTT and drawn on the LED matrix.")
figure("example_density.png", "Real data from walk1: the 6 sensor inputs (top two), and the trained net's step density "
       "vs. the labelled steps (bottom). Each bump is one step.")
figure("training_curve.png", "Training loss (shape + count) going down over the 1,500 back-propagation steps.", 5.8)

# ── uncertainty ──────────────────────────────────────────────────────────────
doc.add_heading("4. Uncertainty analysis", level=1)
para(("b", "Method: leave-one-recording-out. "),
     "For each recording, a brand-new net is trained on all the other recordings and then counts the held-out one "
     "exactly the way the live counter does. So every number below is from data the net never saw. The same "
     "recordings are also counted by two classical pedometers for comparison: an |a| peak counter and 2 × the gyro "
     "leg swings.")
table([["Walk", "True", "Neural net", "|a| peaks", "2 × gyro"]] +
      [[r["name"].rsplit("_", 1)[0], r["true"], f"{r['net']:.1f}", r["accel"], r["gyro"]] for r in walks],
      [2.6, 0.9, 1.1, 1.0, 1.0])
table([
    ["Method", "Accuracy on walks", "False steps / min (standing + biking)"],
    ["Neural net (PyTorch)", f"{acc('net'):.1f}%", f"{fpm('net'):.1f}"],
    ["|a| peak counter", f"{acc('accel'):.1f}%", f"{fpm('accel'):.1f}"],
    ["2 × gyro swings", f"{acc('gyro'):.1f}%", f"{fpm('gyro'):.1f}"],
], [2.4, 1.8, 2.5])
para(("b", "Result: "), f"the net got {acc('net'):.1f}% of steps right on walks it never saw. Its errors have a bias of "
     "+8.2% (it tends to over-count) and a spread of 8.3%, so a count of N steps is N × 1.08 ± 16% (95%, Bland–Altman). "
     "The live page shows this as the ± next to the count.")
bike = max(zeros, key=lambda r: r["net"])
para(("b", "Where it beats the classic pedometers: not-walking. "),
     "Peak counters count every pedal stroke as a step (69–73 'steps' per minute of biking). The net learned that "
     "biking isn't walking: it gave less than 1 step on two of the three biking recordings and about 1 step per "
     f"minute standing. The exception is one biking recording ({bike['net']:.0f} steps when held out) – with only "
     "three biking recordings, holding one out leaves too little biking to learn from, which is why the average "
     f"false-step rate is {fpm('net'):.1f}/min.")
para(("b", "Sources of uncertainty (honest list):"))
bullet(("b", "Too few walks. "), "With 3 counted walks, every held-out test leaves just 2 to learn from, so the ± "
       "is itself very uncertain (the 95% CI on the mean error is 1.2–14.5%). An earlier version trained on only 2 "
       "walks showed ±65 steps for exactly this reason.")
bullet(("b", "Ground truth. "), "My own count is ±1–2 steps, and on walk3 (150) all three methods agree on ~172, "
       "so my count there may be the thing that's off.")
bullet(("b", "Start/stop. "), "The first and last shuffles of a walk and fidgeting while standing add fractions of "
       "a step, which is where most of the over-count bias comes from.")
bullet(("b", "Sensor dropouts. "), "The board sometimes lost Wi-Fi/power mid-recording (22–75 s gaps in the broker "
       "log); those recordings were cut or excluded.")
figure("uncertainty.png", "Left: held-out counts vs. my counts for all three methods. Right: Bland–Altman plot of the "
       "net's % error with the bias and 95% limits.")
figure("false_steps.png", "False steps per minute on the 0-step recordings (standing and biking).", 5.8)

# ── live demo ────────────────────────────────────────────────────────────────
doc.add_heading("5. Live demo", level=1)
para("On the dashboard's Count tab I pressed Start counting and walked while counting out loud. I counted 27–28 "
     "steps; the net showed 30 – 2–3 steps over, consistent with the +8% bias found in the uncertainty analysis. "
     "The count also appears on the UNO Q's LED matrix.")
para(("b", "🎥 Video: "), ("i", "(add link here)"))

# ── experiment ───────────────────────────────────────────────────────────────
doc.add_heading("6. Experiment: pre-training on public data (and why I turned it off)", level=1)
para("Because I only had a few counted walks, I tried transfer learning: pre-train the net on the open IPOL 2023 "
     "healthy-gait dataset (Voisard et al., doi:10.5201/ipol.2023.497 – 19 people, 110 walks, 3,831 hand-checked "
     "heel strikes from foot-mounted IMUs), then fine-tune on my own data.")
bullet("Pre-training worked on the public data itself: 96.2% accuracy on people it never saw (better spread than a "
       "peak counter, 3.1% vs 6.1%).")
bullet("Getting there exposed two real bugs: resampling 100 → 50 Hz with zero padding faked a jolt at the start of "
       "every recording, and the conv layers' zero padding did the same at the edges. Fixing both (line padding, "
       "replicate padding) took the validation from 70% to 96%.")
bullet("But it didn't transfer: the public sensors are on the foot, mine is on the ankle, and the zero-shot net "
       "counted 30–100 'steps' while I stood still. When fine-tuned with only standing/biking data of my own, it "
       "learned the shortcut “the ankle sensor never steps”.")
para("So the final net trains on my recordings only (", ("c", "python train_steps.py"), "), and transfer learning is "
     "an opt-in flag (", ("c", "--transfer"), ") for when I have more data to compare fairly.")

# ── reflection ───────────────────────────────────────────────────────────────
doc.add_heading("7. Reflection", level=1)
doc.add_heading("What worked?", level=2)
para("Turning step counting into a density whose sum is the count. It lets one net both place steps in time and "
     "count them, lets recordings with only a total still train it, and makes live counting a simple running sum. "
     "Reusing the gait identifier's hardware, MQTT link and 0-step recordings also meant the whole pipeline worked "
     "from the first walk.")
doc.add_heading("What didn't, and what I learned?", level=2)
para("Data quantity and quality mattered far more than the model. A disconnected cable gave a recording of zeros, "
     "counting one foot instead of both halved my labels, and a net trained on two walks looked great on one test "
     "and failed completely on the other. Public data looked like an easy fix but came from a different body "
     "location, and a net will happily learn a shortcut (\"ankle = no steps\") if the data allows it. Testing on "
     "held-out recordings is what exposed every one of these problems.")
doc.add_heading("Is the neural net better than a simple pedometer?", level=2)
para(f"On clean walking, not yet: the classic counters scored {acc('accel'):.0f}–{acc('gyro'):.0f}% vs the net's "
     f"{acc('net'):.0f}% with only 3 walks. Where the net already wins is rejecting things that aren't walking – "
     "biking and standing – which is exactly where pedometers fail. With more walks (and stairs) I expect its walking "
     "accuracy to catch up, because unlike a fixed threshold it can keep learning.")
doc.add_heading("What would I do next?", level=2)
bullet("Record 5–6 more counted walks, plus stairs and stop-and-go, so the uncertainty estimate becomes reliable.")
bullet("Fix the power/Wi-Fi dropouts (secure cable, buffer samples on the board).")
bullet("Run the trained net on the UNO Q itself, so the step counter works without the laptop.")
doc.add_heading("How I used Claude", level=2)
para("Claude helped write the PyTorch training code, the dashboard and the uncertainty analysis, found the two "
     "padding bugs, and made the how-it-works animation, which is rendered from the trained net's real activations, "
     "gradients and loss curve. I recorded and counted all the data, ran the tests, and decided which approach "
     "to keep.")

doc.add_heading("References", level=1)
para("Voisard C. et al., “A Reference Data Set for the Study of Healthy Subject Gait with Inertial Measurement "
     "Units,” Image Processing On Line 13 (2023) 314–320, doi:10.5201/ipol.2023.497.")
para("PyTorch – pytorch.org · Mosquitto MQTT broker – mosquitto.org")

doc.save(OUT)
print("Saved", OUT)
