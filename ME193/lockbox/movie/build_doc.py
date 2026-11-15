"""
Builds LunchLock_NeuralNet_Explainer.docx: the timed voice-over script for LunchLock_explainer.mp4
plus a "How the neural network works" write-up with frames from the video (for the Notion page).

  python build_doc.py        (run make_movie.py --stills first)
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
STILLS = HERE / "stills"
scenes = json.loads((HERE / "scenes.json").read_text())

# What to say over each scene (first person, ~2.3 words per second so it fits the scene).
SCRIPT = [
    ("Title card.",
     "This is LunchLock. I'll show you how a neural network I built counts the fingers I hold up."),
    ("Three panels: press the button, show 3-1-4, the lid opens.",
     "LunchLock is a small lunch box for a shared fridge. You press the button, show your secret code "
     "with your fingers, three, one, four, and only then does the lid open."),
    ("The camera crop shrinks to 96 × 96 pixels; a zoom shows the pixel values.",
     "The webcam on the Arduino UNO Q looks at a box where you hold your hand. I shrink that picture to "
     "96 by 96 pixels. To the network, every pixel is just a number."),
    ("Photo → hand mask → 12 × 12 grid with the yellow finger-height profile.",
     "First, my own idea. Skin is redder than the wall, so I compare each pixel's redness to the "
     "background and get a hand mask. A twelve by twelve grid measures where the hand is, and the yellow "
     "line shows how high it reaches. Raised fingers are the tall bumps. The mask goes into the network "
     "right next to the photo."),
    ("A 5 × 5 learned filter slides across the image; the feature map fills in.",
     "Next is convolution. A small five by five filter slides across the image. At every spot it "
     "multiplies the pixels by its weights and adds them up into one number. Bright means its pattern is "
     "there, like the edge of a finger. The network learned these weights by itself, and the first layer "
     "has sixteen of them."),
    ("Feature map before ReLU (blue = negative), the ReLU graph, and after ReLU.",
     "Then ReLU. It keeps positive numbers and turns negative ones into zero. That simple bend is what lets "
     "the network learn shapes instead of only straight lines."),
    ("An 8 × 8 corner of the feature map is pooled 2 × 2 into 4 × 4.",
     "Max pooling keeps only the biggest number in every two by two block. The map gets half as wide and "
     "half as tall. So the network cares whether a finger is there, not its exact pixel, and the next layer "
     "has four times less work."),
    ("Real feature maps after each block: 16 at 48 × 48, 32 at 24 × 24, 64 at 12 × 12.",
     "I repeat convolution, ReLU and pooling three times, with sixteen, thirty two and sixty four filters. "
     "That's the width. Three conv layers plus one final layer is the depth. Each block finds bigger "
     "patterns: edges, then finger parts, then raised finger here."),
    ("All features in one row → 6 output neurons → probability bars: '3 fingers'.",
     "All those maps are flattened into one long row of about nine thousand numbers. One linear layer gives "
     "a score for zero to five fingers, and softmax turns the scores into probabilities. Here it's sure: "
     "three fingers."),
    ("The training-loss curve draws; forward and backward passes through the layers.",
     "How did it learn? For every training photo I compare its answer with the right one using cross-entropy "
     "loss. Backpropagation uses the chain rule to work out how much each of the eighty thousand weights "
     "caused the error, and gradient descent nudges them all downhill. After eighty rounds it gets every "
     "held-out photo right."),
    ("Anna's V data: a straight line gets stuck; two ReLU neurons fit the V.",
     "Here's why ReLU matters, using the V data from Anna's code. One neuron without ReLU is a straight line, "
     "y equals m x plus b. Gradient descent moves m and b downhill, but the error gets stuck around four "
     "ninety, because a line can't bend. Two neurons with ReLU make two ramps. Each one learns one arm of "
     "the V, and the error drops to about four."),
    ("Live confidence over time with the 85 % line; the digit boxes fill with 3, 1, 4 → OPEN.",
     "Finally the lock policy. The network guesses ten times a second, but one guess is never trusted. A "
     "digit only counts if the same answer stays above eighty five percent for most of a second. Three, "
     "one, four, open."),
    ("End card.",
     "Trained in PyTorch on my laptop, running on the Arduino UNO Q. Your hand is the key."),
]

assert len(SCRIPT) == len(scenes), "script and scenes out of sync"
_tot = int(sum(s['length_s'] for s in scenes))
TOTAL = f"{_tot // 60}:{_tot % 60:02d}"

FONT = "Arial"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(0.9))

styles = doc.styles
styles["Normal"].font.name = FONT
styles["Normal"].font.size = Pt(11)
styles["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
for name, size, color in (("Title", 26, "1F2A44"), ("Heading 1", 17, "1F2A44"), ("Heading 2", 13, "2F5597")):
    st = styles[name]
    st.font.name, st.font.size, st.font.bold = FONT, Pt(size), True
    st.font.color.rgb = RGBColor.from_string(color)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_col_widths(table, widths):
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = w


def para(text_, bold=False, italic=False, size=None, color=None, after=6, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text_)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    p.paragraph_format.space_after = Pt(after)
    if align:
        p.alignment = align
    return p


def bullet(text_, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text_)
    p.paragraph_format.space_after = Pt(3)
    return p


def figure(path, caption, width=6.4):
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(caption, italic=True, size=9, color="666666", after=10, align=WD_ALIGN_PARAGRAPH.CENTER)


def still(n):
    hits = sorted(STILLS.glob(f"{n:02d}_*.png"))
    return hits[0] if hits else STILLS / "missing.png"


def mmss(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


# ───────────────────────────────────────── title
doc.add_paragraph("LunchLock: How the Neural Network Works", style="Title")
para("Voice-over script for the explainer movie, and the write-up for the Notion page", italic=True,
     color="555555", after=12)

doc.add_heading("How to use this document", level=1)
bullet(f" LunchLock_explainer.mp4 ({TOTAL}, silent, 1280 × 720). Every image in it comes from the real trained "
       "network: my photo, its learned filters, its feature maps, its training curve and the V-data fits.",
       "The video:")
bullet(" record yourself reading the 'What to say' column while the video plays, e.g. a Zoom recording "
       "with the video shared on screen. Each line is written to fit its scene at a relaxed pace; the "
       "start time tells you when to begin.", "Voice-over:")
bullet(" the second half of this document explains the network in prose with frames from the video. "
       "Copy it into the 'Neural Network' section of the Notion page next to the video.", "Notion:")

# ───────────────────────────────────────── script table
doc.add_heading("Voice-over script", level=1)
table = doc.add_table(rows=1, cols=3)
table.style = "Table Grid"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
widths = (Inches(0.95), Inches(1.95), Inches(3.8))
hdr = table.rows[0].cells
for cell, label in zip(hdr, ("Time", "On screen", "What to say")):
    cell.text = ""
    r = cell.paragraphs[0].add_run(label)
    r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    shade(cell, "1F2A44")
for sc, (screen, say) in zip(scenes, SCRIPT):
    row = table.add_row().cells
    row[0].text = ""
    p = row[0].paragraphs[0]
    p.add_run(f"{mmss(sc['start_s'])}\n").bold = True
    t = p.add_run(f"{sc['length_s']} s")
    t.font.size, t.font.color.rgb = Pt(9), RGBColor(0x66, 0x66, 0x66)
    row[1].text = ""
    p = row[1].paragraphs[0]
    p.add_run(f"{sc['scene']}. {sc['name']}\n").bold = True
    s = p.add_run(screen)
    s.font.size = Pt(9.5)
    row[2].text = say
    if sc["scene"] % 2 == 0:
        for c in row:
            shade(c, "F2F5FA")
set_col_widths(table, widths)
para("")
para(f"Total length {TOTAL}. If you speak faster or slower, change a scene's length in SCENES in make_movie.py "
     "and run it again; the times in this table come from scenes.json.", size=9, color="666666")

# ───────────────────────────────────────── write-up
doc.add_page_break()
doc.add_heading("How the neural network works", level=1)
para("LunchLock's network takes one picture of my hand and answers one question: how many fingers, 0 to 5? "
     "It's a convolutional neural network (CNN) that I built and trained from scratch in PyTorch. It has about "
     "80,800 weights, small enough to run about 15 times a second on the Arduino UNO Q.")

doc.add_heading("1. What goes in", level=2)
para("The webcam picture is cropped to the hand box and shrunk to 96 × 96 pixels. Each pixel is three "
     "numbers (red, green, blue), so 27,648 numbers go in, and 6 scores come out.")
figure(still(3), "The UNO Q camera's crop, shrunk to 96 × 96. To the network every pixel is just a number.")

doc.add_heading("2. The hand mask: my 'grid' idea (early fusion)", level=2)
para("Before any learning, every pixel's redness (the Cr channel of YCrCb, divided by brightness) is compared "
     "with the background. Skin is redder than the wall, so the hand lights up as a white silhouette. Because "
     "the comparison is relative to each picture, it barely changes with lighting. From the mask I also "
     "compute a 12 × 12 grid of how much hand covers each cell, and a finger-height profile in which raised "
     "fingers are tall bumps. The mask is stacked next to the grayscale photo, so the very first layer sees "
     "two pictures: the photo and where the hand is. This is called early fusion.")
para("An ablation test showed how much it matters: switching the mask off drops accuracy from 100 % to 33 %. "
     "The grid numbers added at the end turned out to be redundant, because the CNN already learns the same "
     "information from the mask.", italic=True)
figure(still(4), "Photo → hand mask → 12 × 12 grid with the finger-height profile (yellow).")

doc.add_heading("3. Convolution, ReLU and pooling", level=2)
bullet(" a 5 × 5 filter slides over the picture. At each position it multiplies the 25 pixels by its 25 "
       "weights and adds them up, plus a bias. The result is a feature map that is bright wherever the "
       "filter's pattern appears (an edge, a finger outline). The weights are not designed by hand: training "
       "learns them.", "Convolution:")
bullet(" ReLU(z) = max(0, z) keeps positive responses and zeroes negative ones. This bend is what lets "
       "stacked layers represent curved, complicated shapes. Without it, the whole network would collapse "
       "into one straight-line function.", "ReLU:")
bullet(" keeps the largest value in every 2 × 2 block, halving the width and height. The network "
       "then cares whether a feature is present rather than its exact pixel, and the next layer does "
       "4 × less work.", "Max-pooling:")
figure(still(5), "A real learned 5 × 5 filter sliding over my photo and the feature map it produces.")
figure(still(7), "Max-pooling on a real corner of a feature map: each 2 × 2 block keeps its maximum.")

doc.add_heading("4. Width and depth", level=2)
para("Convolution → ReLU → pooling is repeated three times, with 16, 32 and 64 filters. That is the width "
     "of each layer. Three convolution layers plus one linear layer is the depth. Each block sees a bigger "
     "part of the picture and finds bigger patterns: edges, then finger parts, then 'a raised finger here'. "
     "The picture shrinks 96 → 48 → 24 → 12 while the number of maps grows 2 → 16 → 32 → 64.")
table2 = doc.add_table(rows=1, cols=3)
table2.style = "Table Grid"
for cell, label in zip(table2.rows[0].cells, ("Layer", "What it does", "Output size")):
    cell.text = ""
    r = cell.paragraphs[0].add_run(label)
    r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    shade(cell, "1F2A44")
for row in [("Input", "grayscale photo + hand mask", "2 × 96 × 96"),
            ("Conv 5×5, 16 filters + ReLU + max-pool", "edges and blobs", "16 × 48 × 48"),
            ("Conv 3×3, 32 filters + ReLU + max-pool", "finger parts", "32 × 24 × 24"),
            ("Conv 3×3, 64 filters + ReLU + max-pool", "'raised finger here'", "64 × 12 × 12"),
            ("Flatten + 168 grid numbers", "one long row", "9,384"),
            ("Linear layer + softmax", "a probability for each finger count", "6")]:
    cells = table2.add_row().cells
    for c, v in zip(cells, row):
        c.text = v
set_col_widths(table2, (Inches(2.6), Inches(2.4), Inches(1.7)))
para("")
figure(still(8), "The network's real feature maps for my 3-finger photo after each block.")

doc.add_heading("5. The answer: linear layer + softmax", level=2)
para("All 64 maps of 12 × 12 are laid out in one row of 9,216 numbers, plus the 168 grid numbers. One "
     "linear layer gives every one of those numbers its own weight for each of the 6 outputs and adds them "
     "up into a score. Softmax turns the 6 scores into probabilities that add up to 100 %, and the highest "
     "one is the answer.")
figure(still(9), "Flattened features → 6 output neurons → probabilities: '3 fingers'.")

doc.add_heading("6. How it learned: cross-entropy, backpropagation and gradient descent", level=2)
bullet(" for each training photo, loss = −log(probability given to the right answer). Giving the right "
       "answer 25 % costs 1.39, giving it 99 % costs only 0.01. I use it instead of squared error because "
       "the task is to pick a category, not to predict a number.", "Cross-entropy loss:")
bullet(" the chain rule, applied backwards from the loss through every layer, gives the gradient: how "
       "much each of the 80,822 weights contributed to the error.", "Backpropagation:")
bullet(" every weight takes a small step downhill on its gradient. I used Adam, a version of gradient "
       "descent that adapts the step size for each weight.", "Gradient descent:")
bullet(" 3,600 photos through the UNO Q's camera and the laptop webcam, randomly rotated, shifted, mirrored, "
       "re-lit and blurred every epoch. The last 20 % of every burst is held out for testing, so neighbouring "
       "near-duplicate frames can't inflate the score.", "Data:")
bullet(" 80 epochs, trained on the laptop (never on the UNO Q), then exported to NumPy for the board. "
       "100 % on 720 held-out photos through the board's camera. The NumPy copy gives the same answer as "
       "PyTorch on 240 of 240 test photos.", "Result:")
figure(still(10), "Training loss falling over 80 epochs; forward pass (blue) and backpropagation (orange).")

doc.add_heading("7. Why ReLU matters: the V data", level=2)
para("Anna's V data shows ReLU's job most clearly. One neuron without an activation is a straight line, "
     "ŷ = m·x + b. Gradient descent moves m and b downhill, but the error gets stuck at about 491, because "
     "no straight line can go down and then up. Two ReLU neurons are two ramps. Each one learned one arm of "
     "the V, both bending at x = 60, and the error fell to 4.1 (an average miss of about 2). The CNN uses the "
     "same building block, about 80,000 times over.")
figure(still(11), "Two ReLU neurons fitting the V (yellow and blue = each neuron's ramp).")

doc.add_heading("8. From guesses to a lock decision", level=2)
para("The network guesses about 10 times a second, but a single guess is never trusted. A digit is "
     "entered only when the same answer stays at 85 % confidence or more for 0.7 seconds. Changing from 1 to 4 "
     "briefly passes through 2 and 3, which must not count. A fist between two equal digits lets you repeat a "
     "digit. You have 20 seconds after the button, and three wrong codes trigger a 30-second lockout.")
figure(still(12), "Real network confidence while a hand shows 3, 1, 4: red dots are the unsure moments "
                  "between digits.")

out = HERE / "LunchLock_NeuralNet_Explainer.docx"
doc.save(out)
print(f"saved {out.name}")
