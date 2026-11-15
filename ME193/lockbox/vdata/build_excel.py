"""
Builds V_gradient_descent.xlsx: 5 gradient-descent iterations on m and b, every number a live formula.

  python build_excel.py

Sheet "Gradient descent":
  - inputs (blue): learning rate, starting m and b   -> change them and the whole sheet updates
  - summary table: one row per iteration  (m, b, loss, dL/dm, dL/db, new m, new b)
  - per-point table: for every data point and every iteration, y_hat, error, error^2, error*x
  - charts: loss per iteration, and the data with the line after each iteration
"""

from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference, ScatterChart, Series
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

HERE = Path(__file__).parent
ITERS = 5

d = np.genfromtxt(HERE / "annacode-v.tsv", delimiter="\t", skip_header=1, encoding="utf-8")
d = d[~np.isnan(d).any(axis=1)]
refl, speed = d[:, 1], d[:, 2]          # x (0-100) and output
N = len(refl)

wb = Workbook()
ws = wb.active
ws.title = "Gradient descent"

F = "Arial"
title = Font(name=F, size=14, bold=True)
bold = Font(name=F, bold=True)
norm = Font(name=F)
blue = Font(name=F, color="0000FF")
hdr_fill = PatternFill("solid", fgColor="DDE4F0")
in_fill = PatternFill("solid", fgColor="FFF2CC")
it_fills = ["F3F6FB", "FFFFFF"]
thin = Side(style="thin", color="BBBBBB")
box = Border(top=thin, bottom=thin, left=thin, right=thin)
center = Alignment(horizontal="center", vertical="center", wrap_text=True)

ws["A1"] = "Fitting the V data from Anna's Code with gradient descent: y = m·x + b"
ws["A1"].font = title
ws["A2"] = ("x = Anna's x / 100,  y = output 1.   "
            "Loss L = (1/n)·Σ(m·x + b − y)²,   dL/dm = (2/n)·Σ(m·x + b − y)·x,   dL/db = (2/n)·Σ(m·x + b − y),   "
            "update: m ← m − α·dL/dm,  b ← b − α·dL/db")
ws["A2"].font = Font(name=F, italic=True, color="555555")

# ---- inputs
ws["A4"], ws["A5"], ws["A6"], ws["A7"] = "Learning rate α", "Starting m", "Starting b", "Number of points n"
for r in range(4, 8):
    ws[f"A{r}"].font = bold
ws["B4"], ws["B5"], ws["B6"] = 0.3, 0, 0
for c in ("B4", "B5", "B6"):
    ws[c].font, ws[c].fill, ws[c].border = blue, in_fill, box
ws["B4"].comment = Comment("Input: change me. 0.3 works because x is scaled to 0.2..1; try 1.2 to see it blow up.", "fit")
ws["C4"] = "← yellow cells are inputs: change them and every iteration recalculates"
ws["C4"].font = Font(name=F, italic=True, color="555555")

# ---- per-point table layout
P0 = 19                      # header row of the per-point table
first, last = P0 + 1, P0 + N
ws[f"B7"] = f"=COUNT(B{first}:B{last})"
ws["B7"].font, ws["B7"].border = norm, box

# ---- summary table
S0 = 10
heads = ["Iteration", "m", "b", "Loss (MSE)", "dL/dm", "dL/db", "new m", "new b"]
for j, h in enumerate(heads):
    c = ws.cell(row=S0, column=1 + j, value=h)
    c.font, c.fill, c.alignment, c.border = bold, hdr_fill, center, box


def pcol(k, which):
    """Column letter of per-point column for iteration k (1..5): which in 0..3 = y_hat, err, err^2, err*x."""
    return L(4 + (k - 1) * 4 + which)


for k in range(1, ITERS + 1):
    r = S0 + k
    ws.cell(row=r, column=1, value=k)
    ws.cell(row=r, column=2, value="=B5" if k == 1 else f"=G{r - 1}")
    ws.cell(row=r, column=3, value="=B6" if k == 1 else f"=H{r - 1}")
    ws.cell(row=r, column=4, value=f"=SUM({pcol(k, 2)}{first}:{pcol(k, 2)}{last})/$B$7")
    ws.cell(row=r, column=5, value=f"=2/$B$7*SUM({pcol(k, 3)}{first}:{pcol(k, 3)}{last})")
    ws.cell(row=r, column=6, value=f"=2/$B$7*SUM({pcol(k, 1)}{first}:{pcol(k, 1)}{last})")
    ws.cell(row=r, column=7, value=f"=B{r}-$B$4*E{r}")
    ws.cell(row=r, column=8, value=f"=C{r}-$B$4*F{r}")
    for j in range(1, 9):
        c = ws.cell(row=r, column=j)
        c.font, c.border = norm, box
        c.number_format = "0" if j == 1 else "0.0000"
        c.alignment = Alignment(horizontal="center")
ws.cell(row=S0 + ITERS + 1, column=1, value="Line after 5 steps:").font = bold
ws.cell(row=S0 + ITERS + 1, column=2, value=f"=G{S0 + ITERS}").number_format = "0.0000"
ws.cell(row=S0 + ITERS + 1, column=3, value=f"=H{S0 + ITERS}").number_format = "0.0000"
ws.cell(row=S0 + ITERS + 1, column=4,
        value="(Python keeps going: after 2000 steps m ≈ 15.79, b ≈ 30.51, loss ≈ 490.7 - the best any straight line can do on a V)").font = \
    Font(name=F, italic=True, color="555555")

# ---- per-point table
ws.cell(row=P0 - 1, column=1, value="Every point, every iteration").font = bold
base = ["x (Anna's Code)", "x = x/100", "y = output"]
for j, h in enumerate(base):
    c = ws.cell(row=P0, column=1 + j, value=h)
    c.font, c.fill, c.alignment, c.border = bold, hdr_fill, center, box
for k in range(1, ITERS + 1):
    for w, h in enumerate(["ŷ = m·x+b", "error ŷ−y", "error²", "error·x"]):
        c = ws[f"{pcol(k, w)}{P0}"]
        c.value = f"iter {k}: {h}"
        c.font, c.alignment, c.border = bold, center, box
        c.fill = PatternFill("solid", fgColor=["DDE4F0", "E8EEF7"][k % 2])

for i in range(N):
    r = first + i
    ws.cell(row=r, column=1, value=float(refl[i])).font = blue
    ws.cell(row=r, column=2, value=f"=A{r}/100").number_format = "0.00"
    ws.cell(row=r, column=3, value=float(speed[i])).font = blue
    for k in range(1, ITERS + 1):
        sr = S0 + k
        ws[f"{pcol(k, 0)}{r}"] = f"=$B${sr}*B{r}+$C${sr}"
        ws[f"{pcol(k, 1)}{r}"] = f"={pcol(k, 0)}{r}-C{r}"
        ws[f"{pcol(k, 2)}{r}"] = f"={pcol(k, 1)}{r}^2"
        ws[f"{pcol(k, 3)}{r}"] = f"={pcol(k, 1)}{r}*B{r}"
        for w in range(4):
            c = ws[f"{pcol(k, w)}{r}"]
            c.number_format = "0.000"
            c.fill = PatternFill("solid", fgColor=it_fills[k % 2])
ws.cell(row=last + 2, column=1,
        value="Source: blue values exported from Anna's Code trainer (annacode-v.tsv), 120 points.").font = \
    Font(name=F, italic=True, color="555555")

# column widths
ws.column_dimensions["A"].width = 20
for col in range(2, 4 + ITERS * 4):
    ws.column_dimensions[L(col)].width = 12.5
ws.freeze_panes = f"D{P0 + 1}"

# ---- charts
loss = LineChart()
loss.title = "Loss after each iteration"
loss.y_axis.title, loss.x_axis.title = "MSE", "iteration"
loss.add_data(Reference(ws, min_col=4, min_row=S0, max_row=S0 + ITERS), titles_from_data=True)
loss.set_categories(Reference(ws, min_col=1, min_row=S0 + 1, max_row=S0 + ITERS))
loss.height, loss.width = 7, 12
ws.add_chart(loss, "K3")

sc = ScatterChart()
sc.title = "Data and the line after each iteration"
sc.x_axis.title, sc.y_axis.title = "x / 100", "y"
xs = Reference(ws, min_col=2, min_row=first, max_row=last)
pts = Series(Reference(ws, min_col=3, min_row=first, max_row=last), xs, title="data")
pts.marker.symbol, pts.graphicalProperties.line.noFill = "circle", True
pts.marker.size = 4
sc.series.append(pts)
for k in (1, 3, 5):
    s = Series(Reference(ws, min_col=4 + (k - 1) * 4, min_row=first, max_row=last), xs, title=f"line, iter {k}")
    s.marker.symbol = "none"
    s.smooth = False
    sc.series.append(s)
sc.height, sc.width = 7, 14
ws.add_chart(sc, "S3")

for ch in (loss, sc):
    ch.roundedCorners = False
    ch.legend.overlay = False
    ch.x_axis.delete = False
    ch.y_axis.delete = False

out = HERE / "V_gradient_descent.xlsx"
wb.save(out)
print(f"saved {out.name}: {N} points, {ITERS} iterations")
