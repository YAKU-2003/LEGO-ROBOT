"""Public gait data for pre-training: IPOL 2023 "A Reference Data Set for the Study of
Healthy Subject Gait with Inertial Measurements Units" (Voisard et al., IPOL 13, 2023,
doi:10.5201/ipol.2023.497).

19 healthy adults, 110 walks (stand ~6 s, walk 10 m, U-turn, walk back, stand). An XSens
IMU on each foot (and the lower back) at 100 Hz: accel in m/s^2 (with gravity), gyro in
rad/s -- the same units as our MPU6050. Every footstep of both feet is marked (toe-off,
heel strike), found by two validated algorithms and then checked by hand.

Each foot sensor becomes one recording on our 50 Hz grid, with the heel strikes of BOTH
feet as its steps (that's what our ankle sensor has to count too).

Download (34 MB): https://www.ipol.im/pub/art/2023/497/497-main.zip
unzipped into step_counter/laptop/public_data/ipol2023/
"""

import glob
import json
import os

import numpy as np
from scipy.signal import resample_poly

import steplib as sl

IPOL_DIR = os.path.join(sl.HERE, "public_data", "ipol2023", "497-main", "GaitData")
IPOL_FS = 100
CITATION = ("Voisard C. et al., A Reference Data Set for the Study of Healthy Subject Gait with Inertial "
            "Measurements Units, Image Processing On Line 13 (2023) 314-320, doi:10.5201/ipol.2023.497")


def ipol_recordings(folder=IPOL_DIR):
    """One recording per foot sensor per walk. Each has the trainlib fields plus
    data [n, 7] (t, ax..gz) so steplib.stream_count can score it like a live walk."""
    out = []
    for meta_path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        strikes = [hs for _, hs in meta["LeftFootEvents"] + meta["RightFootEvents"]]  # [toe-off, heel strike]
        for foot in ("rf", "lf"):
            raw = np.loadtxt(os.path.join(folder, f"{meta['Filename']}_{foot}.txt"), skiprows=2, ndmin=2)
            # 100 Hz -> 50 Hz (anti-aliased). padtype="line": zero-padding would fake a jolt at the start.
            X = resample_poly(raw[:, 1:7], 1, IPOL_FS // sl.FS, axis=0, padtype="line")
            events = np.array(sorted(int(round(s * sl.FS / IPOL_FS)) for s in strikes), int)
            events = events[(events >= 0) & (events < len(X))]
            t = np.arange(len(X)) / sl.FS
            out.append({"name": f"ipol_{meta['Filename']}_{foot}", "subject": int(meta["Subject"]), "foot": foot,
                        "X": X, "inp": sl.net_inputs(X), "ok": np.ones(len(X), bool), "events": events,
                        "target": sl.density_target(len(X), events), "steps": len(events),
                        "data": np.column_stack([t, X]), "minutes": len(X) / sl.FS / 60})
    return out
