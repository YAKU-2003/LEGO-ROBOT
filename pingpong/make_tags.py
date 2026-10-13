"""Make printable AprilTags (tag36h11) for the level select: 1 = EASY, 2 = MEDIUM, 3 = HARD.

Run:  my_env\\Scripts\\python.exe pingpong\\make_tags.py
Then print pingpong\\tags\\tag_1.png etc. (keep the white border around each tag).
"""

import os

import cv2
import numpy as np
from cv2 import aruco

OUT = os.path.join(os.path.dirname(__file__), "tags")
LABELS = {1: "EASY", 2: "MEDIUM", 3: "HARD"}
SIZE = 600  # pixels of the black tag itself

os.makedirs(OUT, exist_ok=True)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
for tag_id, label in LABELS.items():
    tag = aruco.generateImageMarker(dictionary, tag_id, SIZE)
    border = SIZE // 6
    page = np.full((SIZE + 2 * border + 80, SIZE + 2 * border), 255, dtype=np.uint8)
    page[border:border + SIZE, border:border + SIZE] = tag
    cv2.putText(page, f"tag {tag_id} = {label}", (border, SIZE + 2 * border + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3, cv2.LINE_AA)
    path = os.path.join(OUT, f"tag_{tag_id}.png")
    cv2.imwrite(path, page)
    print(f"wrote {path}")
