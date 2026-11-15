"""
See what the UNO Q's camera sees, and use the lock from the laptop.

  python board_view.py

The board (unoq-lunchlock-app) must be running and connected to this laptop's broker.
Window shows the board's whole camera view with the hand box, what the network thinks
(state, finger count, confidence) and the 3 code boxes - about 5 frames a second.

Keys (click the window first):
  SPACE / b   the safe's button (start entering the code)
  o / c       open / close the lid by hand
  s           save the current view to board_view.jpg
  q           quit
"""

import time

import cv2
import numpy as np

from latch_link import LatchLink


def main():
    link = LatchLink(print_events=False)
    win = "LunchLock - board camera (SPACE = button, q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 560)
    last_keepalive = 0.0
    while True:
        if time.time() - last_keepalive > 2:          # ask the board to keep streaming
            link.keep_preview_on()
            last_keepalive = time.time()

        if link.last_preview and time.time() - link.t_preview < 3:
            img = cv2.imdecode(np.frombuffer(link.last_preview, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = np.zeros((360, 640, 3), np.uint8)
            msg = "waiting for the board..." if not link.board_online else "board online, waiting for camera..."
            cv2.putText(img, msg, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(img, "is unoq-lunchlock-app running and MQTT_HOST = this laptop's IP?",
                        (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        # recent lock events under the picture
        panel = np.zeros((130, img.shape[1], 3), np.uint8)
        for i, (t, text) in enumerate(link.events[-5:]):
            cv2.putText(panel, f"{t}  {text}", (10, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1)
        cv2.imshow(win, np.vstack([img, panel]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord(" "), ord("b")):
            link.press_button()
        elif key == ord("o"):
            link.send("open")
        elif key == ord("c"):
            link.send("close")
        elif key == ord("s") and img is not None:
            cv2.imwrite("board_view.jpg", img)
            print("saved board_view.jpg")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
