"""
Calibrate the LunchLock latch from the laptop (UNO Q app must be running).

  python latch_cli.py

Type a command and press Enter:
  +       nudge open 10 ms (a few degrees)      ++   nudge open 40 ms
  -       nudge closed 10 ms                    --   nudge closed 40 ms
          (anticlockwise can never go past position 0 = closed)
  home    "the latch is closed RIGHT NOW"  -> position = 0
  open    go to the open position              close   go back to (just short of) closed
  setopen use the CURRENT position as the open position
  stop    stop the motor                       q       quit
  b       press the safe's button (when the whole lock runs on the UNO Q)
  snap    save what the board's camera sees in the box -> board_crop.jpg
  s       show the board's status (state, finger guess, fps)

Calibration, once:
  1. Make sure the latch is in the closed position, type  home
  2. Type + / ++ until the lid is free to open (watch it!), then  setopen
  3. Test  close  and  open  a few times. Write the open value into OPEN_MS in sketch.ino
     so it survives a reset.
"""

import time
from pathlib import Path

from latch_link import LatchLink


def main():
    link = LatchLink()
    print("Waiting for the board...", end="", flush=True)
    for _ in range(40):
        if link.board_online:
            break
        time.sleep(0.25)
    print(" online" if link.board_online else " NOT responding (is the UNO Q app running and MQTT_HOST right?)")
    print("Commands: + ++ - -- home open close setopen stop b snap s q")

    jogs = {"+": 10, "++": 40, "-": -10, "--": -40}
    while True:
        cmd = input(f"[pos {link.position_ms} ms] > ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            link.send("stop")
            break
        if cmd in jogs:
            link.send(f"jog {jogs[cmd]}")
        elif cmd == "setopen":
            if link.position_ms:
                link.send(f"openms {link.position_ms}")
                print(f"open position = {link.position_ms} ms  (put OPEN_MS = {link.position_ms} in sketch.ino)")
            else:
                print("position unknown or 0 - jog open first")
        elif cmd in ("home", "open", "close", "stop"):
            link.send(cmd)
        elif cmd == "b":
            link.press_button()
            print("button pressed - show your code to the board's camera")
        elif cmd == "s":
            print(link.board_status or "no status from the board yet")
        elif cmd == "snap":
            link.request_snap()
            for _ in range(20):
                time.sleep(0.25)
                if link.last_crop:
                    break
            if link.last_crop:
                out = Path(__file__).with_name("board_crop.jpg")
                out.write_bytes(link.last_crop)
                print(f"saved {out.name} - this is exactly what the network sees")
            else:
                print("no picture came back (is the lock running on the board?)")
        elif cmd:
            print("commands: + ++ - -- home open close setopen stop b snap s q")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
