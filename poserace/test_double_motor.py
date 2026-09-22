"""Quick test: connect to a LEGO Education Double Motor and drive it
interactively from a REPL -- type a command, press Enter, see it happen."""

import legoeducation as le

CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0994"

DEFAULT_SPEED = 50  # percent, used until you set a different one

HELP = """
Commands:
  w        drive forward
  s        drive backward
  a        turn left in place
  d        turn right in place
  x        stop
  <number> set the speed percent used by w/s/a/d (e.g. "30")
  h        show this help again
  q        stop, disconnect, and quit
"""

motor = le.DoubleMotor()

print(f"Scanning for Double Motor (green card, serial {CARD_SERIAL})...")
motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)

if not motor.connected:
    print("Could not connect. Make sure the Double Motor is on and in range.")
else:
    print("Connected!")
    print(HELP)

    speed = DEFAULT_SPEED
    try:
        while True:
            command = input(f"[speed {speed}%] > ").strip().lower()

            if command == "":
                continue
            elif command == "w":
                motor.movement_move_tank(speed, speed, blocking=False)
                print("Driving forward...")
            elif command == "s":
                motor.movement_move_tank(-speed, -speed, blocking=False)
                print("Driving backward...")
            elif command == "a":
                motor.movement_move_tank(-speed, speed, blocking=False)
                print("Turning left...")
            elif command == "d":
                motor.movement_move_tank(speed, -speed, blocking=False)
                print("Turning right...")
            elif command == "x":
                motor.movement_stop()
                print("Stopped.")
            elif command == "h":
                print(HELP)
            elif command == "q":
                break
            elif command.lstrip("-").isdigit():
                speed = max(0, min(100, int(command)))
                print(f"Speed set to {speed}%.")
            else:
                print(f"Unknown command {command!r}. Type 'h' for help.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Always stop and disconnect, even if the loop above raises or is
        # interrupted, so the motor never keeps running and the BLE
        # connection is never left open.
        print("Stopping...")
        motor.movement_stop()

        motor.disconnect()
        print("Disconnected.")
