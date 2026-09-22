"""Quick test: connect to a specific LEGO Education Double Motor (identified
by its Connection Card color + serial number) and drive it like a car."""

import time

import legoeducation as le

CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0994"

DRIVE_SPEED = 50  # percent, used for the forward/backward legs
TURN_SPEED = 50  # percent, used for the in-place turns
STEP_DURATION = 1.5  # seconds per maneuver

motor = le.DoubleMotor()

print(f"Scanning for Double Motor (green card, serial {CARD_SERIAL})...")
motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

if not motor.connected:
    print("Could not connect. Make sure the Double Motor is on and in range.")
else:
    print("Connected!")

    try:
        print(f"Driving forward at {DRIVE_SPEED}% for {STEP_DURATION}s...")
        motor.movement_move_tank(DRIVE_SPEED, DRIVE_SPEED, blocking=False)
        time.sleep(STEP_DURATION)

        print("Turning left in place...")
        motor.movement_move_tank(-TURN_SPEED, TURN_SPEED, blocking=False)
        time.sleep(STEP_DURATION)

        print("Turning right in place...")
        motor.movement_move_tank(TURN_SPEED, -TURN_SPEED, blocking=False)
        time.sleep(STEP_DURATION)

        print(f"Driving backward at {DRIVE_SPEED}% for {STEP_DURATION}s...")
        motor.movement_move_tank(-DRIVE_SPEED, -DRIVE_SPEED, blocking=False)
        time.sleep(STEP_DURATION)
    finally:
        # Always stop and disconnect, even if a maneuver above raises, so the
        # motor never keeps running and the BLE connection is never left open.
        print("Stopping...")
        motor.movement_stop()

        motor.disconnect()
        print("Disconnected.")
