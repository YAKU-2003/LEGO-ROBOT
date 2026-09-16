"""Quick test: connect to a LEGO Education Single Motor and spin it."""

import legoeducation as le

motor = le.SingleMotor()

print("Scanning for Single Motor...")
motor.connect()

if not motor.connected:
    print("Could not connect. Make sure the Single Motor is on and in range.")
else:
    print("Connected!")

    print("Running motor forward at 50% speed for 2 seconds...")
    motor.motor_run_for_time(2000, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=50)

    print("Rotating motor 360 degrees...")
    motor.motor_run_for_degrees(360, speed=30)

    motor.motor_stop()
    motor.disconnect()
    print("Disconnected.")
