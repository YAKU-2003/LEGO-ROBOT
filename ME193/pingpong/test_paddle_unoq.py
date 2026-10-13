"""Test + calibrate the Arduino UNO Q paddle (MPU6050 + light sensor).

Same as test_paddle.py (which does the LEGO Double Motor) with the UNO Q as the
default. Start the "Ping Pong Paddle" app on the board first, then run
(from the lego_robot_project folder):

    my_env\\Scripts\\python.exe ME193\\pingpong\\test_paddle_unoq.py

It shows the HIT / MISS icons on the LED matrix, calibrates your swing
(saved to ME193/pingpong/paddle_config_unoq.json), then prints live readings.
"""

from test_paddle import main

if __name__ == "__main__":
    main(default_paddle="unoq")
