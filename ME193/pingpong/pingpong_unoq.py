"""Virtual Ping-Pong with the Arduino UNO Q paddle (MPU6050 + light sensor).

The same game as pingpong.py (which uses the LEGO Double Motor) -- this file
just makes the UNO Q the default paddle. Everything else (levels, opponents,
points, MQTT score, sounds, keys) is identical and shared.

Before running:
    1. On the UNO Q, run the "Ping Pong Paddle" app (ME193/pingpong/unoq-paddle-app)
       from Arduino App Lab. Its MQTT_HOST must be this laptop's IP.
    2. Calibrate once:  my_env\\Scripts\\python.exe ME193\\pingpong\\test_paddle_unoq.py

Run (from the lego_robot_project folder):
    my_env\\Scripts\\python.exe ME193\\pingpong\\pingpong_unoq.py
    my_env\\Scripts\\python.exe ME193\\pingpong\\pingpong_unoq.py --hand right --mute

This laptop runs the MQTT broker (Mosquitto) the board connects to; the game
starts it automatically if it isn't already running. LED-matrix icons on the
board replace the Double Motor's buzz, and covering the light sensor = start /
rematch at your last level.
"""

from pingpong import main

if __name__ == "__main__":
    main(default_paddle="unoq")
