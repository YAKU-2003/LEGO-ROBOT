"""Step 1: test a paddle and calibrate its swing detection.

    1. Connects to the paddle:
         --paddle lego (default)  LEGO Double Motor over Bluetooth
         --paddle unoq            Arduino UNO Q + MPU6050 over MQTT -- or just run
                                  test_paddle_unoq.py (start the "Ping Pong Paddle"
                                  app on the board first)
    2. Feedback test: UNO Q shows the HIT / MISS icons on its LED matrix,
       LEGO buzzes the HIT / MISS patterns.
    3. Calibration: hold the paddle still, then swing it 5 times. Saves the
       swing threshold + your normal swing strength (used for SMASH) to
       paddle_config_unoq.json / paddle_config.json -- the game loads it.
    4. Live test (30 s, or press Enter to finish): shows the gyro (+ light
       sensor) and prints / flashes / buzzes on every swing it detects.

Run (from the lego_robot_project folder):
    my_env\\Scripts\\python.exe pingpong\\test_paddle.py
    my_env\\Scripts\\python.exe pingpong\\test_paddle.py --card-serial 7572 --card-color orange
    my_env\\Scripts\\python.exe pingpong\\test_paddle.py --skip-calibration
"""

import argparse
import statistics
import time

from paddle import Paddle, UnoQPaddle, load_threshold, save_config

LIVE_S = 30  # live test length (or press Enter to finish early)


def enter_pressed():
    """True if Enter was pressed (Windows console; elsewhere the live test just runs LIVE_S)."""
    try:
        import msvcrt
    except ImportError:
        return False
    while msvcrt.kbhit():
        if msvcrt.getwch() in "\r\n":
            return True
    return False


def collect(paddle, seconds):
    t0 = time.monotonic()
    time.sleep(seconds)
    return paddle.samples_since(t0)


def find_peaks(samples, floor, min_gap_s=0.3):
    """Local maxima of gyro magnitude above `floor`, at least min_gap_s apart."""
    peaks = []
    for i in range(1, len(samples) - 1):
        t, g = samples[i][0], samples[i][7]
        if g > floor and g >= samples[i - 1][7] and g >= samples[i + 1][7]:
            if peaks and t - peaks[-1][0] < min_gap_s:
                if g > peaks[-1][1]:
                    peaks[-1] = (t, g)
            else:
                peaks.append((t, g))
    return [g for _, g in peaks]


def calibrate(paddle, units):
    input("\nCALIBRATION 1/2: hold the paddle STILL, then press Enter...")
    print("  hold still for 2 seconds...")
    rest = collect(paddle, 2.0)
    if len(rest) < 10:
        print(f"  only got {len(rest)} samples -- is the paddle sending data?")
        return None
    rest_g = [s[7] for s in rest]
    rest_a = [(s[1] ** 2 + s[2] ** 2 + s[3] ** 2) ** 0.5 for s in rest]
    print(f"  got {len(rest)} samples ({len(rest) / 2:.0f} per second)")
    print(f"  at rest: gyro |g| mean {statistics.mean(rest_g):.3g}, max {max(rest_g):.3g} {units}"
          f"   accel |a| mean {statistics.mean(rest_a):.3g}")

    input("\nCALIBRATION 2/2: press Enter, then make 5 normal ping-pong SWINGS (8 seconds)...")
    print("  GO! swing 5 times...")
    saved_threshold = paddle.threshold
    paddle.threshold = float("inf")  # don't detect swings during calibration
    swing = collect(paddle, 8.0)
    paddle.threshold = saved_threshold
    rest_max = max(rest_g)
    biggest = max((s[7] for s in swing), default=0.0)
    peaks = find_peaks(swing, floor=max(rest_max * 3, 0.3 * biggest))
    print(f"  found {len(peaks)} swing peaks: {[float(f'{p:.3g}') for p in peaks]}")

    if len(peaks) < 3:
        print("  Not enough clear swings found. Try again with bigger swings.")
        return None

    typical = statistics.median(peaks)
    # 40% of the way from "still" to a typical swing: low enough that softer
    # swings still count, high enough that waving the paddle around doesn't.
    threshold = rest_max + 0.4 * (typical - rest_max)
    print(f"  typical swing peak {typical:.3g} {units}  ->  swing threshold = {threshold:.3g} {units}")
    print(f"  (a SMASH is a swing peaking above {typical * 1.35:.3g} {units})")
    save_config({
        "swing_threshold": round(threshold, 3),
        "units": units,
        "rest_gyro_max": round(rest_max, 3),
        "rest_accel_mean": round(statistics.mean(rest_a), 3),
        "swing_peaks": [round(p, 3) for p in peaks],
    }, paddle.config_path)
    print(f"  saved to {paddle.config_path}")
    paddle.typical_peak = typical
    return threshold


def main(default_paddle="lego"):
    """default_paddle: "lego" here; test_paddle_unoq.py calls main("unoq")."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddle", choices=("lego", "unoq"), default=default_paddle)
    parser.add_argument("--broker", default="localhost", help="MQTT broker for the UNO Q (default: this laptop)")
    parser.add_argument("--card-serial", default="7572", help="Double Motor connection card serial")
    parser.add_argument("--card-color", default="orange", help="Double Motor connection card color")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    if args.paddle == "lego":
        paddle, units = Paddle(card_serial=args.card_serial, card_color=args.card_color), "raw"
        print("Connecting to Double Motor (turn it on, press its button so the light blinks)...")
        paddle.connect()
    else:
        paddle, units = UnoQPaddle(broker=args.broker), "rad/s"
        print("Connecting to the UNO Q paddle over MQTT...")
        paddle.connect()
        print("Waiting for data -- start the 'Ping Pong Paddle' app on the UNO Q in App Lab...")
        while not paddle.connected:
            time.sleep(0.2)
        time.sleep(1.2)
        print(f"Receiving {paddle.rate:.0f} samples/s from the UNO Q (light sensor = {paddle.light:.0f})")
    print("Connected!")

    try:
        time.sleep(0.5)
        what = "LED matrix" if args.paddle == "unoq" else "haptic"
        print(f"\n{what} test: HIT...")
        paddle.buzz("hit")
        time.sleep(1.0)
        print(f"{what} test: MISS...")
        paddle.buzz("miss")
        time.sleep(1.5)

        if not args.skip_calibration:
            threshold = calibrate(paddle, units)
            if threshold is None:
                threshold = load_threshold(paddle.config_path)
            if threshold:
                paddle.threshold = threshold
        print(f"\nSwing threshold in use: {paddle.threshold:.3g} {units}")

        print(f"\nLIVE TEST ({LIVE_S} s, or press Enter to finish): swing the paddle -- each detected "
              "swing is printed" + ("; cover the light sensor to test the button." if args.paddle == "unoq" else ".")
              + "\n")
        last_swing = paddle.swings[-1] if paddle.swings else None
        count = presses = 0
        t_end = time.monotonic() + LIVE_S
        while time.monotonic() < t_end and not enter_pressed():
            # One short status line, rewritten in place (kept narrow so the terminal doesn't wrap it).
            left = t_end - time.monotonic()
            gmag = paddle.gyro_mag()
            bar = "#" * min(20, int(20 * gmag / (paddle.threshold * 2)))
            light = ""
            if getattr(paddle, "light", None) is not None:
                light = f"  light {paddle.light:4.0f} {'COVERED' if paddle.covered else 'open   '}"
            print(f"\r  |g| {gmag:6.2f} {bar:<20}{light}  [{left:4.1f}s]", end="", flush=True)

            newest = paddle.swings[-1] if paddle.swings else None
            if newest != last_swing:
                last_swing = newest
                count += 1
                time.sleep(0.15)  # let the swing's peak arrive before measuring it
                power = paddle.swing_power(newest)
                print(f"\r  SWING #{count}!  {power:.2f}x your normal swing{'  -> SMASH' if power >= 1.35 else ''}"
                      + " " * 20)
                paddle.buzz("smash" if power >= 1.35 else "hit")
            if paddle.serve_pressed():
                presses += 1
                print("\r  LIGHT SENSOR COVERED (= serve / rematch button)" + " " * 20)
                paddle.buzz("point")
            time.sleep(0.1)
        print(f"\n\nDone: {count} swings detected"
              + (f", light sensor pressed {presses} time(s)." if args.paddle == "unoq" else "."))
    except KeyboardInterrupt:
        print("\nQuitting...")
    finally:
        paddle.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
