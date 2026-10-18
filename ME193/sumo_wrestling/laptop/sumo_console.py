# Laptop side of the sumo wrestler:
#   - reads the LEGO colour sensor over Bluetooth and publishes it on "sumo/color"
#   - shows live telemetry from the UNO Q and sends commands on "sumo/cmd"
# Start the broker first:  mosquitto -c mosquitto.conf -v
#
#   python sumo_console.py [--broker 127.0.0.1] [--card-serial 0994] [--card-color green]
#
# Keys:  g = go (auto, 3 s start delay)   space/x = STOP   t = motor test   q = quit
import argparse
import json
import msvcrt
import threading
import time

import legoeducation as le
import paho.mqtt.client as mqtt

from lelib import colorSensor

KEYS = {"g": "auto", " ": "stop", "x": "stop", "t": "test"}
BLUE_NAMES = {"Blue", "Azure"}  # colour names that count as the ring edge
COLOR_HZ = 20

state = {"tele": None, "last": 0.0, "color": "--"}


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"broker connected (rc={reason_code}); waiting for the UNO Q...")
    client.subscribe([("sumo/telemetry", 0), ("sumo/status", 0)])


def on_message(client, userdata, msg):
    if msg.topic == "sumo/status":
        print(f"\nUNO Q: {msg.payload.decode(errors='replace')}")
        return
    try:
        state["tele"] = json.loads(msg.payload)
        state["last"] = time.time()
    except ValueError:
        pass


def color_loop(client, sensor):
    while True:
        try:
            name = sensor.detect_color()
        except Exception as e:  # sensor dropped -- UNO Q stops on its own when readings go stale
            state["color"] = f"ERROR {e}"
            time.sleep(0.5)
            continue
        state["color"] = name
        client.publish("sumo/color", json.dumps({"color": name, "blue": name in BLUE_NAMES}))
        time.sleep(1 / COLOR_HZ)


def line(t):
    flags = ("BLUE " if t["blue"] else "") + ("HIT " if t["hit"] else "")
    return (f"{t['rate']:3.0f}/s {t['mode']:8s} -> {t['decision']:13s} "
            f"a=({t['ax']:5.1f},{t['ay']:5.1f},{t['az']:5.1f}) gz={t['gz']:5.2f} "
            f"sensor={state['color']:9s} {flags}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--card-serial", default="0994", help="serial on the LEGO connection card")
    ap.add_argument("--card-color", default="green", help="colour of the LEGO connection card")
    args = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sumo-console")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, 1883, keepalive=10)
    client.loop_start()

    print("Connecting to the colour sensor...")
    sensor = colorSensor()
    sensor.connect(card_serial=args.card_serial,
                   card_color=getattr(le, f"LEGO_COLOR_{args.card_color.upper()}"))
    print("colour sensor connected")
    threading.Thread(target=color_loop, args=(client, sensor), daemon=True).start()

    print("g=go  space/x=STOP  t=motor test  q=quit\n")
    try:
        while True:
            while msvcrt.kbhit():
                k = msvcrt.getwch().lower()
                if k == "q":
                    raise KeyboardInterrupt
                if k in KEYS:
                    client.publish("sumo/cmd", KEYS[k])
                    print(f"\n>> sent {KEYS[k]}")
            t = state["tele"]
            if t is None:
                status = f"sensor={state['color']}  no telemetry yet (is the UNO Q app running with the right MQTT_HOST?)"
            elif time.time() - state["last"] > 1:
                status = f"sensor={state['color']}  telemetry STALE -- UNO Q disconnected?"
            else:
                status = line(t)
            print("\r" + status[:150].ljust(150), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        client.publish("sumo/cmd", "stop")
        time.sleep(0.2)
        print("\nsent stop, bye")


if __name__ == "__main__":
    main()
