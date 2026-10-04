"""Detect the green minifig in the webcam feed and publish its position over MQTT.

Topic minifig/pos, x/y normalized 0..1 (origin top-left):
  {"found": true, "x": 0.42, "y": 0.61, "conf": 0.88}
  {"found": false}

Test without MQTT first:  python detect_publish.py --no-mqtt
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(Path(__file__).parent / "runs" / "detect" / "green_minifig" / "weights" / "best.pt"))
parser.add_argument("--broker", default="localhost")
parser.add_argument("--port", type=int, default=1883)
parser.add_argument("--topic", default="minifig/pos")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--flip", action="store_true", help="mirror the image horizontally")
parser.add_argument("--rate", type=float, default=10.0, help="max MQTT messages per second")
parser.add_argument("--no-mqtt", action="store_true", help="only show detections, don't publish")
args = parser.parse_args()

model = YOLO(args.weights)

client = None
if not args.no_mqtt:
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {args.camera}")

last_pub = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if args.flip:
        frame = cv2.flip(frame, 1)

    result = model(frame, conf=args.conf, verbose=False)[0]
    h, w = frame.shape[:2]
    annotated = result.plot()

    if len(result.boxes):
        best = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        msg = {"found": True, "x": round(cx / w, 3), "y": round(cy / h, 3),
               "conf": round(float(result.boxes.conf[best]), 2)}
        cv2.circle(annotated, (int(cx), int(cy)), 6, (255, 0, 0), -1)
    else:
        msg = {"found": False}

    now = time.time()
    if client and now - last_pub >= 1.0 / args.rate:
        client.publish(args.topic, json.dumps(msg))
        last_pub = now

    cv2.putText(annotated, json.dumps(msg), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("minifig detection (Esc to quit)", annotated)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
if client:
    client.loop_stop()
