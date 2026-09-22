import time

from mqttlib import MQTTClient

TOPIC = "ME193"
HOST = "test.mosquitto.org"


def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")


with MQTTClient(host=HOST) as client:
    client.subscribe(TOPIC, on_message)
    time.sleep(1)  # give the subscription time to reach the broker

    client.publish(TOPIC, "hello world")
    print(f"Published 'hello world' to '{TOPIC}' on {HOST}")

    time.sleep(1)  # give the message time to come back before disconnecting
