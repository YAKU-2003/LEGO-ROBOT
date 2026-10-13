'''
Shared MQTT wrapper for all ME193 examples.

Usage:
    from mqttlib import MQTTClient
'''

import uuid

import paho.mqtt.client as mqtt

BROKER_HOST = "broker.hivemq.com"
BROKER_PORT = 1883


class MQTTClient:
    """Thin wrapper around paho-mqtt for topic subscribe/publish with
    per-topic callbacks. Use as a context manager so the connection is
    always closed cleanly:

        with MQTTClient() as client:
            client.subscribe(TOPIC, on_message)
            client.publish(TOPIC, "hello world")
    """

    def __init__(self, host=BROKER_HOST, port=BROKER_PORT, client_id=None):
        self.host = host
        self.port = port
        self._callbacks = {}

        client_id = client_id or f"me193-{uuid.uuid4().hex[:8]}"
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_message = self._on_message

    def connect(self):
        self._client.connect(self.host, self.port)
        self._client.loop_start()
        return self

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def subscribe(self, topic, callback):
        """Subscribe to topic, calling callback(topic, payload) for each message."""
        self._callbacks[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic, payload):
        self._client.publish(topic, payload)

    def _on_message(self, client, userdata, message):
        callback = self._callbacks.get(message.topic)
        if callback:
            callback(message.topic, message.payload.decode())

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
