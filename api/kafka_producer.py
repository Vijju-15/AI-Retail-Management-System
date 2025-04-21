import json
from confluent_kafka import Producer

conf = {
    'bootstrap.servers': 'localhost:9092'
}
producer = Producer(conf)

def send_to_kafka(topic: str, message):
    def delivery_report(err, msg):
        if err is not None:
            print(f"❌ Delivery failed: {err}")
        else:
            print(f"✅ Message delivered to {msg.topic()} [{msg.partition()}]")

    if isinstance(message, (dict, list)):
        message = json.dumps(message)  # ✅ serialize to JSON string

    producer.produce(topic, message.encode('utf-8'), callback=delivery_report)
    producer.flush()
