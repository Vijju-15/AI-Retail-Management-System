from kafka import KafkaConsumer
import json
from predict_forecast import make_forecast
from predict_pricing import make_pricing_prediction
from predict_segment import segment_customer

def safe_json_deserializer(m):
    try:
        return json.loads(m.decode("utf-8"))
    except Exception as e:
        print(f"❌ Failed to decode message: {m} → {e}")
        return None

consumer = KafkaConsumer(
    'forecast_requests',
    'price_optimization',
    'customer_events',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='retail-ai-group',
    value_deserializer=safe_json_deserializer
)

print("✅ Listening to topics: forecast_requests, price_optimization, customer_events")

for message in consumer:
    data = message.value
    if data is None:
        continue

    topic = message.topic
    print(f"\n📩 Received on topic '{topic}': {data}")

    if topic == 'forecast_requests':
        result = make_forecast(data)
        print(f"🧠 Forecast Result: {result}")

    elif topic == 'price_optimization':
        result = make_pricing_prediction(data)
        print(f"💰 Pricing Prediction: {result}")

    elif topic == 'customer_events':
        result = segment_customer(data)
        print(f"📊 Customer Segment: {result}")

    else:
        print("⚠️ Unknown topic. No handler defined.")
