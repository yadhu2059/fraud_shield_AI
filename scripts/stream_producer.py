import os
import csv
import json
import time
import random
import requests
from datetime import datetime

CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "PS_20174392719_1491204439457_log.csv"))
API_URL = "http://localhost:8000/api/v1/transaction"

# Try to import kafka for streaming
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

def get_simulated_features(name_orig):
    # Deterministic device/geo generation based on username hash to keep user profiles consistent
    random.seed(hash(name_orig))
    
    # Simulate device fingerprint
    devices = [
        {"device_id": "dev_mac_739281a", "os": "macOS", "browser": "Safari"},
        {"device_id": "dev_win_810928f", "os": "Windows", "browser": "Chrome"},
        {"device_id": "dev_ios_901827b", "os": "iOS", "browser": "Safari Mobile"},
        {"device_id": "dev_and_610298d", "os": "Android", "browser": "Chrome Mobile"}
    ]
    device = random.choice(devices)
    
    # Geolocation: Delhi, India area as default
    base_lat, base_lng = 28.6139, 77.2090
    lat = base_lat + random.uniform(-0.1, 0.1)
    lng = base_lng + random.uniform(-0.1, 0.1)
    
    # Biometric speed score (time-to-type, key interval consistency, mahalanobis score)
    # Fraudsters might have higher speed / bot-like timing or abnormal typing patterns
    biometric_score = random.uniform(0.7, 0.99)
    
    # Reset random seed
    random.seed(None)
    
    return {
        "device_id": device["device_id"],
        "os": device["os"],
        "browser": device["browser"],
        "latitude": lat,
        "longitude": lng,
        "biometric_score": biometric_score,
        "ip_address": f"192.168.1.{random.randint(2, 254)}"
    }

def main():
    # Check Kafka producer connection
    producer = None
    if KAFKA_AVAILABLE:
        try:
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=1000
            )
            print("[+] Connected to Redpanda (Kafka) on localhost:9092.")
        except Exception as e:
            print(f"[-] Failed to connect to Redpanda: {e}. Falling back to REST API streaming.")
    else:
        print("[*] 'kafka-python' library not installed. Falling back to REST API streaming.")

    if not os.path.exists(CSV_PATH):
        print(f"[!] PaySim dataset CSV not found at: {CSV_PATH}")
        print("[*] Generating live simulated transaction stream directly (no download required)...")
        run_synthetic_stream(producer)
        return

    print(f"[*] Starting streaming transaction simulation from {CSV_PATH}...")
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            # Format raw data fields
            payload = {
                "step": int(row["step"]),
                "type": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "isFraud": int(row["isFraud"]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Enrich with behavioral/device metadata
            metadata = get_simulated_features(payload["nameOrig"])
            payload.update(metadata)
            
            # Streaming implementation
            if producer:
                try:
                    producer.send('transactions', payload)
                    print(f"[Stream] Sent Tx from {payload['nameOrig']} to {payload['nameDest']} (Amount: {payload['amount']})")
                except Exception as e:
                    print(f"[-] Stream send failed: {e}. Retrying via API...")
                    post_to_api(payload)
            else:
                post_to_api(payload)
                
            count += 1
            if count >= 100:  # Only stream first 100 for simulation demo, or sleep to rate limit
                print(f"[+] Successfully simulated {count} transactions.")
                break
                
            time.sleep(0.5) # Sleep 500ms between transactions

def run_synthetic_stream(producer):
    count = 0
    while True:
        # Simulate active transaction metrics
        is_fraud = 1 if (count > 0 and count % 8 == 0) else 0
        amount = random.uniform(10.0, 800.0) if is_fraud == 0 else random.uniform(6000.0, 15000.0)
        
        name_orig = f"C{random.randint(1000, 1500)}"
        name_dest = f"C{random.randint(2000, 2500)}"
        
        payload = {
            "step": count // 10 + 1,
            "type": random.choice(["TRANSFER", "CASH_OUT", "PAYMENT"]),
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": random.uniform(1000, 50000),
            "newbalanceOrig": 0.0,
            "nameDest": name_dest,
            "oldbalanceDest": random.uniform(0, 10000),
            "newbalanceDest": 0.0,
            "isFraud": is_fraud,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Enrich with biometrics & device/location
        # Fraud gets anomalous biometrics and device switches
        if is_fraud:
            # Trigger device switch & fast typing (bot-like)
            payload.update({
                "device_id": f"dev_new_{random.randint(100, 999)}",
                "os": random.choice(["Android", "Windows"]),
                "browser": "Chrome",
                "latitude": 28.6139 + random.uniform(2.0, 10.0), # Impossible travel distance
                "longitude": 77.2090 + random.uniform(2.0, 10.0),
                "biometric_score": random.uniform(0.1, 0.35), # Bot-like anomaly
                "ip_address": f"10.20.30.{random.randint(1, 255)}"
            })
        else:
            metadata = get_simulated_features(name_orig)
            payload.update(metadata)
            
        if producer:
            try:
                producer.send('transactions', payload)
                print(f"[Stream] Sent Tx from {payload['nameOrig']} to {payload['nameDest']} (Amount: {payload['amount']})")
            except Exception as e:
                post_to_api(payload)
        else:
            post_to_api(payload)
            
        count += 1
        time.sleep(0.8) # Wait 800ms


def post_to_api(payload):
    try:
        response = requests.post(API_URL, json=payload, timeout=1.0)
        if response.status_code == 200:
            result = response.json()
            decision = result.get("decision", "UNKNOWN")
            score = result.get("trust_score", 0.0)
            reason = result.get("reason", "")
            print(f"[API] Sent Tx | Decision: {decision} | Trust Score: {score:.2f} | Reason: {reason}")
        else:
            print(f"[-] API error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[-] API connection failed (is FastAPI running?): {e}")

if __name__ == "__main__":
    main()
