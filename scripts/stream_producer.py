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
    # Deterministic base profiles to keep users grounded in their home parameters
    random.seed(hash(name_orig))
    
    # Base user location
    base_lat, base_lng = 28.6139, 77.2090
    user_lat = base_lat + random.uniform(-0.1, 0.1)
    user_lng = base_lng + random.uniform(-0.1, 0.1)
    
    # Regular device list
    regular_devices = [
        {"device_id": f"dev_{hash(name_orig)%1000}_a", "os": "Windows", "browser": "Chrome"},
        {"device_id": f"dev_{hash(name_orig)%1000}_b", "os": "Android", "browser": "Chrome Mobile"}
    ]
    
    # Reset random seed to allow natural, runtime probability variations
    random.seed(None)
    
    # 1. Device variation: 10% chance a safe user logs in from a new device
    if random.random() < 0.10:
        device_id = f"dev_{random.randint(1000, 9999)}_new"
        os_name = random.choice(["iOS", "macOS", "Windows"])
        browser_name = random.choice(["Safari", "Firefox", "Chrome"])
    else:
        d = random.choice(regular_devices)
        device_id = d["device_id"]
        os_name = d["os"]
        browser_name = d["browser"]
        
    # 2. Location variation: 5% chance user is traveling (Mumbai coordinates, ~1000km shift)
    if random.random() < 0.05:
        lat = 19.0760 + random.uniform(-0.1, 0.1)
        lng = 72.8777 + random.uniform(-0.1, 0.1)
    else:
        lat = user_lat + random.uniform(-0.005, 0.005)
        lng = user_lng + random.uniform(-0.005, 0.005)
        
    # 3. Biometrics variation: typing speed fluctuates naturally between 0.45 and 0.99
    # (Sometimes they type a bit slower or are momentarily distracted)
    biometric_score = random.uniform(0.45, 0.99)
    
    return {
        "device_id": device_id,
        "os": os_name,
        "browser": browser_name,
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
        raise FileNotFoundError(
            f"PaySim dataset CSV not found at: {CSV_PATH}. "
            "Please run 'python scripts/setup_data.py' or download the dataset manually first before running the stream."
        )

    print(f"[*] Starting streaming transaction simulation from {CSV_PATH}...")
    
    # Pre-scan the CSV file to load a pool of 100 normal transactions and 20 fraud transactions
    print("[*] Pre-scanning CSV to load a mix of normal and fraud transactions...")
    normal_pool = []
    fraud_pool = []
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_fraud = int(row["isFraud"])
            if is_fraud == 1 and len(fraud_pool) < 20:
                fraud_pool.append(row)
            elif is_fraud == 0 and len(normal_pool) < 100:
                normal_pool.append(row)
                
            if len(fraud_pool) >= 20 and len(normal_pool) >= 100:
                break
                
    print(f"[+] Loaded {len(normal_pool)} normal and {len(fraud_pool)} fraud transactions from CSV.")
    
    # Interleave normal and fraud transactions
    mixed_transactions = []
    normal_idx = 0
    fraud_idx = 0
    
    for i in range(100):
        # Every 8th transaction, inject a fraud transaction if available
        if i > 0 and i % 8 == 0 and fraud_idx < len(fraud_pool):
            mixed_transactions.append(fraud_pool[fraud_idx])
            fraud_idx += 1
        elif normal_idx < len(normal_pool):
            mixed_transactions.append(normal_pool[normal_idx])
            normal_idx += 1
            
    # Stream the mixed transactions
    for count, row in enumerate(mixed_transactions):
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
        if payload["isFraud"] == 1:
            # Fraud gets anomalous biometrics, device switches, and travel anomaly
            payload.update({
                "device_id": f"dev_fraud_{random.randint(100, 999)}",
                "os": random.choice(["Android", "Windows", "iOS"]),
                "browser": "Chrome",
                "latitude": 28.6139 + random.uniform(5.0, 15.0),  # Impossible travel distance
                "longitude": 77.2090 + random.uniform(5.0, 15.0),
                "biometric_score": random.uniform(0.05, 0.35),  # Bot-like speed/anomalous typing
                "ip_address": f"10.20.30.{random.randint(1, 255)}"
            })
        else:
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
            
        time.sleep(0.8)  # Sleep 800ms between transactions
        
    print(f"[+] Successfully simulated {len(mixed_transactions)} mixed transactions.")

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
