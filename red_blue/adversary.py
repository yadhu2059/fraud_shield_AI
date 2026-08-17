import os
import sys

# Patch Python search path to include workspace root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pickle
import random
import json
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from fast_lane.train_fast import load_data_and_preprocess

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
XGB_BASE_PATH = os.path.join(MODEL_DIR, "xgb_base.pkl")
XGB_MODEL_BASE_ALT = os.path.join(MODEL_DIR, "xgb_model_base.pkl")
RETRAIN_API = "http://localhost:8000/api/v1/retrain"

def generate_adversarial_patterns():
    """
    Generates 100 adversarial evasive fraud transactions across three vectors:
    - amount_splitting
    - biometric_emulation
    - spoofed_device_geo
    """
    print("[*] Generating 100 randomized adversarial fraud patterns (Red Team)...")
    random.seed(42)
    np.random.seed(42)
    
    adversarial_data = []
    
    # 1. Amount splitting with jitter (40 patterns)
    for _ in range(40):
        adversarial_data.append({
            "amount": float(random.uniform(100.0, 500.0)),
            "amount_to_avg_ratio": float(random.uniform(0.5, 1.2)),
            "tx_count": int(random.randint(6, 18)),
            "geo_distance": float(random.uniform(0.5, 4.0)),
            "is_new_device": 0,
            "biometric_score": float(random.uniform(0.70, 0.90)),
            "isFraud": 1,
            "vector": "amount_splitting"
        })
        
    # 2. Biometric emulation / spoofing (30 patterns)
    for _ in range(30):
        adversarial_data.append({
            "amount": float(random.uniform(1200.0, 4000.0)),
            "amount_to_avg_ratio": float(random.uniform(2.0, 4.5)),
            "tx_count": int(random.randint(1, 3)),
            "geo_distance": float(random.uniform(0.2, 2.5)),
            "is_new_device": 1,
            "biometric_score": float(random.uniform(0.85, 0.96)),
            "isFraud": 1,
            "vector": "biometric_emulation"
        })
        
    # 3. Spoofed device/geo combinations (30 patterns)
    for _ in range(30):
        adversarial_data.append({
            "amount": float(random.uniform(5000.0, 15000.0)),
            "amount_to_avg_ratio": float(random.uniform(5.0, 12.0)),
            "tx_count": int(random.randint(1, 2)),
            "geo_distance": float(random.uniform(8.0, 25.0)),
            "is_new_device": 1,
            "biometric_score": float(random.uniform(0.40, 0.75)),
            "isFraud": 1,
            "vector": "spoofed_device_geo"
        })
        
    df_adv = pd.DataFrame(adversarial_data)
    return df_adv

def simulate_red_vs_blue():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Weight & Baseline Safety: If the baseline backup xgb_base.pkl is missing, initialize it
    if not os.path.exists(XGB_BASE_PATH):
        import shutil
        initialized = False
        for path in [XGB_MODEL_BASE_ALT, XGB_PATH]:
            if os.path.exists(path):
                shutil.copy(path, XGB_BASE_PATH)
                print(f"[*] Initialized baseline model backup at {XGB_BASE_PATH}")
                initialized = True
                break
        if not initialized:
            print(f"[-] No model weights available at all. Run fast_lane/train_fast.py first.")
            return

    # Load original base model for before-retrain evaluation
    with open(XGB_BASE_PATH, "rb") as f:
        blue_model = pickle.load(f)

    # Generate Red Team attacks (100 rows)
    df_adv = generate_adversarial_patterns()
    
    # 70/30 Stratified split on attack vectors to prevent data leakage
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_adv, test_size=0.30, random_state=42, stratify=df_adv["vector"])
    
    X_adv_train = df_train.drop(columns=["isFraud", "vector"])
    y_adv_train = df_train["isFraud"]
    
    X_adv_test = df_test.drop(columns=["isFraud", "vector"])
    y_adv_test = df_test["isFraud"]

    # Evaluate Blue Team defense on Red Team test attacks (Before retraining)
    preds = blue_model.predict(X_adv_test)
    missed_count = sum(1 for p in preds if p == 0)
    detection_rate_before = (len(preds) - missed_count) / len(preds) * 100.0
    
    print("\n" + "="*50)
    print(" RED VS. BLUE SIMULATION (BEFORE RETRAINING)")
    print("="*50)
    print(f"Total Red Team Evasive Attacks Tested (Held-out Test): {len(preds)}")
    print(f"Blue Team Detected: {len(preds) - missed_count} attacks")
    print(f"Blue Team Missed: {missed_count} attacks (False Negatives)")
    print(f"Initial Detection Rate: {detection_rate_before:.2f}%")
    print("="*50 + "\n")

    # Retrain Blue Team using X_adv_train combined with normal baseline data
    print("[*] Retraining Blue Team model on adversarial feedback loop...")
    
    # Load original background data or generate synthetic balanced fallback
    try:
        X_orig, y_orig, _ = load_data_and_preprocess(num_rows=50000)
    except Exception as e:
        print(f"[-] Could not load original training data: {e}. Generating 2,000 normal background transactions fallback...")
        # Generate 2,000 normal background transactions so scale_pos_weight is non-zero
        np.random.seed(42)
        random.seed(42)
        bg_data = []
        for _ in range(2000):
            bg_data.append({
                "amount": float(np.random.exponential(scale=1000.0)),
                "amount_to_avg_ratio": float(random.uniform(0.7, 1.3)),
                "tx_count": int(random.randint(1, 5)),
                "geo_distance": float(random.uniform(0.0, 1.5)),
                "is_new_device": 0,
                "biometric_score": float(random.uniform(0.80, 0.99)),
                "isFraud": 0
            })
        df_bg = pd.DataFrame(bg_data)
        X_orig = df_bg.drop(columns=["isFraud"])
        y_orig = df_bg["isFraud"]
        
    X_new = pd.concat([X_orig, X_adv_train], ignore_index=True)
    y_new = pd.concat([y_orig, y_adv_train], ignore_index=True)
    
    num_fraud = sum(y_new)
    scale_pos_weight = (len(y_new) - num_fraud) / num_fraud if num_fraud > 0 else 1.0
    
    retrained_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    retrained_model.fit(X_new, y_new)

    # Evaluate After Retraining strictly on held-out test set (unseen adversarial patterns)
    preds_after = retrained_model.predict(X_adv_test)
    missed_after = sum(1 for p in preds_after if p == 0)
    detection_rate_after = (len(preds_after) - missed_after) / len(preds_after) * 100.0

    print("\n" + "="*50)
    print(" RED VS. BLUE SIMULATION (AFTER RETRAINING)")
    print("="*50)
    print(f"Blue Team Detected: {len(preds_after) - missed_after} attacks")
    print(f"Blue Team Missed: {missed_after} attacks")
    print(f"New Detection Rate: {detection_rate_after:.2f}%")
    print(f"Security Improvement: +{(detection_rate_after - detection_rate_before):.2f}%")
    print("="*50 + "\n")

    # Save the retrained model (overwrite active models/xgb_model.pkl)
    with open(XGB_PATH, "wb") as f:
        pickle.dump(retrained_model, f)
    print(f"[+] Adapted Blue Team model successfully saved to {XGB_PATH}")

    # Compute attack breakdown per vector
    attack_breakdown = {}
    for vec in df_test["vector"].unique():
        vec_mask = (df_test["vector"] == vec)
        vec_X = X_adv_test[vec_mask]
        
        # Before
        vec_preds_before = blue_model.predict(vec_X)
        detected_before = sum(1 for p in vec_preds_before if p == 1)
        rate_before = (detected_before / len(vec_preds_before)) * 100.0
        
        # After
        vec_preds_after = retrained_model.predict(vec_X)
        detected_after = sum(1 for p in vec_preds_after if p == 1)
        rate_after = (detected_after / len(vec_preds_after)) * 100.0
        
        attack_breakdown[vec] = {
            "before": float(rate_before),
            "after": float(rate_after)
        }

    # Call FastAPI reload endpoint to hot-swap model in memory
    try:
        response = requests.post(RETRAIN_API, json={}, timeout=2.0)
        if response.status_code == 200:
            print("[+] FastAPI decision engine notified and model reloaded in-memory.")
        else:
            print(f"[-] API retrain notification returned code {response.status_code}")
    except requests.exceptions.RequestException:
        print("[!] FastAPI API is currently offline. Retrained weights will load on next API startup.")

    # Output structured JSON payload for main.py parsing
    result_payload = {
        "initial_detection_rate": float(detection_rate_before),
        "retrained_detection_rate": float(detection_rate_after),
        "improvement": float(detection_rate_after - detection_rate_before),
        "attack_breakdown": attack_breakdown
    }
    print("JSON_OUTPUT:" + json.dumps(result_payload))

if __name__ == "__main__":
    simulate_red_vs_blue()
