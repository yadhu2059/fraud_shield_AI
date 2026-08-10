import os
import sys

# Patch Python search path to include workspace root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pickle
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from fast_lane.train_fast import load_data_and_preprocess

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
RETRAIN_API = "http://localhost:8000/api/v1/retrain"

def generate_adversarial_patterns():
    """
    Generates adversarial evasive fraud transactions designed to bypass XGBoost.
    Examples:
        - Split transfers: small amounts that don't trigger large limit rules but occur in rapid succession (high velocity).
        - Biometric spoofing: fraud with highly consistent human-like biometric dynamics.
        - Geographic spoofing: transaction located near the user's past location but initiated from a new device.
    """
    print("[*] Generating 20 adversarial fraud patterns (Red Team)...")
    
    adversarial_data = []
    
    # Pattern 1: Amount splitting (high velocity, small amounts)
    for i in range(10):
        adversarial_data.append({
            "amount": 250.0, # Small amount
            "amount_to_avg_ratio": 1.1, # Close to user average
            "tx_count": 8 + i, # Rapid sequence
            "geo_distance": 5.0, # Normal location
            "is_new_device": 0, # Spoofed device
            "biometric_score": 0.85, # Mimics human biometrics
            "isFraud": 1
        })
        
    # Pattern 2: Biometric / spoofed identity fusion
    for i in range(10):
        adversarial_data.append({
            "amount": 1500.0,
            "amount_to_avg_ratio": 3.1, # Moderate increase
            "tx_count": 2, 
            "geo_distance": 1.0, 
            "is_new_device": 1, # New device
            "biometric_score": 0.95, # Perfect human emulation (e.g. bypasses bot-detector)
            "isFraud": 1
        })
        
    df_adv = pd.DataFrame(adversarial_data)
    return df_adv

def simulate_red_vs_blue():
    if not os.path.exists(XGB_PATH):
        print(f"[-] Base XGBoost model not found at {XGB_PATH}. Run fast_lane/train_fast.py first.")
        return

    # Load current model (Blue Team)
    with open(XGB_PATH, "rb") as f:
        blue_model = pickle.load(f)

    # Generate Red Team attacks
    df_adv = generate_adversarial_patterns()
    X_adv = df_adv.drop(columns=["isFraud"])
    y_adv = df_adv["isFraud"]

    # Evaluate Blue Team defense on Red Team attacks (Before retraining)
    preds = blue_model.predict(X_adv)
    probs = blue_model.predict_proba(X_adv)[:, 1]
    
    missed_count = sum(1 for p in preds if p == 0)
    detection_rate_before = (len(preds) - missed_count) / len(preds) * 100.0
    
    print("\n" + "="*50)
    print(" RED VS. BLUE SIMULATION (BEFORE RETRAINING)")
    print("="*50)
    print(f"Total Red Team Evasive Attacks: {len(preds)}")
    print(f"Blue Team Detected: {len(preds) - missed_count} attacks")
    print(f"Blue Team Missed: {missed_count} attacks (False Negatives)")
    print(f"Initial Detection Rate: {detection_rate_before:.2f}%")
    print("="*50 + "\n")

    # Retrain Blue Team to adapt (Active Defense)
    print("[*] Retraining Blue Team model on adversarial feedback loop...")
    
    # Load a portion of original dataset
    try:
        X_orig, y_orig, _ = load_data_and_preprocess(num_rows=50000)
    except Exception as e:
        print(f"[-] Could not load original training data: {e}. Retraining on feedback loop only.")
        X_orig, y_orig = pd.DataFrame(), pd.Series()
        
    # Append adversarial examples to dataset to patch the security gap
    X_new = pd.concat([X_orig, X_adv], ignore_index=True)
    y_new = pd.concat([y_orig, y_adv], ignore_index=True)
    
    # Retrain model
    scale_pos_weight = (len(y_new) - sum(y_new)) / sum(y_new)
    retrained_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    retrained_model.fit(X_new, y_new)

    # Evaluate After Retraining
    preds_after = retrained_model.predict(X_adv)
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

    # Save the retrained model (overwrite original)
    with open(XGB_PATH, "wb") as f:
        pickle.dump(retrained_model, f)
    print(f"[+] Adapted Blue Team model successfully saved to {XGB_PATH}")

    # Call FastAPI reload endpoint to hot-swap model in memory
    try:
        response = requests.post(RETRAIN_API, json={}, timeout=2.0)
        if response.status_code == 200:
            print("[+] FastAPI decision engine notified and model reloaded in-memory.")
        else:
            print(f"[-] API retrain notification returned code {response.status_code}")
    except requests.exceptions.RequestException:
        print("[!] FastAPI API is currently offline. Retrained weights will load on next API startup.")

if __name__ == "__main__":
    simulate_red_vs_blue()
