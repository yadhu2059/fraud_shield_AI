import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# Try to import ONNX exporters
try:
    import onnxmltools
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_SUPPORTED = True
except ImportError:
    ONNX_SUPPORTED = False

CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "PS_20174392719_1491204439457_log.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def load_data_and_preprocess(num_rows=200000):
    if not os.path.exists(CSV_PATH):
        print(f"[!] Kaggle dataset CSV not found at {CSV_PATH}.")
        print("[*] Generating synthetic transaction backup to compile base models...")
        np.random.seed(42)
        num_synthetic = 10000
        accounts = [f"C{np.random.randint(1000, 1500)}" for _ in range(num_synthetic)]
        dests = [f"C{np.random.randint(2000, 2500)}" for _ in range(num_synthetic)]
        amounts = np.random.exponential(scale=200.0, size=num_synthetic)
        is_fraud = np.random.choice([0, 1], size=num_synthetic, p=[0.99, 0.01])
        amounts = np.where(is_fraud == 1, amounts * 15.0, amounts)
        
        df = pd.DataFrame({
            "step": np.random.randint(1, 100, size=num_synthetic),
            "type": np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], size=num_synthetic),
            "amount": amounts,
            "nameOrig": accounts,
            "nameDest": dests,
            "isFraud": is_fraud
        })
    else:
        print(f"[*] Loading first {num_rows} records from {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH, nrows=num_rows)
    
    # Feature Engineering
    print("[*] Performing feature engineering...")
    
    # Sort by step/time
    df = df.sort_values(by="step")
    
    # Compute rolling average per account
    # In live serving this comes from Redis, but during training we simulate it
    df["avg_amount"] = df.groupby("nameOrig")["amount"].transform(lambda x: x.expanding().mean().shift(1)).fillna(df["amount"])
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount"] + 1e-5)
    
    # Transaction counts (velocity)
    df["tx_count"] = df.groupby("nameOrig").cumcount()
    
    # Geolocation simulation (generate random coordinate offsets to create distances)
    # We will simulate geolocation shifts
    np.random.seed(42)
    df["latitude"] = 28.6139 + np.random.normal(0, 0.05, len(df))
    df["longitude"] = 77.2090 + np.random.normal(0, 0.05, len(df))
    
    # Shift geo to find distance
    df["last_lat"] = df.groupby("nameOrig")["latitude"].shift(1).fillna(df["latitude"])
    df["last_lng"] = df.groupby("nameOrig")["longitude"].shift(1).fillna(df["longitude"])
    
    # Haversine distance in km
    lat1, lon1, lat2, lon2 = np.radians(df["last_lat"]), np.radians(df["last_lng"]), np.radians(df["latitude"]), np.radians(df["longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    df["geo_distance"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
    
    # Device fingerprint simulation
    # Simulate a device change flag
    df["device_id"] = df["nameOrig"].apply(lambda x: f"dev_{hash(x)%1000}")
    df["last_device_id"] = df.groupby("nameOrig")["device_id"].shift(1).fillna(df["device_id"])
    df["is_new_device"] = (df["device_id"] != df["last_device_id"]).astype(int)
    
    # Biometrics
    # In fraud transactions, biometrics (such as speed/key intervals) tend to be anomalous
    # Let's simulate a biometric anomaly score where fraud has lower (more suspicious) scores
    df["biometric_score"] = np.where(df["isFraud"] == 1, np.random.uniform(0.1, 0.6, len(df)), np.random.uniform(0.7, 0.99, len(df)))
    
    # Define features
    feature_cols = [
        "amount", 
        "amount_to_avg_ratio", 
        "tx_count", 
        "geo_distance", 
        "is_new_device", 
        "biometric_score"
    ]
    
    X = df[feature_cols]
    y = df["isFraud"]
    
    return X, y, feature_cols

def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    X, y, feature_cols = load_data_and_preprocess()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[*] Training dataset shape: {X_train.shape} (Fraud rate: {y_train.mean():.4f})")
    
    # 1. Train XGBoost
    print("[*] Training XGBoost Classifier...")
    # Using scale_pos_weight since fraud is highly imbalanced
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    print(f"[+] XGBoost train score: {xgb_model.score(X_train, y_train):.4f}")
    print(f"[+] XGBoost test score: {xgb_model.score(X_test, y_test):.4f}")
    
    # Save XGBoost
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"[+] XGBoost saved to {xgb_path}")

    # 2. Train Isolation Forest (unsupervised anomaly detection)
    print("[*] Training Isolation Forest...")
    # Train only on normal transactions to learn normal behavior profile
    X_normal = X_train[y_train == 0]
    if len(X_normal) == 0:
        X_normal = X_train
    iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iforest.fit(X_normal)
    
    # Save Isolation Forest
    iforest_path = os.path.join(MODEL_DIR, "iforest_model.pkl")
    with open(iforest_path, "wb") as f:
        pickle.dump(iforest, f)
    print(f"[+] Isolation Forest saved to {iforest_path}")

    # Try exporting to ONNX if support is installed
    if ONNX_SUPPORTED:
        try:
            print("[*] Converting models to ONNX format...")
            
            # Convert XGBoost to ONNX
            initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
            onnx_xgb = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_type)
            onnx_xgb_path = os.path.join(MODEL_DIR, "xgb_model.onnx")
            onnxmltools.utils.save_model(onnx_xgb, onnx_xgb_path)
            print(f"[+] XGBoost ONNX exported successfully to {onnx_xgb_path}")
            
            # Convert Isolation Forest to ONNX
            onnx_iforest = convert_sklearn(iforest, initial_types=initial_type)
            onnx_if_path = os.path.join(MODEL_DIR, "iforest_model.onnx")
            with open(onnx_if_path, "wb") as f:
                f.write(onnx_iforest.SerializeToString())
            print(f"[+] Isolation Forest ONNX exported successfully to {onnx_if_path}")
            
        except Exception as e:
            print(f"[-] ONNX export failed: {e}. Standard pickle models will be used.")
    else:
        print("[*] ONNX conversion libraries not installed. Skipping ONNX export. Models saved as Pickles.")

if __name__ == "__main__":
    train_models()
